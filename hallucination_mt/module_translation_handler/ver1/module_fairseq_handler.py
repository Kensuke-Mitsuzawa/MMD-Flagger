import typing as ty
import torch
import logging
import tempfile
import random
from pathlib import Path

import zlib
import pickle

import GPUtil

from fairseq.hub_utils import GeneratorHubInterface

from .module_base import BaseTranslationModelHandler
from ..ver2.module_base import EvaluationTargetTranslationPair

from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ...exceptions import ParameterSettingException

from ..ver2.module_base import TranslationResultContainer


module_logger = logging.getLogger(__name__)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())



class DecodedTranslationObject(ty.NamedTuple):
    source_tensor_tokens: torch.Tensor  # tensor of token-ids
    target_tensor_tokens: torch.Tensor  # tensor of token-ids
    tensor_attention: torch.Tensor  # tensor of attention weights (source-len x target-len)
    score: float  # ????
    positional_scores: torch.Tensor  # tensor of scores (target-len)
    target_text: str  # decoded text

    def __str__(self):
        return f"Translation: {self.target_text}"


class FaiseqTranslationModelHandler(BaseTranslationModelHandler):
    def __init__(self,
                 model_encoder_decoder_mt: GeneratorHubInterface,
                 n_sampling: int = 1,
                 is_sampling: bool = True,
                 sampling_topk: int = -1,
                 sampling_topp: float = -1.0,     
                 max_len_a: float = 0.0,
                 max_len_b: int = 200,
                 random_seed: int = -1,
                 is_select_gpu_flexible: bool = True,
                 data_format_return: str = 'ver1',
                 is_zlib_compress: bool = True,
                 is_save_convert_float16: bool = True,
                 path_cache_dir: ty.Optional[Path] = None,
                 is_use_cache: bool = True,
                 ):
        """A class for handling the Fairseq translation model.
        
        Args:
            random_seed: A random seed for the FairSeq Call. 
                If -1, the random seed is set randomly.
        """
        super().__init__()

        assert data_format_return in ('ver1', 'ver2')
        self.data_format_return = data_format_return

        self.is_use_cache = is_use_cache

        # monkey pacth of the method
        if data_format_return == 'ver1':
            self.translate_beam_search = self.translate_beam_search_ver1
        elif data_format_return == 'ver2':
            self.translate_beam_search = self.translate_beam_search_ver2
        else:
            raise ValueError()
        # end if            


        self.is_zlib_compress = is_zlib_compress
        self.is_save_convert_float16 = is_save_convert_float16
        
        if path_cache_dir is None:
            self.path_cache_dir = Path(tempfile.mkdtemp())
            self.path_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.path_cache_dir = Path(path_cache_dir)
        # end if            

        assert isinstance(model_encoder_decoder_mt, GeneratorHubInterface), "model_encoder_decoder_mt must be an instance of fairseq.hub_utils.GeneratorHubInterface"
        self.model_encoder_decoder_mt = model_encoder_decoder_mt

        self.n_sampling = n_sampling

        self.is_sampling = is_sampling

        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp

        # parameters for the generation. See the following link.
        # https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L835
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

        self.random_seed = random_seed

        self.is_select_gpu_flexible = is_select_gpu_flexible

    # ------------------------------
    # utils from ver2 (backport)

    def translate_beam_search_ver2(self,
                              input_text: EvaluationTargetTranslationPair,
                              temperature: float = 1.0,
                              max_len_a: float = 0.0,
                              max_len_b: int = 200,
                              target_layers_extraction: ty.Optional[ty.List[str]] = None) -> TranslationResultContainer:
        if self.is_use_cache:
            false_or_cache = self._is_exist_cache_or_fetch(
                input_text.sentence_id,
                tau_param=temperature,
                n_sampling=None)
            
            if isinstance(false_or_cache, TranslationResultContainer):
                return false_or_cache
            # end if
        # end if
        
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b


        if torch.cuda.is_available():
            # multiple GPUs
            if self.is_select_gpu_flexible:
                # select the less busy GPU
                device = torch.device(f"cuda:{self._get_less_busy_cuda_device()}")
            else:
                device = torch.device("cuda:0")
            # end if
        else:
            device = torch.device("cpu")    
        # end if

        fairseq_interface = self.model_encoder_decoder_mt.to(device)

        tensor_source_tokens = fairseq_interface.encode(input_text.source)
        
        out = fairseq_interface.generate(
            tensor_source_tokens, 
            temperature=temperature, 
            sampling=False, 
            beam=5,
            max_len_a=self.max_len_a,
            max_len_b=self.max_len_b)[0]
        
        args_translation = dict(
            temperature=temperature, 
            sampling=False, 
            beam=5,
            max_len_a=self.max_len_a,
            max_len_b=self.max_len_b            
        )

        tensor_tokens = out['tokens'].cpu()

        translation_text = fairseq_interface.decode(tensor_tokens)  # type: ignore

        # tensor_attention = out['attention'].cpu()
        score = out['score'].cpu()
        positional_scores = out['positional_scores'].cpu()
        tensor_tokens = tensor_tokens.cpu()

        # vector extraction
        with torch.no_grad():
            embeddings = fairseq_interface.models[0].decoder.embed_tokens(out['tokens'])
        # end with

        return_obj = TranslationResultContainer(
            source_text=input_text.source,
            translation_text=translation_text,
            source_language='NA',
            target_language='NA',
            source_tensor_tokens=tensor_source_tokens.cpu(),
            target_tensor_tokens=tensor_tokens.cpu(),
            log_probability_score=positional_scores,
            dict_layer_embeddings={'decoder.word_embedding': embeddings},
            argument_translation_conditions=args_translation
        )        
        # dict_container = _container._asdict()
        # dict_container['dict_layer_embeddings'] = {'decoder.word_embedding': embeddings}
        # _container_updated = TranslationResultContainer(**dict_container)
        # # end for
        if self.is_use_cache:
            self._save_cache(
                sentence_id=input_text.sentence_id, 
                translation_obj=return_obj,
                tau_param=temperature,
                n_sampling=None)
        # end if


        return return_obj

    def translation_stochatstic_sampling(self,
                                        input_text: EvaluationTargetTranslationPair,
                                        temperature: float,
                                        n_sampling: int,
                                        max_len_a: float,
                                        max_len_b: int,
                                        n_max_attempts: int,
                                        batch_size: int,
                                        target_layers_extraction: ty.Optional[ty.List[str]] = None,
                                        is_sampling_in_iteration: bool = False,
                                        is_auto_recovery_sampling: bool = True,
                                        ) -> ty.List[TranslationResultContainer]:
        if self.is_use_cache:
            exists_or_cache = self._is_exist_cache_or_fetch(
                input_text.sentence_id,
                tau_param=temperature,
                n_sampling=n_sampling)
            if isinstance(exists_or_cache, list):
                return exists_or_cache
            # end if        
        # end if
                
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

        seq_container_translation = self.sample_multiple_times(
            input_text=input_text.source, 
            temperature=temperature, 
            n_sampling=n_sampling)

        seq_container_updated = []
        for _container in seq_container_translation:
            with torch.no_grad():
                token_tensor = torch.tensor(_container.target_tensor_tokens)  # Batch size of 1
                model_encoder_decoder_mt = model_encoder_decoder_mt.to(torch.device('cpu'))
                embeddings = model_encoder_decoder_mt.models[0].decoder.embed_tokens(token_tensor)
            # end with
            dict_container = _container._asdict()
            dict_container['dict_layer_embeddings'] = {'decoder.word_embedding': embeddings}
            _container_updated = TranslationResultContainer(**dict_container)
            seq_container_updated.append(_container_updated)
        # end for

        if self.is_use_cache:
            self._save_cache(
                sentence_id=input_text.sentence_id, 
                tau_param=temperature,
                n_sampling=n_sampling,
                translation_obj=seq_container_updated)
        # end if

        return seq_container_updated    

    def get_all_possible_layers(self) -> ty.Tuple[ty.List[str], ty.List[str]]:
        return [], [self._get_decoder_word_embedding_layer_name()]

    def _get_decoder_word_embedding_layer_name(self) -> str:
        return "decoder.word_embedding"
    
    def _get_cache_file_name(self, sentene_id: str, is_zlib_compress: bool) -> str:
        if is_zlib_compress:
            return f'{sentene_id}.pkl.zlib'
        else:
            return f'{sentene_id}.pt'

    def _generate_cache_file_path(self, 
                                  sentene_id: str,
                                  tau_parameter: float,
                                  n_sampling: ty.Optional[int],
                                  is_zlib_compress: bool = True
                                  ) -> Path:
        _file_name = self._get_cache_file_name(sentene_id, is_zlib_compress=is_zlib_compress)

        if n_sampling is None:
            return self.path_cache_dir / self.__class__.__name__.__str__() / 'beam' / str(tau_parameter) / _file_name
        else:
            assert n_sampling is not None
            return self.path_cache_dir / self.__class__.__name__.__str__() / 'stochastic' / str(tau_parameter) / str(n_sampling) / _file_name
        # end if

    def _save_cache(self, 
                    sentence_id: str,
                    tau_param: float,
                    translation_obj: ty.Union[TranslationResultContainer, ty.List[TranslationResultContainer]],
                    n_sampling: ty.Optional[int]):
        if self.is_save_convert_float16:
            # converting float32 object into float16.
            if isinstance(translation_obj, TranslationResultContainer):
                translation_obj = translation_obj.convert_embedding_float16()
            else:
                translation_obj = [o.convert_embedding_float16() for o in translation_obj]
            # end if
        # end if

        if isinstance(translation_obj, TranslationResultContainer):
            _path_file = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=None, is_zlib_compress=self.is_zlib_compress)
            _obj = translation_obj._asdict()
        elif isinstance(translation_obj, list):
            _path_file = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=self.is_zlib_compress)
            _obj = [o._asdict() for o in translation_obj]
        else:
            raise TypeError()
        # end if
            
        _path_file.parent.mkdir(parents=True, exist_ok=True)

        if self.is_zlib_compress:
            pickled_data = pickle.dumps(_obj)
            compressed_data_zlib = zlib.compress(pickled_data)
            with open(_path_file, "wb") as f:
                f.write(compressed_data_zlib)
            # end with
        else:
            torch.save(_obj, _path_file)
        # end if

    def _load_cache(self, 
                    sentence_id: str,
                    tau_param: float,
                    n_sampling: ty.Optional[int]                    
                    ) -> ty.Optional[ty.Union[TranslationResultContainer, ty.List[TranslationResultContainer]]]:
        _path_file_zlib = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=True)
        _path_file_pt = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=False)

        try:
            if _path_file_zlib.exists():
                with _path_file_zlib.open('rb') as f:
                    obj_saved = pickle.loads(zlib.decompress(f.read()))
            elif _path_file_pt.exists():
                obj_saved = torch.load(_path_file_pt)
            else:
                return None
            # end with
        except (zlib.error, IOError) as e:
            # the cache file is broken.
            return None
        # end if
        
        if isinstance(obj_saved, list):
            obj_cache = [TranslationResultContainer(**o) for o in obj_saved]
        else:
            obj_cache = TranslationResultContainer(**obj_saved)
        # end if
        return obj_cache
    
    def _is_exist_cache(self, 
                        sentence_id: str,
                        tau_param: float,
                        n_sampling: ty.Optional[int]) -> ty.Optional[Path]:
        _path_file = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=True)
        if _path_file.exists():
            return _path_file
        # end if
        _path_file = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=False)
        if _path_file.exists():
            return _path_file
        # end if
        #
        return None        

    def _is_exist_cache_or_fetch(self, 
                                 sentence_id: str,
                                 tau_param: float,
                                 n_sampling: ty.Optional[int]
                                 ) -> ty.Union[bool, TranslationResultContainer, ty.List[TranslationResultContainer]]:
        _path_file = self._is_exist_cache(sentence_id, tau_param=tau_param, n_sampling=n_sampling)
        if _path_file is None:
            return False
        # end if
        
        if _path_file.exists():
            _obj = self._load_cache(sentence_id, tau_param=tau_param, n_sampling=n_sampling)
            if _obj is None:
                return False
            else:
                return _obj
        else:
            return False

    # ------------------------------
    # Sampling

    def _sampling_multi_input(self,
                              fairseq_interface: GeneratorHubInterface,
                              tensor_source_tokens: torch.Tensor,
                              temperature: float,
                              n_sampling: int) -> ty.List[ty.Dict]:
        seq_input_tensor = [tensor_source_tokens] * n_sampling

        with torch.random.fork_rng():
            if self.random_seed != -1:
                torch.manual_seed(self.random_seed)
                torch.cuda.manual_seed_all(self.random_seed)  # if you are using multi-GPU.
            else:
                seed = random.randint(0, 9999999 - 1)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            # end if

            translations = fairseq_interface.generate(
                tokenized_sentences=seq_input_tensor,
                sampling=self.is_sampling,
                temperature=temperature,
                sampling_topk=self.sampling_topk,
                sampling_topp=self.sampling_topp,
                beam=1,
                max_len_a=self.max_len_a,
                max_len_b=self.max_len_b
                )
            output_stack = translations
        # end with

        return output_stack

    def _sampling_single_input(self,
                               fairseq_interface: GeneratorHubInterface,
                               tensor_source_tokens: torch.Tensor,
                               temperature: float,
                               n_sampling: int,
                               n_max_attempts: int = 10) -> ty.List[ty.Dict]:
        output_stack = []
        i_error_attempt = 0
        with torch.random.fork_rng():
            if self.random_seed != -1:
                torch.manual_seed(self.random_seed)
                torch.cuda.manual_seed_all(self.random_seed)  # if you are using multi-GPU.
            else:
                seed = random.randint(0, 2**32 - 1)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            # end if
            while len(output_stack) < n_sampling:
                try:
                    translations = fairseq_interface.generate(
                        tokenized_sentences=tensor_source_tokens,  # type: ignore
                        sampling=self.is_sampling,
                        temperature=temperature,
                        sampling_topk=self.sampling_topk,
                        sampling_topp=self.sampling_topp,
                        beam=1,
                        max_len_a_mt=self.max_len_a,
                        max_len_b_mt=self.max_len_b)
                except (AssertionError, RuntimeError) as e:
                    if i_error_attempt >= n_max_attempts:
                        error_message = (
                                f"Exceeded the maximum number of attempts: {n_max_attempts}",
                                f"Exception: {e}",
                                f"With the temperature paramater = {temperature}",
                                f"Source Text: {tensor_source_tokens}"
                        )
                        module_logger.error(error_message)
                        raise ParameterSettingException(error_message)
                    else:
                        i_error_attempt += 1
                        continue
                    # end if
                else:
                    output_stack.append(translations)
                # end try
            # end while
        # end with
        return output_stack
    # end def
    
    def _call_fairseq_interface(self, 
                                tensor_source_tokens: torch.Tensor,
                                fairseq_interface: GeneratorHubInterface,
                                temperature: float,
                                n_sampling: int,
                                is_sampling_in_iteration: bool = False,
                                is_auto_recovery_sampling: bool = True,
                                n_max_attempts: int = 100) -> ty.List[ty.Dict]:
        """Simply, I call the fairseq interface to generate translations.
        This interface has dedicated procedures for calling the fairseq translation model since the interface often causes assertion errors when the temperature is a small value.
        See the description of `is_auto_recovery_sampling` for the details.

        Args:
            is_auto_recovery_sampling: If True, the function tries to recover the sampling process when the assertion error occurs.
                It switches the sampling method to the iteration-based sampling automatically when this method encounters the assertion error.
            n_max_attempts: The maximum number of attempts to recover the sampling process.
                When the attemtps exceed this value, the function raises an exception.
        """
        with torch.no_grad():
            # Note: Possible `generate` options are defined at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810 
            # `generate` method executes `inference_step` method of `TranslationTask` class.
            # `**kwargs` arguments are passed to `build_generator` method of `TranslationTask` class first,
            # and then, the generator object is passed to the `inference_step` method.
            # See `build_generator` API at here: https://fairseq.readthedocs.io/en/latest/tasks.html#fairseq.tasks.FairseqTask.build_generator
            # The args object is `fairseq.dataclass.configs.GenerationConfig`.
            # The `GenerationConfig` definition is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810
            
            
            # Note about the `generate` outcomes.
            # The outcome comes from `generate` method of `fairseq.sequence_generator.SequenceGenerator`: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L189
            # The outcome object is from the method `_generate`: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L206
            # However, no documentations available.
            # The outcome object seems to be: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L717
            # I list-up keys of the object: 
            # - `tokens`: The tokenized output Torch.Tensor.
            # - `score`: ????.
            # - `attention`: The attention weights, (src_len x tgt_len).
            # - `alignment`: ?????
            # - `positional_scores`: The score. The tensor size is the same as `tokens`.
                # The definition seems to be: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L668-L672

            if is_sampling_in_iteration:
                output_stack = self._sampling_single_input(
                    fairseq_interface=fairseq_interface,
                    tensor_source_tokens=tensor_source_tokens,
                    temperature=temperature,
                    n_max_attempts=n_max_attempts,
                    n_sampling=n_sampling)
            else:
                try:
                    output_stack = self._sampling_multi_input(
                        fairseq_interface=fairseq_interface,
                        tensor_source_tokens=tensor_source_tokens,
                        temperature=temperature,
                        n_sampling=n_sampling)
                except (AssertionError, RuntimeError) as e:
                    if is_auto_recovery_sampling:
                        module_logger.warning(f"Assertion error occurred: {e}")
                        output_stack = self._sampling_single_input(
                            fairseq_interface=fairseq_interface,
                            tensor_source_tokens=tensor_source_tokens,
                            temperature=temperature,
                            n_max_attempts=n_max_attempts,
                            n_sampling=n_sampling)
                    else:
                        raise e
                    # end if
                # end try-except
            # end if
        # end with

        return output_stack

    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]
    
    # ------------------------------
    # Interface

    def translate_beam_search_ver1(self,
                            input_text: str,
                            temperature: float) -> ty.Union[DecodedTranslationObject, TranslationResultContainer]:
        """Translate the input text with beam search.
        """
        if torch.cuda.is_available():
            # multiple GPUs
            if self.is_select_gpu_flexible:
                # select the less busy GPU
                device = torch.device(f"cuda:{self._get_less_busy_cuda_device()}")
            else:
                device = torch.device("cuda:0")
            # end if
        else:
            device = torch.device("cpu")    
        # end if

        fairseq_interface = self.model_encoder_decoder_mt.to(device)

        tensor_source_tokens = fairseq_interface.encode(input_text)
        
        out = fairseq_interface.generate(
            tensor_source_tokens, 
            temperature=temperature, 
            sampling=False, 
            beam=5,
            max_len_a=self.max_len_a,
            max_len_b=self.max_len_b)[0]
        
        args_translation = dict(
            temperature=temperature, 
            sampling=False, 
            beam=5,
            max_len_a=self.max_len_a,
            max_len_b=self.max_len_b            
        )

        tensor_tokens = out['tokens'].cpu()

        translation_text = fairseq_interface.decode(tensor_tokens)  # type: ignore

        tensor_attention = out['attention'].cpu()
        score = out['score'].cpu()
        positional_scores = out['positional_scores'].cpu()
        tensor_tokens = tensor_tokens.cpu()

        del fairseq_interface

        if self.data_format_return == 'ver1':
            return DecodedTranslationObject(
                source_tensor_tokens=tensor_source_tokens,
                target_tensor_tokens=tensor_tokens,
                tensor_attention=tensor_attention,
                score=score,
                positional_scores=positional_scores,
                target_text=translation_text
            )
        elif self.data_format_return == 'ver2':
            return_obj = TranslationResultContainer(
                source_text=input_text,
                translation_text=translation_text,
                source_language='NA',
                target_language='NA',
                source_tensor_tokens=tensor_source_tokens.cpu(),
                target_tensor_tokens=tensor_tokens.cpu(),
                log_probability_score=positional_scores,
                dict_layer_embeddings=None,
                argument_translation_conditions=args_translation
            )

            return return_obj
        else:
            raise NotImplementedError()

    def sample_multiple_times(
            self,
            input_text: str,
            temperature: float,
            n_sampling: int,
            is_sampling_in_iteration: bool = False,
            n_max_attempts: int = 100) -> ty.Union[ty.List[TranslationResultContainer], ty.List[DecodedTranslationObject]]:
        """A custom function to sample multiple times with the same input text.
        This function conducts tokenization just one time.
        
        Args:
            is_sampling_in_iteration: If True, the sampling is executed in the iteration.
                This flag is for saving the RAM or GPU memory.
                However, the execution speed will be slower.
        """
        # Check the device
        if torch.cuda.is_available():
            # multiple GPUs
            if self.is_select_gpu_flexible:
                # select the less busy GPU
                device = torch.device(f"cuda:{self._get_less_busy_cuda_device()}")
            else:
                device = torch.device("cuda:0")
            # end if
        else:
            device = torch.device("cpu")    
        # end if

        fairseq_interface = self.model_encoder_decoder_mt.to(device)

        # The `encode` method is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/hub_utils.py#L246-L249
        # The method does: tokenize, apply_bpe, and binarize.
        tensor_source_tokens = fairseq_interface.encode(input_text)

        output_stack = self._call_fairseq_interface(
            tensor_source_tokens=tensor_source_tokens,
            fairseq_interface=fairseq_interface,
            temperature=temperature,
            is_sampling_in_iteration=is_sampling_in_iteration,
            n_max_attempts=n_max_attempts,
            n_sampling=n_sampling)

        # decoding from the token-id -> text
        seq_decoded_objects = []
        for __object in output_stack:
            assert len(__object) == 1, f"Unexpected length of the output: {len(__object)}"
            assert len(__object[0]) > 0, f"Unexpected length of the output: {len(__object[0])}"
            assert isinstance(__object[0], dict), f"Unexpected type of the output: {type(__object[0])}"
            __dict_obj = __object[0]
            tensor_tokens = __dict_obj['tokens'].to(device)

            translation_text = fairseq_interface.decode(tensor_tokens)  # type: ignore

            tensor_attention = __dict_obj['attention'].cpu()
            score = __dict_obj['score'].cpu()
            positional_scores = __dict_obj['positional_scores'].cpu()
            tensor_tokens = tensor_tokens.cpu()

            if self.data_format_return == 'ver1':
                __decoded_obj = DecodedTranslationObject(
                    source_tensor_tokens=tensor_source_tokens,
                    target_tensor_tokens=tensor_tokens,
                    tensor_attention=tensor_attention,
                    score=score.item(),
                    positional_scores=positional_scores,
                    target_text=translation_text)
                # Note: I guess `score` is a perplexity value.
                # Since `score = sum(positional_scores) / len(positional_scores)`
                seq_decoded_objects.append(__decoded_obj)
            elif self.data_format_return == 'ver2':
                return_obj = TranslationResultContainer(
                    source_text=input_text,
                    translation_text=translation_text,
                    source_language='NA',
                    target_language='NA',
                    source_tensor_tokens=tensor_source_tokens.cpu(),
                    target_tensor_tokens=tensor_tokens.cpu(),
                    log_probability_score=positional_scores,
                    dict_layer_embeddings=None,
                    argument_translation_conditions={}
                )
                seq_decoded_objects.append(return_obj)
            else:
                raise NotImplementedError()
            # end if
        # end for

        return seq_decoded_objects
