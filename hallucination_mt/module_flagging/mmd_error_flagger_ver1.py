import typing as ty
import torch
import logging
import tqdm
import dataclasses
import json
import sys
from pathlib import Path

from fairseq.hub_utils import GeneratorHubInterface

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel

from ..commons.data_models import EvaluationTargetTranslationPair

from ..module_translation_handler.ver1.module_fairseq_handler import (
    FaiseqTranslationModelHandler,
    ParameterSettingException)
from ..module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ..guerreiro_2023_wmt.utils_models.utils import (
    load_model, 
    extract_word_embeddings,
    extract_word_embeddings_batch
)



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


class _FlaggingResult(ty.NamedTuple):
    mmd_a: float
    mmd_b: float
    is_hallucination: bool
    source: str
    target: str
    translation_population_a: ty.Optional[ty.List[str]] = None
    translation_population_b: ty.Optional[ty.List[str]] = None
    tensor_word_embedding_population_a: ty.Optional[torch.Tensor] = None
    tensor_word_embedding_population_b: ty.Optional[torch.Tensor] = None
    embedding_word_target: ty.Optional[torch.Tensor] = None


@dataclasses.dataclass
class RecordFlaggingResult:
    sentence_id: ty.Union[int, str]
    source_language_text: str
    target_language_text: str
    temperature_a: float
    temperature_b: float
    mmd_a: float
    mmd_b: float
    is_hallucination: bool
    translation_population_a: ty.Optional[ty.List[str]] = None
    translation_population_b: ty.Optional[ty.List[str]] = None

    def __post_init__(self):
        assert isinstance(self.sentence_id, (int, str)), f"Invalid type: {type(self.sentence_id)}"
        self.sentence_id = str(self.sentence_id)



class MmdErrorFlaggerVer1(object):
    def __init__(self,
                 model_encoder_decoder_mt: GeneratorHubInterface,
                 n_sampling: int,
                 temperature_low: float,
                 temperature_high: float,
                 mmd_estimator: ty.Optional[QuadraticMmdEstimator] = None,
                 seq_calibration_text: ty.Optional[ty.List[str]] = None,
                 mode_preprocess: str = "avg",
                 median_options: str = "dimensionwise",
                 median_heuristic_operation: str = "median",
                 sampling_topk: int = -1,
                 sampling_topp: float = -1.0,     
                 path_cache_dir: ty.Optional[Path] = None,
                 max_len_a: float = 0.0,
                 max_len_b: int = 200,
                 random_seed_fairseq: int = -1):
        assert isinstance(model_encoder_decoder_mt, GeneratorHubInterface), "model_encoder_decoder_mt must be an instance of fairseq.hub_utils.GeneratorHubInterface"
        self.model_encoder_decoder_mt = model_encoder_decoder_mt

        self.fairseq_handler = FaiseqTranslationModelHandler(
            is_sampling=True,
            model_encoder_decoder_mt=model_encoder_decoder_mt,
            n_sampling=n_sampling,
            sampling_topk=sampling_topk,
            sampling_topp=sampling_topp,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            random_seed=random_seed_fairseq)

        self.mode_preprocess = mode_preprocess
        self.median_options = median_options
        self.median_heuristic_operation = median_heuristic_operation

        self.n_sampling = n_sampling
        self.temperature_low = temperature_low
        self.temperature_high = temperature_high

        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp

        # parameters for the generation. See the following link.
        # https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L835
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

        path_cache_dir_low_temp, path_cache_dir_high_temp, path_cache_dir_given_translation, is_use_cache  = self._set_cache_dir(path_cache_dir)
        self.path_cache_dir_low_temp = path_cache_dir_low_temp
        self.path_cache_dir_high_temp = path_cache_dir_high_temp
        self.path_cache_dir_given_translation = path_cache_dir_given_translation
        self.is_use_cache = is_use_cache

        if mmd_estimator is None:
            assert seq_calibration_text is not None, "seq_calibration_dataset is required to initialize mmd estimator."
            self.mmd_estimator = self.__init_mmd_estimator(seq_calibration_text)
        else:
            self.mmd_estimator = mmd_estimator

    def _set_cache_dir(self, path_cache_dir: ty.Optional[Path]) -> ty.Tuple[ty.Optional[Path], ty.Optional[Path], ty.Optional[Path], bool]:
        if path_cache_dir is not None:
            # setting directories for the cache.
            path_cache_dir_low_temp = path_cache_dir / f"temperature_{self.temperature_low}"
            path_cache_dir_low_temp.mkdir(exist_ok=True, parents=True)
            path_cache_dir_high_temp = path_cache_dir / f"temperature_{self.temperature_high}"
            path_cache_dir_high_temp.mkdir(exist_ok=True, parents=True)

            path_cache_dir_given_translation = path_cache_dir / "given_translation"
            path_cache_dir_given_translation.mkdir(exist_ok=True, parents=True)
            is_use_cache = True
        else:
            path_cache_dir_low_temp = None
            path_cache_dir_high_temp = None
            path_cache_dir_given_translation = None
            is_use_cache = False
        # end if

        return path_cache_dir_low_temp, path_cache_dir_high_temp, path_cache_dir_given_translation, is_use_cache

    def _calibrate_kernel_function(
            self,
            seq_calibration_text: ty.List[str]
            ) -> QuadraticKernelGaussianKernel:
                
        _calibration_emb_tokens = extract_word_embeddings_batch(
            self.model_encoder_decoder_mt, 
            seq_calibration_text)
        # pre-processing of tensors.
        # __a_emb -> a_emb, a fixed shape
        calibration_emb_fixed = self._preprocess_tensors(_calibration_emb_tokens)

        if self.median_options == 'single':
            _is_dimension_wise = False
        elif self.median_options == 'dimensionwise':
            _is_dimension_wise = True
        else:
            raise ValueError(f"Invalid median options: {self.median_options}")
        # end if

        kernel_func_obj = QuadraticKernelGaussianKernel(
            is_dimension_median_heuristic=_is_dimension_wise,
            heuristic_operation=self.median_heuristic_operation,
            ard_weights=torch.ones(calibration_emb_fixed.shape[1])
        )
        module_logger.debug("Computing length scale using the calibration set...")
        if _is_dimension_wise:
            # TODO: there is the safe guard avoiding L2(x, x).
            tensor_length_scale = kernel_func_obj._get_median_dim(
                x=calibration_emb_fixed,
                y=calibration_emb_fixed,
                is_safe_guard_same_xy=False)
        else:
            tensor_length_scale = kernel_func_obj._get_median_single(
                x=calibration_emb_fixed,
                y=calibration_emb_fixed)
        # end if
        module_logger.debug("Done computing the length scale...")    
        assert tensor_length_scale is not None

        # set the computed length-scale to the kernel object.
        kernel_func_obj.bandwidth = torch.nn.Parameter(tensor_length_scale, requires_grad=False)
        kernel_func_obj.ard_weights = torch.nn.Parameter(torch.ones(calibration_emb_fixed.shape[1]), requires_grad=False)

        return kernel_func_obj

    def __init_mmd_estimator(self, seq_calibration_text: ty.List[str]) -> QuadraticMmdEstimator:
        kernel_func_obj = self._calibrate_kernel_function(seq_calibration_text)
        mmd_estimator = QuadraticMmdEstimator(kernel_func_obj, variance_term='sutherland_2017')

        return mmd_estimator

    def _preprocess_tensors(self,
                            seq_tensor: ty.List[torch.Tensor]
                            ) -> torch.Tensor:
        """
        Args:
            seq_tensor: The list of tensors. Each tensor is (T: number of tokens, embed_dim).

        Returns:
            torch.Tensor: The tensor is (N: the num. of documents, D_emb: embedding-size).
        """
        def mode_avg(document_tensor: torch.Tensor) -> torch.Tensor:
            """I want to compute the average of the tensor.
            
            Args:
                document_tensor: The tensor is (T: number of tokens, embed_dim).
            """
            assert len(document_tensor.shape) == 2, f"Expected 2D tensor, got {len(document_tensor.shape)}"
            return torch.mean(document_tensor, dim=0)
        # end mode_avg

        if self.mode_preprocess == "avg":
            return torch.stack([mode_avg(_t) for _t in seq_tensor], dim=0)
        else:
            raise ValueError(f"Invalid mode preprocess: {self.mode_preprocess}")

    # # ------------------------------
    # # Sampling

    # def _sampling_multi_input(self,
    #                           fairseq_interface: GeneratorHubInterface,
    #                           tensor_source_tokens: torch.Tensor,
    #                           temperature: float) -> ty.List[ty.Dict]:
    #     seq_input_tensor = [tensor_source_tokens] * self.n_sampling
    #     translations = fairseq_interface.generate(
    #         tokenized_sentences=seq_input_tensor,
    #         sampling=True,
    #         temperature=temperature,
    #         sampling_topk=self.sampling_topk,
    #         sampling_topp=self.sampling_topp,
    #         beam=1,
    #         max_len_a_mt=self.max_len_a,
    #         max_len_b_mt=self.max_len_b
    #         )
    #     output_stack = translations

    #     return output_stack

    # def _sampling_single_input(self,
    #                            fairseq_interface: GeneratorHubInterface,
    #                            tensor_source_tokens: torch.Tensor,
    #                            temperature: float,
    #                            n_max_attempts: int = 100) -> ty.List[ty.Dict]:
    #     output_stack = []
    #     i_error_attempt = 0
    #     while len(output_stack) < self.n_sampling:
    #         try:
    #             translations = fairseq_interface.generate(
    #                 tokenized_sentences=tensor_source_tokens,  # type: ignore
    #                 sampling=True,
    #                 temperature=temperature,
    #                 sampling_topk=self.sampling_topk,
    #                 sampling_topp=self.sampling_topp,
    #                 beam=1,
    #                 max_len_a_mt=self.max_len_a,
    #                 max_len_b_mt=self.max_len_b)
    #         except (AssertionError, RuntimeError) as e:
    #             if i_error_attempt >= n_max_attempts:
    #                 module_logger.error(
    #                     (
    #                         f"Exceeded the maximum number of attempts: {n_max_attempts}",
    #                         f"Exception: {e}",
    #                         f"With the temperature paramater = {temperature}",
    #                         f"Source Text: {tensor_source_tokens}"
    #                     )
    #                 )
    #                 raise e
    #             else:

    #                 i_error_attempt += 1
    #             # end if
    #             continue
    #         # end try
    #         else:
    #             output_stack.append(translations)
    #     # end while
    #     return output_stack
    # # end def
    
    # def _call_fairseq_interface(self, 
    #                             tensor_source_tokens: torch.Tensor,
    #                             fairseq_interface: GeneratorHubInterface,
    #                             temperature: float,
    #                             is_sampling_in_iteration: bool = False,
    #                             is_auto_recovery_sampling: bool = True,
    #                             n_max_attempts: int = 100) -> ty.List[ty.Dict]:
    #     """Simply, I call the fairseq interface to generate translations.
    #     This interface has dedicated procedures for calling the fairseq translation model since the interface often causes assertion errors when the temperature is a small value.
    #     See the description of `is_auto_recovery_sampling` for the details.

    #     Args:
    #         is_auto_recovery_sampling: If True, the function tries to recover the sampling process when the assertion error occurs.
    #             It switches the sampling method to the iteration-based sampling automatically when this method encounters the assertion error.
    #         n_max_attempts: The maximum number of attempts to recover the sampling process.
    #             When the attemtps exceed this value, the function raises an exception.
    #     """
    #     with torch.no_grad():
    #         # Note: Possible `generate` options are defined at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810 
    #         # `generate` method executes `inference_step` method of `TranslationTask` class.
    #         # `**kwargs` arguments are passed to `build_generator` method of `TranslationTask` class first,
    #         # and then, the generator object is passed to the `inference_step` method.
    #         # See `build_generator` API at here: https://fairseq.readthedocs.io/en/latest/tasks.html#fairseq.tasks.FairseqTask.build_generator
    #         # The args object is `fairseq.dataclass.configs.GenerationConfig`.
    #         # The `GenerationConfig` definition is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810
            
            
    #         # Note about the `generate` outcomes.
    #         # The outcome comes from `generate` method of `fairseq.sequence_generator.SequenceGenerator`: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L189
    #         # The outcome object is from the method `_generate`: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L206
    #         # However, no documentations available.
    #         # The outcome object seems to be: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L717
    #         # I list-up keys of the object: 
    #         # - `tokens`: The tokenized output Torch.Tensor.
    #         # - `score`: ????.
    #         # - `attention`: The attention weights, (src_len x tgt_len).
    #         # - `alignment`: ?????
    #         # - `positional_scores`: The score. The tensor size is the same as `tokens`.
    #             # The definition seems to be: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L668-L672

    #         if is_sampling_in_iteration:
    #             output_stack = self._sampling_single_input(
    #                 fairseq_interface=fairseq_interface,
    #                 tensor_source_tokens=tensor_source_tokens,
    #                 temperature=temperature,
    #                 n_max_attempts=n_max_attempts)
    #         else:
    #             try:
    #                 output_stack = self._sampling_multi_input(
    #                     fairseq_interface=fairseq_interface,
    #                     tensor_source_tokens=tensor_source_tokens,
    #                     temperature=temperature)
    #             except (AssertionError, RuntimeError) as e:
    #                 if is_auto_recovery_sampling:
    #                     module_logger.warning(f"Assertion error occurred: {e}")
    #                     output_stack = self._sampling_single_input(
    #                         fairseq_interface=fairseq_interface,
    #                         tensor_source_tokens=tensor_source_tokens,
    #                         temperature=temperature,
    #                         n_max_attempts=n_max_attempts)
    #                 else:
    #                     raise e
    #                 # end if
    #             # end try-except
    #         # end if
    #     # end with

    #     return output_stack

    # def _sample_multiple_times(
    #         self,
    #         input_text: str,
    #         temperature: float,
    #         is_sampling_in_iteration: bool = False) -> ty.List[DecodedTranslationObject]:
    #     """A custom function to sample multiple times with the same input text.
    #     This function conducts tokenization just one time.
        
    #     Args:
    #         is_sampling_in_iteration: If True, the sampling is executed in the iteration.
    #             This flag is for saving the RAM or GPU memory.
    #             However, the execution speed will be slower.
    #     """
    #     # Check the device
    #     if torch.cuda.is_available():
    #         # TODO: multiple GPUs
    #         device = torch.device("cuda:0")
    #     else:
    #         device = torch.device("cpu")    
    #     # end if

    #     fairseq_interface = self.model_encoder_decoder_mt.to(device)

    #     # The `encode` method is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/hub_utils.py#L246-L249
    #     # The method does: tokenize, apply_bpe, and binarize.
    #     tensor_source_tokens = fairseq_interface.encode(input_text)

    #     output_stack = self._call_fairseq_interface(
    #         tensor_source_tokens=tensor_source_tokens,
    #         fairseq_interface=fairseq_interface,
    #         temperature=temperature,
    #         is_sampling_in_iteration=is_sampling_in_iteration)

    #     # decoding from the token-id -> text
    #     seq_decoded_objects = []
    #     for __object in output_stack:
    #         assert len(__object) == 1, f"Unexpected length of the output: {len(__object)}"
    #         assert len(__object[0]) > 0, f"Unexpected length of the output: {len(__object[0])}"
    #         assert isinstance(__object[0], dict), f"Unexpected type of the output: {type(__object[0])}"
    #         __dict_obj = __object[0]
    #         tensor_tokens = __dict_obj['tokens'].to(device)

    #         translation_text = fairseq_interface.decode(tensor_tokens)  # type: ignore

    #         tensor_attention = __dict_obj['attention'].cpu()
    #         score = __dict_obj['score'].cpu()
    #         positional_scores = __dict_obj['positional_scores'].cpu()
    #         tensor_tokens = tensor_tokens.cpu()

    #         __decoded_obj = DecodedTranslationObject(
    #             source_tensor_tokens=tensor_source_tokens,
    #             target_tensor_tokens=tensor_tokens,
    #             tensor_attention=tensor_attention,
    #             score=score.item(),
    #             positional_scores=positional_scores,
    #             target_text=translation_text)
    #         # Note: I guess `score` is a perplexity value.
    #         # Since `score = sum(positional_scores) / len(positional_scores)`
    #         seq_decoded_objects.append(__decoded_obj)
    #     # end for

    #     return seq_decoded_objects

    def _compute_mmd_distance(self,
                             tensor_original_translation: torch.Tensor,
                             tensor_new_translation: torch.Tensor
                             ) -> float:
        """I want to compute the MMD distance between the original and the new translation."""
        is_same_tensor = torch.equal(tensor_original_translation, tensor_new_translation)
        if is_same_tensor:
            return 0.0
        # end if

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        # end if

        tensor_original_translation = tensor_original_translation.to(device)
        tensor_new_translation = tensor_new_translation.to(device)
        _estimator = self.mmd_estimator.to(device)

        with torch.no_grad():
            # distance_mmd = self.mmd_estimator.forward(tensor_original_translation, tensor_new_translation)
            distance_mmd = _estimator.forward(tensor_original_translation, tensor_new_translation)
        # end with

        return distance_mmd.mmd.item()
    
    def _save_cache_files(self,
                          path_cache_dir: Path,
                          sentence_id: str,
                          temperature: float,
                          translation_population: ty.List[str],
                          tensor_word_embedding: torch.Tensor,
                          source_text: ty.Optional[str] = None,
                          target_text: ty.Optional[str] = None) -> None:
        assert path_cache_dir.exists(), f"Invalid path: {path_cache_dir}"
        
        path_cache_file_json = path_cache_dir / f"{sentence_id}.json"
        path_cache_file_tensor_pt = path_cache_dir / f"{sentence_id}.pt"

        # save the cache
        dict_population = dict(population=translation_population, 
                               sentence_id=sentence_id, 
                               temperature=temperature,
                               source_text=source_text,
                               target_text=target_text,
                               max_len_a=self.max_len_a,
                               max_len_b=self.max_len_b)
        with open(path_cache_file_json, "w") as f:
            json.dump(dict_population, f)
        # end with

        dict_embedding = dict(embedding=tensor_word_embedding, 
                              sentence_id=sentence_id, 
                              temperature=temperature,
                              source_text=source_text,
                              target_text=target_text,
                              max_len_a=self.max_len_a,
                              max_len_b=self.max_len_b)
        torch.save(dict_embedding, path_cache_file_tensor_pt)


    def _flag_hallucination_one_record(self,
                            eval_target: EvaluationTargetTranslationPair,
                            population_a: ty.Optional[ty.List[str]] = None,
                            population_b: ty.Optional[ty.List[str]] = None,
                            word_emb_a: ty.Optional[torch.Tensor] = None,
                            word_emb_b: ty.Optional[torch.Tensor] = None,
                            word_emb_y: ty.Optional[torch.Tensor] = None,
                            n_max_attempts: int = 5
                            ) -> _FlaggingResult:
        """This function is for flagging hallucination in the pseudo-hallucination generation process.
        
        Argss:
            population_a: The population A for the flagging. This argument is for cache. If None, I generate the population A.
            population_b: The population B for the flagging. This argument is for cache. If None, I generate the population B.

        Exceptions:
            `ParameterSettingException`: The exception is raised when FairSeq Interface raises an exception.
            The exception is mostly due to too low temperature parameter.
        """
        if population_a is not None and population_b is not None:
            _population_a = population_a
            _population_b = population_b
        elif self.is_use_cache:
            assert self.path_cache_dir_low_temp is not None
            assert self.path_cache_dir_high_temp is not None
            assert self.path_cache_dir_given_translation is not None
            _population_a, word_emb_a = self._load_cache_files(path_cache_dir=self.path_cache_dir_low_temp, sentence_id=eval_target.sentence_id)
            _population_b, word_emb_b = self._load_cache_files(path_cache_dir=self.path_cache_dir_high_temp, sentence_id=eval_target.sentence_id)
            __, word_emb_y = self._load_cache_files(path_cache_dir=self.path_cache_dir_given_translation, sentence_id=eval_target.sentence_id)
            if _population_a is None:
                _population_a = None
            else:
                assert isinstance(_population_a, list) and all([isinstance(__s, str) for __s in _population_a]), f"Invalid cache: {self.path_cache_dir_low_temp}"
                assert isinstance(word_emb_a, torch.Tensor), f"Invalid cache: {self.path_cache_dir_low_temp}"
            # end if

            if _population_b is None:
                _population_b = None
            else:
                assert isinstance(_population_b, list) and all([isinstance(__s, str) for __s in _population_b]), f"Invalid cache: {self.path_cache_dir_high_temp}"        
                assert isinstance(word_emb_b, torch.Tensor), f"Invalid cache: {self.path_cache_dir_high_temp}"
            # end if

            if word_emb_y is not None:
                assert isinstance(word_emb_y, torch.Tensor), f"Invalid cache: {self.path_cache_dir_given_translation}"
            # end if
        else:
            _population_a = None
            _population_b = None
        # end if

        # sampling with a low temperature
        if _population_a is None:
            module_logger.debug(f"Generating a population A with a low temperature: {self.temperature_low} ...")
            container_sample_a = self.fairseq_handler.sample_multiple_times(
                input_text=eval_target.source,
                temperature=self.temperature_low,
                n_max_attempts=n_max_attempts,
                n_sampling=self.n_sampling)
            _population_a = [__obj.target_text for __obj in container_sample_a]
            assert _population_a is not None, f"Error in the input texts: {eval_target.source}"
        # end if
        if _population_b is None:
            # sampling with a high temperature
            module_logger.debug(f"Generating a population B with a high temperature: {self.temperature_high} ...")    
            # container_sample_b = self._sample_multiple_times(
            #     input_text=eval_target.source,
            #     temperature=self.temperature_high)
            container_sample_b = self.fairseq_handler.sample_multiple_times(
                input_text=eval_target.source,
                temperature=self.temperature_high,
                n_max_attempts=n_max_attempts,
                n_sampling=self.n_sampling)
            _population_b = [__obj.target_text for __obj in container_sample_b]
            assert _population_b is not None, f"Error in the input texts: {eval_target.source}"
        # end if

        if word_emb_y is None:
            # feature embedding for the target_language_text
            y_emv_unfixed = extract_word_embeddings(
                self.model_encoder_decoder_mt, 
                eval_target.target)
            assert len(y_emv_unfixed.shape) == 2, f"Expected 2D tensor, got {len(y_emv_unfixed.shape)}"
            # __y_emb -> y_emb, a fixed shape, the same shape as a_emb and b_emb
            # Note: To avoid Exception, I create a population having 2 samples.
            # TODO: I want that without dataset of having the same sample-size.
            # Note: the number of samples requires the same size. However, it should be the same value with the 1 sample...
            # y_emb = self._preprocess_tensors([y_emv_unfixed] * self.n_sampling)
            y_emb = self._preprocess_tensors([y_emv_unfixed] * 2)
        else:
            y_emb = word_emb_y
        # end if

        if word_emb_a is None:
            # feature embedding for A
            a_emb_unfixed = extract_word_embeddings_batch(
                self.model_encoder_decoder_mt, 
                _population_a)
            # pre-processing of tensors.
            # __a_emb -> a_emb, a fixed shape
            a_emb = self._preprocess_tensors(a_emb_unfixed)
        else:
            a_emb = word_emb_a
        # end if

        if word_emb_b is None:
            # feature embedding for B
            b_emb_unfixed = extract_word_embeddings_batch(
                self.model_encoder_decoder_mt, 
                _population_b)        
            # __b_emb -> b_emb, a fixed shape
            b_emb = self._preprocess_tensors(b_emb_unfixed)
        else:
            b_emb = word_emb_b
        # end if

        # compute MMD distance between y_emb and A_emb
        mmd_a = self._compute_mmd_distance(
            tensor_original_translation=y_emb,
            tensor_new_translation=a_emb)

        # compute MMD distance between y_emb and B_emb
        mmd_b = self._compute_mmd_distance(
            tensor_original_translation=y_emb,
            tensor_new_translation=b_emb)

        # flagging
        is_hallucination = mmd_b < mmd_a
        
        if self.is_use_cache:
            assert self.path_cache_dir_low_temp is not None
            assert self.path_cache_dir_high_temp is not None
            assert self.path_cache_dir_given_translation is not None
            self._save_cache_files(
                path_cache_dir=self.path_cache_dir_low_temp,
                sentence_id=eval_target.sentence_id,
                temperature=self.temperature_low,
                translation_population=_population_a,
                tensor_word_embedding=a_emb,
                source_text=eval_target.source,
                target_text=eval_target.target)
            self._save_cache_files(
                path_cache_dir=self.path_cache_dir_high_temp,
                sentence_id=eval_target.sentence_id,
                temperature=self.temperature_high,
                translation_population=_population_b,
                tensor_word_embedding=b_emb,
                source_text=eval_target.source,
                target_text=eval_target.target)
            self._save_cache_files(
                path_cache_dir=self.path_cache_dir_given_translation,
                sentence_id=eval_target.sentence_id,
                temperature=self.temperature_low,
                translation_population=[eval_target.target],
                tensor_word_embedding=y_emb)
        # end if

        # return
        return _FlaggingResult(
            mmd_a=mmd_a, 
            mmd_b=mmd_b, 
            source=eval_target.source,
            target=eval_target.target,
            is_hallucination=is_hallucination,
            translation_population_a=population_a,
            translation_population_b=population_b,
            tensor_word_embedding_population_a=a_emb,
            tensor_word_embedding_population_b=b_emb,
            embedding_word_target=y_emb)

    def _load_cache_files(self,
                          path_cache_dir: Path,
                          sentence_id: str) -> ty.Tuple[ty.Optional[ty.List[str]], ty.Optional[torch.Tensor]]:
        assert path_cache_dir is not None and path_cache_dir.exists(), f"Invalid path: {path_cache_dir}"
        __path_cache_population_low = path_cache_dir / f"{sentence_id}.json"
        if __path_cache_population_low.exists():
            module_logger.debug(f"Loaded the cache: {__path_cache_population_low}")            
            
            # loading the translated text
            _population_a_cache: ty.Dict = json.load(open(__path_cache_population_low))
            assert "population" in _population_a_cache, f"Invalid cache: {__path_cache_population_low}"
            _population_a = _population_a_cache["population"]
            assert isinstance(_population_a, list) and all([isinstance(__s, str) for __s in _population_a]), f"Invalid cache: {__path_cache_population_low}"

            # loading the word embedding
            __path_cache_population_low_pt = path_cache_dir / f"{sentence_id}.pt"
            assert __path_cache_population_low_pt.exists(), f"Invalid path: {__path_cache_population_low_pt}"
            _embedding_a = torch.load(__path_cache_population_low_pt)
            assert "embedding" in _embedding_a, f"Invalid cache: {__path_cache_population_low_pt}"
            _embedding_a = _embedding_a["embedding"]
            assert isinstance(_embedding_a, torch.Tensor), f"Invalid cache: {__path_cache_population_low_pt}"
            return _population_a, _embedding_a
        else:
            return None, None


    def flag_hallucination(self, 
                           seq_evaluation_target: ty.List[EvaluationTargetTranslationPair]
                           ) -> ty.List[RecordFlaggingResult]:
        assert self.mmd_estimator is not None, "Mmd estimator not initialized yet."
        # main loop of flagging 
        seq_flagging_results = []
        
        for __dataset_record in tqdm.tqdm(seq_evaluation_target, desc=f"Processing {__name__}", file=sys.stdout):
            __result_flagging = self._flag_hallucination_one_record(eval_target=__dataset_record)
            module_logger.debug(f'sentence-id: {__dataset_record.sentence_id}, mmd-a: {__result_flagging.mmd_a}, mmd-b: {__result_flagging.mmd_b}, is-hallucination: {__result_flagging.is_hallucination}')
            __output_obj = RecordFlaggingResult(
                sentence_id=__dataset_record.sentence_id,
                source_language_text=__dataset_record.source,
                target_language_text=__dataset_record.target,
                temperature_a=self.temperature_low,
                temperature_b=self.temperature_high,
                mmd_a=__result_flagging.mmd_a,
                mmd_b=__result_flagging.mmd_b,
                is_hallucination=__result_flagging.is_hallucination,
                translation_population_a=__result_flagging.translation_population_a,
                translation_population_b=__result_flagging.translation_population_b)
            seq_flagging_results.append(__output_obj)
        # end for
        return seq_flagging_results
