import typing as ty
import abc
import re
import pickle
import zlib
import tempfile

from pathlib import Path

from ...commons.data_models import EvaluationTargetTranslationPair

import torch

import GPUtil


class TranslationResultContainer(ty.NamedTuple):
    """A data container of holding a translation"""
    source_text: str
    translation_text: str

    source_language: str
    target_language: str

    source_tensor_tokens: torch.Tensor  # tensor of token-ids
    target_tensor_tokens: torch.Tensor  # tensor of token-ids
    
    log_probability_score: ty.Optional[float]  # log-probability score

    dict_layer_embeddings: ty.Optional[ty.Dict[str, torch.Tensor]] = None  # dictionary object of holding {layer-name: tensor}
    argument_translation_conditions: ty.Optional[ty.Dict[str, ty.Any]] = None  # a dictionary object of holding beam-search length, random-seed etc.

    def __str__(self):
        return f"Source: {self.source_text}, Translation: {self.translation_text}"

    def convert_embedding_float16(self) -> "TranslationResultContainer":
        if self.dict_layer_embeddings is None:
            return self
        else:
            obj_fields = self._asdict()
            for _key, _v in obj_fields['dict_layer_embeddings'].items():
                assert isinstance(_v, torch.Tensor)
                _tensor_float16 = _v.to(torch.float16).cpu()
                obj_fields['dict_layer_embeddings'][_key] = _tensor_float16
            # end for
        # end if

        tuple_reconstruct = TranslationResultContainer(**obj_fields)
        return tuple_reconstruct
    

class BaseTranslationModelHandlerVer2(object, metaclass=abc.ABCMeta):
    def __init__(self, 
                 path_cache_dir: ty.Optional[Path],
                 is_use_cache: bool = True,
                 is_zlib_compress: bool = True,
                 is_save_convert_float16: int = False):
        if path_cache_dir is None:
            self.path_cache_dir = Path(tempfile.mkdtemp())
        else:   
            self.path_cache_dir = path_cache_dir
        # end if
        self.is_use_cache = is_use_cache
        self.is_zlib_compress = is_zlib_compress

        self.is_save_convert_float16 = is_save_convert_float16
        
        if self.is_use_cache:
            (self.path_cache_dir / self.__class__.__name__.__str__()).mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]

    @staticmethod
    def _parse_layer_sequence(target_layers_extraction: ty.List[str]) -> ty.List[ty.Tuple[str, int]]:
        # parsing the `target_layers_extraction`
        _seq_target_layers = [_target_decoder for _target_decoder in target_layers_extraction if 'decoder' in _target_decoder]
        pattern = re.compile(r'decoder.([0-9]+)')
        seq_layer_number = []
        for _target_layer_name in _seq_target_layers:
            _match_res = pattern.search(_target_layer_name)
            if _match_res is not None:
                _val = int(_match_res.group(1))
                seq_layer_number.append([_target_layer_name, _val])
            # end if
        # end for

        return seq_layer_number    

    @abc.abstractmethod
    def get_all_possible_layers(self) -> ty.Tuple[ty.List, ty.List]:
        raise NotImplementedError()

    @abc.abstractmethod
    def translate_beam_search(self,
                              input_text: EvaluationTargetTranslationPair,
                              temperature: float = 1.0,
                              max_len_a: float = 0.0,
                              max_len_b: int = 200,
                              target_layers_extraction: ty.Optional[ty.List[str]] = None
                              ) -> TranslationResultContainer:
        raise NotImplementedError()

    @abc.abstractmethod
    def translate_sample_multiple_times(self,
                                        input_text: EvaluationTargetTranslationPair,
                                        temperature: float,
                                        n_sampling: int,
                                        max_len_a: float,
                                        max_len_b: int,
                                        n_max_attempts: int,
                                        batch_size: int,
                                        target_layers_extraction: ty.Optional[ty.List[str]] = None,
                                        is_sampling_in_iteration: bool = False,
                                        is_auto_recovery_sampling: bool = True
                                        ) -> ty.List[TranslationResultContainer]:
        raise NotImplementedError()
