import abc
import GPUtil
import typing as ty

import torch

from ...module_translation_handler.ver2.module_base import (
    BaseTranslationModelHandlerVer2,
    TranslationResultContainer,
    EvaluationTargetTranslationPair)

OPTION_EMBEDDING_LAYERS = [
    "decoder.embed_tokens",
]



class ReturnTuple_method_extract_hidden_states(ty.NamedTuple):
    source_text: str
    translated_text: str
    encoder_layer2states: ty.Dict[int, torch.Tensor]
    decoder_layer2states: ty.Dict[int, torch.Tensor]



class BaseVectorExtractorVer2(object, metaclass=abc.ABCMeta):
    def __init__(self,
                 translation_handler: BaseTranslationModelHandlerVer2):
        self.translation_handler = translation_handler

    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]

    def get_all_possible_layers(self) -> ty.Tuple[ty.List[str], ty.List[str]]:
        raise NotImplementedError()
        
    # -----------------------------------------------
    # interface of vector extraction using the teacher forcing 

    def get_hidden_vectors_teacher_forcing(self):
        pass

    # -----------------------------------------------
    # interface of extraction and translation at the same time.

    def translation_beam_search(self,
                              input_text: EvaluationTargetTranslationPair,
                              temperature: float = 1.0,
                              max_len_a: float = 0.0,
                              man_len_b: int = 200,
                              target_layers_extraction: ty.Optional[ty.List[str]] = None) -> TranslationResultContainer:
        res = self.translation_handler.translate_beam_search(
            input_text=input_text,
            temperature=temperature,
            max_len_a=max_len_a,
            max_len_b=man_len_b,
            target_layers_extraction=target_layers_extraction
        )
        return res

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
        res = self.translation_handler.translate_sample_multiple_times(
            input_text=input_text,
            temperature=temperature,
            n_sampling=n_sampling,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            n_max_attempts=n_max_attempts,
            batch_size=batch_size,
            target_layers_extraction=target_layers_extraction,
            is_sampling_in_iteration=is_sampling_in_iteration,
            is_auto_recovery_sampling=is_auto_recovery_sampling,
        )

        return res
