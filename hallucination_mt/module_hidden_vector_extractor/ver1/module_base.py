import abc
import GPUtil
import typing as ty

import torch

OPTION_EMBEDDING_LAYERS = [
    "decoder.embed_tokens",
]



class ReturnTuple_method_extract_hidden_states(ty.NamedTuple):
    source_text: str
    translated_text: str
    encoder_layer2states: ty.Dict[int, torch.Tensor]
    decoder_layer2states: ty.Dict[int, torch.Tensor]



class BaseVectorExtractor(object, metaclass=abc.ABCMeta):
    pass

    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]

    def get_all_possible_layers(self) -> ty.Tuple[ty.List[str], ty.List[str]]:
        raise NotImplementedError()
    
    def extract_encoder_output_teacher_forcing(self):
        raise NotImplementedError()

    def extract_hidden_states(self):
        raise NotImplementedError()

    def extract_encoder_output(self):
        raise NotImplementedError()

    def extract_word_embeddings_batch(self,
                                      seq_sentence: ty.List[str],
                                      target_lang: ty.Optional[str] = None,
                                      seq_token_id_tensor: ty.Optional[ty.List[torch.Tensor]] = None,
                                    ) -> ty.List[torch.Tensor]:
        raise NotImplementedError("The method is not implemented.")
