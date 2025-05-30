import typing as ty
from pathlib import Path
import logging

import numpy as np
import torch

# from fairseq.models.transformer import TransformerModel
# from fairseq.hub_utils import GeneratorHubInterface
# from fairseq.models.transformer.transformer_encoder import TransformerEncoderBase
# from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase

from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ...module_translation_handler.ver1 import FaiseqTranslationModelHandler
from ...module_translation_handler.ver2.module_base import (
    EvaluationTargetTranslationPair,
    TranslationResultContainer)
from ...module_translation_handler.ver2 import (
     FairSeqTranslationModelHandlerVer2,
     TransformersTranslationModelHandlerVer2
)
from .module_base import (BaseVectorExtractorVer2, ReturnTuple_method_extract_hidden_states)

module_logger = logging.getLogger(__name__)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())


class FairSeqVectorExtractorVer2CustomTranslationHandlerVer1(BaseVectorExtractorVer2):
    """FairSeqVectorExtractorVer2 is really really slow due to the intermediate layer prodcedures.
    I use ver1 just for the word embedding"""
    def __init__(self,
                 translation_handler: FaiseqTranslationModelHandler,
                 ) -> None:
           """
           """
           self.translation_handler = translation_handler
           self.translation_handler.data_format_return = 'ver2'
           
    def translation_beam_search(self,
                              input_text: EvaluationTargetTranslationPair,
                              temperature: float = 1.0,
                              max_len_a: float = 0.0,
                              man_len_b: int = 200,
                              target_layers_extraction: ty.Optional[ty.List[str]] = None) -> TranslationResultContainer:
        self.translation_handler.max_len_a = max_len_a
        self.translation_handler.max_len_b = man_len_b

        _container = self.translation_handler.translate_beam_search(
            input_text=input_text.source,
            temperature=temperature,
        )
        # vector extraction
        with torch.no_grad():
            token_tensor = torch.tensor(_container.target_tensor_tokens)  # Batch size of 1
            model_encoder_decoder_mt = self.translation_handler.model_encoder_decoder_mt.to(torch.device('cpu'))
            embeddings = model_encoder_decoder_mt.models[0].decoder.embed_tokens(token_tensor)
        # end with
        dict_container = _container._asdict()
        dict_container['dict_layer_embeddings'] = {'decoder.word_embedding': embeddings}
        _container_updated = TranslationResultContainer(**dict_container)
        # end for

        return _container_updated

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
        self.translation_handler.max_len_a = max_len_a
        self.translation_handler.max_len_b = max_len_b

        seq_container_translation = self.translation_handler.sample_multiple_times(
            input_text=input_text.source, 
            temperature=temperature, 
            n_sampling=n_sampling)

        seq_container_updated = []
        for _container in seq_container_translation:
            with torch.no_grad():
                token_tensor = _container.target_tensor_tokens.detach().clone()  # Batch size of 1
                model_encoder_decoder_mt = self.translation_handler.model_encoder_decoder_mt.to(torch.device('cpu'))
                embeddings = model_encoder_decoder_mt.models[0].decoder.embed_tokens(token_tensor)
            # end with
            dict_container = _container._asdict()
            dict_container['dict_layer_embeddings'] = {'decoder.word_embedding': embeddings}
            _container_updated = TranslationResultContainer(**dict_container)
            seq_container_updated.append(_container_updated)
        # end for

        return seq_container_updated


class FairSeqVectorExtractorVer2(BaseVectorExtractorVer2):
    def __init__(self,
                 translation_handler: FairSeqTranslationModelHandlerVer2,
                 ) -> None:
        """
        """
        self.translation_handler = translation_handler