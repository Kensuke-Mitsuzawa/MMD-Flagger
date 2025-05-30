import typing as ty
from pathlib import Path
import logging

import numpy as np
import torch

from .module_base import (BaseVectorExtractorVer2, OPTION_EMBEDDING_LAYERS, ReturnTuple_method_extract_hidden_states)
from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ...module_translation_handler.ver2 import TransformersTranslationModelHandlerVer2

module_logger = logging.getLogger(__name__)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())




class TransformerVectorExtractorVer2(BaseVectorExtractorVer2):
    def __init__(self,
                 translation_handler: TransformersTranslationModelHandlerVer2
                 ) -> None:
        super().__init__(translation_handler=translation_handler)
