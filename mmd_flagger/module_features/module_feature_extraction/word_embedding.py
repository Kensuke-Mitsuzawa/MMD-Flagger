from typing import Any, Literal, List
import torch
import torch.nn.functional as F

from pydantic import Field, BaseModel, config, model_validator

from mmd_flagger.module_mmd_flagger.module_utils.tensor_preprocessor import (
    TensorPreprocessorWordEmbeddings,
    MODE_VECTOR_PREPROCESS
)

from ..models import GenerationInfoDict

from .base import BaseFeatureExtractor, BaseExtractedFeatureObject




def format_and_flatten_tensor(tensor: torch.Tensor, threshold: int) -> torch.Tensor:
    """
    Adjusts a tensor of shape (N, D) to (threshold, D) by clipping or padding,
    then flattens it to a 1D vector of size (threshold * D).
    """
    n_tokens, n_dim = tensor.shape
    
    if n_tokens > threshold:
        # Case 1: Exceeds threshold -> Clip to the first 'threshold' tokens
        # (Alternatively, use tensor[-threshold:] if you want the LAST tokens)
        adjusted_tensor = tensor[:threshold, :]
        
    elif n_tokens < threshold:
        # Case 2: Below threshold -> Apply padding
        # F.pad expects padding in reverse order of dimensions: (Left, Right, Top, Bottom)
        # We want to pad the 'token' dimension (dim 0), specifically at the bottom.
        padding_size = threshold - n_tokens
        # (0, 0) means no padding for n_dim; (0, padding_size) pads the end of n_token
        adjusted_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=0.0)
        
    else:
        # Case 3: Exactly matches threshold
        adjusted_tensor = tensor

    # Flatten the (threshold, n_dim) tensor into a 1D vector
    return adjusted_tensor.flatten()


class WordEmbeddingsOutput(BaseExtractedFeatureObject):
    registry_name: str = "word-embedding"
    model_config = config.ConfigDict(arbitrary_types_allowed=True)
    
    n_tokens_response: int = Field(description="The number of tokens stored in `embedding_vector`.")
    n_dimension: int = Field(description="The number of dimensions of one token stored in `embedding_vector`.")

    word_embeddings: torch.Tensor = Field(description="Tensor of shape (T, D), where T=generated-tokens, D=dims.")

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        # return ['avg', 'concant']
        return ['avg']

    @model_validator(mode='after')
    def check_tensor_shape(self) -> 'WordEmbeddingsOutput':
        if len(self.word_embeddings.shape) != 2:
            raise ValueError(f"Tensor of shape must be (T, D). Given {len(self.word_embeddings)}")
        
        if self.word_embeddings.shape[1] != self.n_dimension:
            raise ValueError("Invalid tensor shape.")
        
        n_token_expected = self.n_tokens_response
        if self.word_embeddings.shape[0] != n_token_expected:
            raise ValueError(f"Invalid tensor shape. Word embedding is expected to be {n_token_expected}. Actual={self.word_embeddings.shape[0]}.")

        return self

    @classmethod
    def get_extractor_class(cls) -> 'type[WordEmbeddingExtractor]':
        return WordEmbeddingExtractor

    def get_feature_vector(self, 
                           method_aggregation: MODE_VECTOR_PREPROCESS = 'avg',
                           n_concat_longest: int = 100) -> torch.Tensor:
        _tensor_preprocess = TensorPreprocessorWordEmbeddings(
            mode_vector_preprocess=method_aggregation,
            mode_max_token_length_vector_concat='fixed',
            option_max_token_length=n_concat_longest)
        
        return _tensor_preprocess.preprocess_tensors([self.word_embeddings])[0]



class WordEmbeddingExtractor(BaseFeatureExtractor):
    def __init__(self):
        pass

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return ['avg', 'concant']

    def extract(self, generation_obj: GenerationInfoDict, **kwargs) -> List[WordEmbeddingsOutput]:
        obj = WordEmbeddingsOutput(
            class_name_extractor=self.__class__.__name__,
            n_tokens_response=generation_obj.generated_token_ids.shape[0],
            n_dimension=generation_obj.word_embeddings_generated_tokens.shape[1],
            word_embeddings=generation_obj.word_embeddings_generated_tokens,
        )
        return [obj]
