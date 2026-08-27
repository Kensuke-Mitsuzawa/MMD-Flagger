from typing import Any, Literal, List, Optional, Union
import torch
import torch.nn.functional as F
from pydantic import Field, BaseModel, config, model_validator

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


PossibleMethodAggregation = Literal['mean', 'concant', 'last_token']
class HiddenStatesOutput(BaseExtractedFeatureObject):
    registry_name: str = "hidden-states"

    layer_index: int

    n_tokens_prompt: int = Field(description="The number of tokens stored in `embedding_vector`.")
    n_tokens_response: int = Field(description="The number of tokens stored in `embedding_vector`.")
    n_dimension: int = Field(description="The number of dimensions of one token stored in `embedding_vector`.")

    embedding_vector: torch.Tensor = Field(description="(T, D), where T=tokens, D=dims")

    model_config = config.ConfigDict(arbitrary_types_allowed=True)

    def get_feature_name(self) -> str:
        return f"{self.registry_name}_{self.layer_index}"

    @model_validator(mode='after')
    def check_tensor_shape(self) -> 'HiddenStatesOutput':
        if not len(self.embedding_vector.shape) == 2:
            raise ValueError(f"Tensor of shape must be (T, D). Given {len(self.embedding_vector)}")
        
        if self.embedding_vector.shape[1] != self.n_dimension:
            raise ValueError(
                f"Dimension mismatch! Expected n_dimension={self.n_dimension}, "
                f"but tensor has {self.embedding_vector.shape[1]} dimensions."
            )
        
        actual_tokens = self.embedding_vector.shape[0]
        expected_tokens = self.n_tokens_prompt + self.n_tokens_response        
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"Token count mismatch! "
                f"Expected {expected_tokens} (Prompt {self.n_tokens_prompt} + Response {self.n_tokens_response}), "
                f"but tensor has {actual_tokens} rows."
            )
        
        return self

    @classmethod
    def get_extractor_class(cls) -> 'type[HiddenStatesExtractor]':
        return HiddenStatesExtractor

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        # return ['mean', 'concant', 'last_token']
        return ['mean', 'last_token']
        
    def get_feature_vector(self, 
                           method_aggregation: PossibleMethodAggregation = 'mean',
                           n_concat_longest: int = 100) -> torch.Tensor:
        if method_aggregation == 'mean':
            return self.embedding_vector.mean(dim=0)
        elif method_aggregation == 'concant':
            return format_and_flatten_tensor(self.embedding_vector, n_concat_longest)
        elif method_aggregation == 'last_token':
            return self.embedding_vector[-1]
        else:
            raise ValueError(f"{method_aggregation} is not defined. It must be of {self.get_supported_aggregations()}")


PossibleLayerCommand = Literal['middle', 'first', 'last']

class HiddenStatesExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        resolved_layer_ids: Optional[List[int]] = None
    ):
        self.resolved_layer_ids = resolved_layer_ids

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return ['mean', 'concant', 'last_token']

    def extract(
        self,
        generation_obj: GenerationInfoDict,
        resolved_layer_ids: Optional[List[Union[int, PossibleLayerCommand]]] = None,
        **kwargs,
    ) -> List[HiddenStatesOutput]:
        # determine which layers to iterate over
        seq_layer_index = list(generation_obj.layer_hidden_states.keys())

        # logic to resolve 
        # if it is a list of ints, then just extract the given layers
        # if it is 'first', then extract the first layer
        # if it is 'last', then extract the last layer
        # if it is 'middle', then extract the middle layer
        target_resolved_layer_ids = []
        if resolved_layer_ids is None:
            target_resolved_layer_ids = getattr(self, 'resolved_layer_ids', None)
        else:
            if all(isinstance(i, int) for i in resolved_layer_ids):
                target_resolved_layer_ids = resolved_layer_ids
            elif any(isinstance(i, str) for i in resolved_layer_ids):
                for _item in resolved_layer_ids:
                    if _item == 'first':
                        target_resolved_layer_ids.append(0)
                    elif _item == 'last':
                        target_resolved_layer_ids.append(len(seq_layer_index) - 1)
                    elif _item == 'middle':
                        target_resolved_layer_ids.append(len(seq_layer_index) // 2)
                    else:
                        raise ValueError(f"{_item} is not defined. It must be of {PossibleLayerCommand}")
                else:
                    raise ValueError(f"{resolved_layer_ids} is not defined. It must be of {PossibleLayerCommand}")
            # end if
        # end if


        seq_layer_index = [l for l in seq_layer_index if l in target_resolved_layer_ids]

        seq_extraction = []
        for _layer_index in seq_layer_index:
            _embedding_states = generation_obj.layer_hidden_states[_layer_index]

            obj = HiddenStatesOutput(
                class_name_extractor=self.__class__.__name__,
                layer_index=_layer_index,
                n_tokens_prompt=generation_obj.prompt_input_ids.shape[0],
                n_tokens_response=generation_obj.generated_token_ids.shape[0],
                n_dimension=_embedding_states.shape[1],
                embedding_vector=_embedding_states
            )
            seq_extraction.append(obj)
        # end for
        return seq_extraction
