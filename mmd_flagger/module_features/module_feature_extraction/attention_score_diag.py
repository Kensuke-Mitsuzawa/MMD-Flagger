from typing import Any, List, Optional
import torch
from pydantic import Field, BaseModel, config, model_validator

from ..models import GenerationInfoDict

from .base import BaseFeatureExtractor, BaseExtractedFeatureObject


"""The name `AttentionEigenVals` is from Binkowski et al. 2025.
The original paper is the following.
Sriramanan, Gaurang, et al. "Llm-check: Investigating detection of hallucinations in large language models." Advances in Neural Information Processing Systems 37 (2024): 34188-34216.
"""


class AttentionEigenValsOutput(BaseExtractedFeatureObject):
    registry_name: str = "AttentionEigenVals"
    model_config = config.ConfigDict(arbitrary_types_allowed=True, extra='ignore')
    
    n_layers: int = Field(description="Number of layers in the model.")
    n_heads: int = Field(description="Number of heads in the model.")
    n_tokens: int = Field(description="Length of the sequence.")
    top_k: int = Field(description="Number of top features to keep per head.")
    
    tensor_scores: torch.Tensor = Field(description="Tensor of shape (L, H, T), where L is the number of layers, H is the number of heads.")
    tensor_index: torch.Tensor = Field(description="The corresponding indices of the tensor. (L, H, T)")

    def get_feature_name(self) -> str:
        return f"{self.registry_name}_{self.top_k}"
    
    @model_validator(mode='after')
    def check_values(self):
        if len(self.tensor_scores.shape) != 3:
            raise ValueError(f"Tensor of shape must be (L, H, T or top-k). Given {len(self.tensor_scores)}")

        if len(self.tensor_index.shape) != len(self.tensor_scores.shape):
            raise ValueError(f"Tensor of shape must be (L, H, T or top-k). Given {len(self.tensor_index)}")
        
        if self.tensor_scores.shape[0] != self.n_layers:
            raise ValueError()

        if self.tensor_scores.shape[1] != self.n_heads:
            raise ValueError()

        if self.top_k == -1:
            assert self.tensor_scores.shape[2] == self.n_tokens
        else:
            assert self.tensor_scores.shape[2] == self.top_k
        # end if

        return self

    @classmethod    
    def get_extractor_class(cls) -> 'type[AttentionEigenValsExtractor]':
        return AttentionEigenValsExtractor

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return []

    def get_feature_vector(self, method_aggregation: Optional[str] = None, **kwargs: Any) -> torch.Tensor:
        top_k = kwargs.get("top_k", 100)
        sliced_scores = self.tensor_scores[:, :, :top_k]
        return sliced_scores.flatten()



# @register_extractor("AttentionEigenVals")
class AttentionEigenValsExtractor(BaseFeatureExtractor):
    def __init__(self, top_k: int = 100):
        self.top_k = top_k

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return []

    def __get_eigvals_per_head_topk(self, attentions: torch.Tensor) -> AttentionEigenValsOutput:
        """        
        Args:
            attentions: Tensor of shape (L, H, T, T)
            k: Number of top features to keep per head.
            
        Returns:
            z: Aggregated feature vector.
        """
        L_layers, H_heads, T, _ = attentions.shape

        if self.top_k == -1:
            _len_seq = T
        else:
            _len_seq = self.top_k
        # end if

        tensor_scores = torch.zeros(L_layers, H_heads, _len_seq, device=attentions.device)
        index_tensor_scores = torch.zeros(L_layers, H_heads, _len_seq, device=attentions.device, dtype=torch.int)

        for l in range(L_layers):
            for h in range(H_heads):
                A = attentions[l, h]
                
                diag_vals = torch.diagonal(A)
                
                # Sort in descending order (or ascending depending on specific spectral theory needs)
                # Usually spectral features use smallest or largest magnitude. 
                # Assuming descending based on "top-k".
                sorted_vals, _ = torch.sort(diag_vals, descending=True)
                ind_sorted_vals = torch.argsort(diag_vals, descending=True)

                if self.top_k == -1:
                    _tilda_z = sorted_vals
                    _ind_tilda_z = ind_sorted_vals
                elif T >= self.top_k:
                    _tilda_z = sorted_vals[:self.top_k]
                    _ind_tilda_z = ind_sorted_vals[:self.top_k]
                else:
                    padding = torch.zeros(self.top_k - T, device=A.device)
                    # end if
                    _tilda_z = torch.cat([sorted_vals, padding])

                    padding_ind = torch.full((self.top_k - T,), -1, device=A.device)
                    _ind_tilda_z = torch.cat([ind_sorted_vals, padding_ind])
                # end if

                tensor_scores[l, h] = _tilda_z
                index_tensor_scores[l, h] = _ind_tilda_z
            # end for
        # end for

        _feature_obj = AttentionEigenValsOutput(
            class_name_extractor=self.__class__.__name__,
            n_layers=L_layers,
            n_heads=H_heads,
            n_tokens=T,
            top_k=self.top_k,
            tensor_scores=tensor_scores,
            tensor_index=index_tensor_scores
        )

        return _feature_obj

    def extract(self, generation_obj: GenerationInfoDict, **kwargs) -> List[AttentionEigenValsOutput]:
        # 1. Logic from previous prompts
        # Input: (L, H, T, T)
        z_vector = self.__get_eigvals_per_head_topk(generation_obj.attention_matrix)
        
        return [z_vector]
