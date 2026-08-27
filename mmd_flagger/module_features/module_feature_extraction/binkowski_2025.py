from typing import Any, List, Optional
import torch
from pydantic import Field, BaseModel, config, model_validator

from ..models import GenerationInfoDict

from .base import BaseFeatureExtractor, BaseExtractedFeatureObject


"""The proposal and name `LapEigvals` is from Binkowski et al. 2025.
The original paper is the following.
Binkowski, Jakub, et al. "Hallucination detection in llms using spectral features of attention maps." Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025.
"""

class LapEigvalsOutput(BaseExtractedFeatureObject):
    registry_name: str = "LapEigvals"

    model_config = config.ConfigDict(arbitrary_types_allowed=True)
    
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
    def get_extractor_class(cls) -> 'type[LapEigvalsExtractor]':
        return LapEigvalsExtractor

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return []

    def get_feature_vector(self, method_aggregation: Optional[str] = None, **kwargs: Any) -> torch.Tensor:
        top_k = kwargs.get("top_k", 100)
        sliced_scores = self.tensor_scores[:, :, :top_k]
        return sliced_scores.flatten()



# @register_extractor("LapEigvals")
class LapEigvalsExtractor(BaseFeatureExtractor):
    def __init__(self, top_k: int = 100):
        self.top_k = top_k

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return []

    def __compute_degree_matrix(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """
        Step 2: Computing the out-of-degree matrix D.
        Formula: d_{i, i} = sum_{u} a_{u, i} / (T - i)
        
        Args:
            attention_matrix: Tensor of shape (T, T)
        Returns:
            D: Diagonal degree matrix of shape (T, T)
        """
        T = attention_matrix.shape[-1]
        
        # Sum over u (rows) to get the total incoming attention for each column i
        # Note: Depending on orientation, if A[u, i] means attention from u to i, we sum dim=0.
        sum_a = torch.sum(attention_matrix, dim=0)
        
        # Calculate normalization factor (T - i)
        # Indices i range from 0 to T-1
        indices = torch.arange(T, device=attention_matrix.device)
        normalization = T - indices
        
        # Handle potential division by zero if T-i is 0 (last token)
        # Though usually T-i >= 1 since i goes up to T-1. 
        # If i can be T, we clamp or mask.
        normalization = torch.clamp(normalization, min=1.0)
        
        d_diag_values = sum_a / normalization
        
        # Construct diagonal matrix
        D = torch.diag(d_diag_values)
        
        return D

    def __get_laplacian_eigvals_per_head_topk(self, attentions: torch.Tensor) -> LapEigvalsOutput:
        """
        Step 3 & 4: Constructing Laplacian and Extracting 'Eigenvalues'.

        The original implementation takes top-k elements for each layer and head.
        To make extracted feature reusable, this module holds all scores. 
        
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
                
                # Step 2: Compute Degree Matrix
                D = self.__compute_degree_matrix(A)
                
                # Step 3: Construct Laplacian Matrix (L = D - A)
                L_mat = D - A
                
                # Step 4: Extracting and Sorting Eigenvalues
                # Formula: z_tilde = sort(diag(L))
                # Note: For triangular matrices (causal attention), eigenvalues are the diagonal elements.
                diag_vals = torch.diagonal(L_mat)
                
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

        _feature_obj = LapEigvalsOutput(
            class_name_extractor=self.__class__.__name__,
            n_layers=L_layers,
            n_heads=H_heads,
            n_tokens=T,
            top_k=self.top_k,
            tensor_scores=tensor_scores,
            tensor_index=index_tensor_scores
        )

        return _feature_obj

    def extract(self, generation_obj: GenerationInfoDict, **kwargs) -> List[LapEigvalsOutput]:
        # 1. Logic from previous prompts
        # Input: (L, H, T, T)
        z_vector = self.__get_laplacian_eigvals_per_head_topk(generation_obj.attention_matrix)
        
        return [z_vector]
