from typing import Any, List, Union
import torch

from pydantic import Field, config

from ..models import GenerationInfoDict

from .base import BaseFeatureExtractor, BaseExtractedFeatureObject


from sentence_transformers import SentenceTransformer

class SemanticEmbeddingOutput(BaseExtractedFeatureObject):
    registry_name: str = "semantic-embedding"
    class_name_extractor: str = "SemanticEmbeddingExtractor"

    model_name_semantic_model: str
    tensor_semantic_embedding: torch.Tensor
    n_tokens_response: int = Field(description="The number of tokens stored in `embedding_vector`.")
    n_dimension: int = Field(description="The number of dimensions of one token stored in `embedding_vector`.")

    model_config = config.ConfigDict(arbitrary_types_allowed=True)    

    def get_feature_name(self) -> str:
        return f"{self.registry_name}_{self.model_name_semantic_model}"

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return []

    @classmethod
    def get_extractor_class(cls) -> 'type[SemanticEmbeddingExtractor]':
        return SemanticEmbeddingExtractor

    def get_feature_vector(self, method_aggregation: Any = None, **kwargs: Any) -> torch.Tensor:
        if isinstance(self.tensor_semantic_embedding, torch.Tensor):
            tensor = self.tensor_semantic_embedding
        else:
            tensor = torch.tensor(self.tensor_semantic_embedding)
        return tensor.flatten()


class SemanticEmbeddingExtractor(BaseFeatureExtractor):
    registry_name: str = "semantic-embedding"
    def __init__(self, model_name_semantic_model: Union[str, SentenceTransformer] = "sentence-transformers/all-MiniLM-L6-v2"):
        if isinstance(model_name_semantic_model, str):
            self.model_name_semantic_model = model_name_semantic_model
            self.model = SentenceTransformer(model_name_semantic_model)
        elif isinstance(model_name_semantic_model, SentenceTransformer):
            self.model = model_name_semantic_model
            self.model_name_semantic_model = getattr(
                model_name_semantic_model, 
                "model_name_or_path", 
                str(model_name_semantic_model)
            )
        else:
            raise ValueError(f"Invalid model_name_semantic_model type: {type(model_name_semantic_model)}")

    def extract(self, generation_obj: GenerationInfoDict, **kwargs) -> List[SemanticEmbeddingOutput]:
        response_text = generation_obj.response_text
        embeddings = self.model.encode(response_text, convert_to_tensor=True)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        
        embeddings = embeddings.detach().cpu()

        if embeddings.ndim == 1:
            n_tokens = 1
            n_dim = embeddings.shape[0]
        else:
            n_tokens = embeddings.shape[0]
            n_dim = embeddings.shape[-1]

        obj = SemanticEmbeddingOutput(
            model_name_semantic_model=self.model_name_semantic_model,
            tensor_semantic_embedding=embeddings,
            n_tokens_response=n_tokens,
            n_dimension=n_dim
        )
        return [obj]
