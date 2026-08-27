from enum import Enum

from pydantic import BaseModel, Field, model_validator, ConfigDict
import torch
from typing import Any, Dict, Optional



class GenerationInfoDict(BaseModel):
    prompt_input_ids: torch.Tensor = Field(description="(T,), where T=tokens.")
    generated_token_ids: torch.Tensor = Field(description="(T,), where T=tokens.")
    layer_hidden_states: Dict[int, torch.Tensor]
    attention_matrix: torch.Tensor = Field(description="(L, H, T, T), where L=layers, H=heads, T=tokens.")
    word_embeddings_generated_tokens: torch.Tensor = Field(description="(T, D), where T=tokens, D=dims.")
    random_seed: Optional[int]
    batch_size: Optional[int]
    repetition_penalty: Optional[float] = None

    model_config = ConfigDict({
        "arbitrary_types_allowed": True,
    })
    
    @model_validator(mode='after')
    def check_shapes(self):
        if len(self.prompt_input_ids.shape) == 2:
            # (1, token_ids) -> (token_ids)
            self.prompt_input_ids = self.prompt_input_ids.squeeze(0)
        # end if

        if len(self.prompt_input_ids.shape) != 1:
            raise ValueError(f"The prompt_input_ids must be (T,). Given: {self.prompt_input_ids.shape}.")
        
        if len(self.generated_token_ids.shape) != 1:
            raise ValueError(f"The generated_token_ids must be (T,). Given: {self.generated_token_ids.shape}.")

        if len(self.attention_matrix.shape) != 4: 
            raise ValueError(f"The attention matrix must be (L, H, T, T). Given: {self.attention_matrix.shape}.")
        
        if len(self.word_embeddings_generated_tokens.shape) != 2:
            raise ValueError(f"(T, D), where T=tokens, D=dims. Given: {self.word_embeddings_generated_tokens.shape}")

        if self.generated_token_ids.shape[0] != self.word_embeddings_generated_tokens.shape[0]:
            raise ValueError(f"Expected shape of word-embedding has {self.generated_token_ids.shape[0]} tokens. Actual token size={self.word_embeddings_generated_tokens.shape[0]}")
        
        n_token_total = (self.prompt_input_ids.shape[0] + self.generated_token_ids.shape[0])
        for _layer_id, _tensor in self.layer_hidden_states.items():
            if (_tensor.shape[0] != n_token_total):
                raise ValueError(f"Error at `layer_hidden_states`. Expected tensor size={n_token_total}, actual={_tensor.shape[0]} at layer-id={_layer_id}.")
        # end for

        return self

