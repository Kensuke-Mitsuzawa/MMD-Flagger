from typing import Optional, Dict
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict


class DecodingStrategyName(str, Enum):
    Argmax = "argmax"
    BeamSearch = "beam_search"
    TopK = "top_k"
    TopP = "top_p"
    Stochastic = "stochastic"
    TeacherForcing = "teacher_forcing"

    def __str__(self) -> str:
        return self.value


class DecodingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)  # Allows this model to be used as a dictionary key
    # end case

    method: DecodingStrategyName
    max_new_tokens: int = 100

    # Beam Search
    num_beams: Optional[int] = Field(default=None, ge=1) # >= 1
    num_return_sequences: int = Field(default=1, ge=1) # >= 1
    early_stopping: Optional[bool] = None # Or bool, if always needed for beam

    # ---------
    # stochastic sampling parameters

    # Top-K
    k: Optional[int] = Field(default=None, ge=1) # >= 1

    # Top-P
    p: Optional[float] = Field(default=None, ge=0.0, le=1.0) # 0.0 <= p <= 1.0

    # Common for Sampling
    temperature: Optional[float] = Field(default=1.0, ge=0.0) # >= 0.0

    random_seed: int = Field(default=42, ge=0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    
    # Real-time stopping and filtering
    stop_strings: Optional[tuple[str, ...]] = Field(default=None, description="Stop generation if any of these strings appear in the output.")
    bad_words: Optional[tuple[str, ...]] = Field(default=None, description="Prevent these words/strings from being generated.")

    # --- Add validators to ensure params match method ---
    # mode='after' allows access to other validated fields via info.data
    @field_validator('num_beams', mode='after')
    def check_beam_search_params(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        # 'info.data' contains the dictionary of fields already processed
        if info.data.get('method') == DecodingStrategyName.BeamSearch:
            if v is None:
                raise ValueError("`num_beams` is required for 'beam_search' method")
        else:
            v = None
        # end if
        return v # Return the value if validation passes

    @field_validator('k', mode='after')
    def check_top_k_params(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        if info.data.get('method') == DecodingStrategyName.TopK and v is None:
            raise ValueError("`k` is required for 'top_k' method")
        if info.data.get('method') not in (DecodingStrategyName.TopK, DecodingStrategyName.Stochastic) and v is not None:
            raise ValueError("`k` is only applicable for 'top_k' or 'stochastic' method")
        return v

    @field_validator('p', mode='after')
    def check_top_p_params(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if info.data.get('method') == DecodingStrategyName.TopP and v is None:
            raise ValueError("`p` is required for 'top_p' method")
        if info.data.get('method') not in (DecodingStrategyName.TopP, DecodingStrategyName.Stochastic) and v is not None:
            raise ValueError("`p` is only applicable for 'top_p' or 'stochastic' method")
        return v
    
    @field_validator('temperature', mode='after')
    def check_temperature_params(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if info.data.get('method') == DecodingStrategyName.TeacherForcing:
            v = None
        return v


    @field_validator('num_return_sequences', mode='after')
    def check_num_return_sequences_params(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if info.data.get('method') == DecodingStrategyName.TeacherForcing:
            v = 1
        return v

    def to_dict(self) -> Dict:
        d_base = self.model_dump()
        d_base['method'] = self.method.value

        return d_base
    