import typing as ty
from pydantic import Field, BaseModel, field_validator, ValidationInfo, ConfigDict
import hashlib

import torch
import numpy as np

from ...utils.llm_decoding_conf_models import DecodingConfig

from .db_keys import SampleSetUniqueId

LabelTypes = ty.Literal['Y_hyp', 'Y_sto']


class TemperatureParameter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    temperature_parameter: torch.Tensor | np.ndarray | float = Field(description="temperature paramaters to generate this sample set.")

    def as_float(self) -> float:
        if isinstance(self.temperature_parameter, (torch.Tensor, np.ndarray)):
            return float(self.temperature_parameter.item())
        else:
            return self.temperature_parameter


class SingleSample(BaseModel):
    """It behaves a LLM's generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    sample_unique_id: str = Field()
    text: str = Field()
    feature_vector: ty.Optional[torch.Tensor] = Field(description="A feature representation of this sample. 1D tensor.") 


class SampleSet(BaseModel):
    label: LabelTypes
    decoding_config: ty.Optional[DecodingConfig] = Field(description="decoding config to generate this sample set. None if the label is h_hyp.")
    temperature_parameter: ty.Optional[TemperatureParameter | float] = Field(description="temperature paramaters to generate this sample set. None if the label is h_hyp.")
    samples: ty.List[SingleSample]

    @field_validator('decoding_config', mode='after')
    @classmethod
    def check_decoding_config(cls, value, info: ValidationInfo) -> DecodingConfig:
        # logic is None OK if Y_hyp else raise.
        if info.data['label'] == 'Y_hyp':
            pass
        else:
            assert value is not None, '`decoding_config` must be given when the sample set is "Y_sto"'
        # end if
        return value

    @field_validator('temperature_parameter', mode='after')
    @classmethod
    def check_temperature_parameter(cls, value, info: ValidationInfo) -> TemperatureParameter:
        # logic is None OK if Y_hyp else raise.
        if info.data['label'] == 'Y_hyp':
            pass
        else:
            assert value is not None, '`temperature_parameter` must be given when the sample set is "Y_sto"'
            if isinstance(value, float):
                value = TemperatureParameter(temperature_parameter=value)
        # end if
        return value

    def get_unique_id(self) -> ty.Union[str, SampleSetUniqueId]:
        """generating the unique id.
        """
        # todo; concat all unique ids of the samples, temp, and decoding_config.
        decoding_config_json = self.decoding_config.model_dump_json() if self.decoding_config is not None else None
        temperature_parameter_json = self.temperature_parameter.model_dump_json() if self.temperature_parameter is not None else None
        seq_sample_unique_ids = '|'.join([o.sample_unique_id for o in self.samples])

        key_doc = f'{self.label}/{seq_sample_unique_ids}/{temperature_parameter_json}/{decoding_config_json}'
        unique_id = hashlib.sha256(key_doc.encode('utf-8')).hexdigest()

        return unique_id

    def get_text_samples(self) -> ty.List[str]:
        return [_o.text for _o in self.samples]

    def get_embedding_samples(self) -> ty.List[torch.Tensor]:
        is_available = [True if _o.feature_vector is not None else False for _o in self.samples]
        if all(is_available) is False:
            msg = 'The feature field `feature_vector` is not available.'
            raise Exception(msg)
        else:
            return [_o.feature_vector for _o in self.samples]  # type: ignore
        # end if
