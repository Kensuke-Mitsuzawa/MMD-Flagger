import typing as ty

from pydantic import BaseModel

from .module_mmd_flagger.mmd_flagger import (
    EstimateReturnObject
)
from .module_mmd_flagger.module_models import (
    SampleSet,
    SingleSample
)

from pydantic import BaseModel, ConfigDict


class LLMResponseTextStochastic(BaseModel):
    temperature: float
    responses: ty.List[str]


class ResultPerFeature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    feature_name: str
    mmd_trajectory: EstimateReturnObject

    def get_score(self) -> float:
        return self.mmd_trajectory.score


class Result(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    score_ensemble: float
    result_obj_feature: ty.List[ResultPerFeature]


class SampleSetContainer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    feature_name: str
    sample_y_hyp: SampleSet
    sample_set_y_stoch: ty.List[SampleSet]
    