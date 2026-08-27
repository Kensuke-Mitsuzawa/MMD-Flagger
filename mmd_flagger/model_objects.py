import typing as ty

from pydantic import BaseModel

from .module_mmd_flagger.mmd_flagger import (
    EstimateReturnObject
)
from .module_mmd_flagger.module_models import (
    SampleSet,
    SingleSample
)

from pydantic import BaseModel


class ResultPerFeature(BaseModel):
    feature_name: str
    mmd_trajectory: EstimateReturnObject

    def get_score(self) -> float:
        return self.mmd_trajectory.score


class Result(BaseModel):
    score_ensemble: float
    result_obj_feature: ResultPerFeature

    
class SampleSetContainer(BaseModel):
    feature_name: str
    sample_y_hyp: SampleSet
    sample_set_y_stoch: ty.List[SampleSet]
    