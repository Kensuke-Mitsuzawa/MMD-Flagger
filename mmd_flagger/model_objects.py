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

from matplotlib.axes import Axes
import seaborn


class LLMResponseTextStochastic(BaseModel):
    temperature: float
    responses: ty.List[str]


class ResultPerFeature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    feature_name: str
    mmd_trajectory: EstimateReturnObject

    def get_score(self) -> float:
        return self.mmd_trajectory.score


class SampleSetContainer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    feature_name: str
    sample_y_hyp: SampleSet
    sample_set_y_stoch: ty.List[SampleSet]


class InterfaceResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    score_ensemble: float
    result_obj_feature: ty.List[ResultPerFeature]

    def render_mmd_trajectories(self, ax_obj: Axes) -> Axes:
        """Render the MMD-trajectories for all features."""
        for _res_per_feat in self.result_obj_feature:
            _res_per_feat.mmd_trajectory.render_mmd_trajectory(ax_obj, name_feature=_res_per_feat.feature_name)
        # end for

        # set the legend box
        ax_obj.legend()

        return ax_obj
