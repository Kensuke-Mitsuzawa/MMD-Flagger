import typing as ty

from pydantic import BaseModel

from .module_mmd_flagger.mmd_flagger import (
    MMDFlagger,
    EstimateReturnObject
)
from .module_ensemble import masate_online_algorithm
from .model_objects import (
    SampleSetContainer,
    ResultPerFeature,
    Result
)



class Interface(object):
    def __init__(
        self,
        mmd_flagger: MMDFlagger
        ):
        self.mmd_flagger = mmd_flagger

    def fit(
        self,
        samples: ty.List[SampleSetContainer]
        ) -> Result:
        """

        This function fit the Interface.

        Args:
            feature2samples: dictionary of feature name to a tuple of (hypothesis sample set, list of stochastic sample sets).

        Returns:
            Result: the result of the fit.
        """

        dict_feat2mmd_flagger_trajectory_obj = {}  
        for _sample_obj in samples:
            _y_hyp = _sample_obj.sample_y_hyp
            _y_sto_samples = _sample_obj.sample_set_y_stoch

            # execute mmd-flagger on all feature types.
            _mmd_trajectory_feat = self.mmd_flagger.estimate(
                _y_hyp, 
                _y_sto_samples, 
                scoring_method="tau_min_and_tau_max_and_tau_argmin"
            )
            dict_feat2mmd_flagger_trajectory_obj[_sample_obj.feature_name] = _mmd_trajectory_feat
        # end for

        # execute the ensemble function here.
        _score_ensemble = masate_online_algorithm.detect_hallucination_masate_online(dict_feat2mmd_flagger_trajectory_obj)

        res_per_feat = []
        for _feat, _mmd_trajectory_feat in dict_feat2mmd_flagger_trajectory_obj.items():
            res_per_feat.append(ResultPerFeature(feature_name=_feat, mmd_trajectory=_mmd_trajectory_feat))
        # end for

        return Result(score_ensemble=_score_ensemble, result_obj_feature=res_per_feat)
