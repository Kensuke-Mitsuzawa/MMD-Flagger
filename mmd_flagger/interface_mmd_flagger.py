import typing as ty

from pydantic import BaseModel

from .module_mmd_flagger.mmd_flagger import (
    MMDFlagger,
    EstimateReturnObject,
    PossibleMetricsMMDFlaggerInterface
)
from .module_ensemble import masate_online_algorithm
from .model_objects import (
    SampleSetContainer,
    ResultPerFeature,
    InterfaceResult
)



class Interface(object):
    def __init__(
        self,
        mmd_flagger: MMDFlagger
        ):
        self.mmd_flagger = mmd_flagger

    def fit(
        self,
        samples: ty.List[SampleSetContainer],
        scoring_methods: ty.Tuple[PossibleMetricsMMDFlaggerInterface, ...] = ("tau_min_and_tau_max_and_tau_argmin", "qunatile_residual") 
        ) -> InterfaceResult:
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

            for _score_method in scoring_methods:
                # execute mmd-flagger on all feature types.
                _mmd_trajectory_feat = self.mmd_flagger.estimate(
                    _y_hyp, 
                    _y_sto_samples, 
                    scoring_method=_score_method
                )
                dict_feat2mmd_flagger_trajectory_obj[(_sample_obj.feature_name, _score_method)] = _mmd_trajectory_feat
            # end for
        # end for

        # execute the ensemble function here.
        dict_input_ensemble = {key_tuple[0] : _val for key_tuple, _val in dict_feat2mmd_flagger_trajectory_obj.items()}
        _score_ensemble = masate_online_algorithm.detect_hallucination_masate_online(dict_input_ensemble)

        res_per_feat = []
        for _key_tuple, _mmd_trajectory_feat in dict_feat2mmd_flagger_trajectory_obj.items():
            res_per_feat.append(ResultPerFeature(
                feature_name=_key_tuple[0], 
                mmd_trajectory=_mmd_trajectory_feat,
                scoring_method=_key_tuple[1]))
        # end for

        return InterfaceResult(score_ensemble=_score_ensemble, result_obj_feature=res_per_feat)
