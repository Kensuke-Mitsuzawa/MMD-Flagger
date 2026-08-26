import typing as ty

import pandas as pd
import torch
import numpy as np

from scipy.stats import entropy

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator

from ..codebase_ver3_1.models import MmdErrorFlagResultVer3
from ...module_models.model_sample_set_container import SampleSet

from ...module_kernels.string_based_gaussian_kernel import StringBasedGaussianKernel
from ...module_kernels.module_distance import module_preprocessing


class MmdMatrix(ty.NamedTuple):
    values_index: np.ndarray
    values_column: np.ndarray
    values_matrix: np.ndarray

    def get_vector_mmd(self, value_index: float) -> np.ndarray:
        """search the corresponding value at index, returning the vector."""
        index_val = np.where(self.values_index == value_index)[0]
        return self.values_matrix[index_val, :]

    def to_pandas(self) -> pd.DataFrame:
        data_object = pd.DataFrame(
            data=self.values_matrix,
            index=self.values_index,
            columns=self.values_column)
        return data_object


class MmdTrajectory(ty.NamedTuple):
    values_index: np.ndarray
    values_trajectory: np.ndarray

    @classmethod
    def from_MmdErrorFlagResultVer3(cls, mmd_flagger_trajectory: MmdErrorFlagResultVer3) -> "MmdTrajectory":
        mmd_trajectory = MmdTrajectory(
            values_index=np.array(mmd_flagger_trajectory.tau_parameter),
            values_trajectory=np.array(mmd_flagger_trajectory.mmd_distances))
        return mmd_trajectory



PossibleMetricsMmdVariability = ty.Literal['coefficient_of_variation', 'entropy', 'outlier_ratio', 'qunatile_residual']


# ---- computing Hypothesis-Set v.s. Stochastic-Set ----


# def _compute_distance_hypothesis_and_stochastic(
#         mmd_estimator: QuadraticMmdEstimator,
#         tensor_hypothesis: torch.Tensor,
#         temperature2tensor: ty.List[StochasticEmbeddingSet],
#         torch_device: torch.device
#     ) -> MmdTrajectory:
#     # computing MMD-sequence between beam-search and stochastic-sampling.
#     _tau_sequence = []
#     _mmd_trajectory = []

#     temperature2tensor = sorted(temperature2tensor, key=lambda o: o.temperature)

#     for temp_sample in temperature2tensor:
#         _tensor = temp_sample.embedding_set
#         _tensor_beam_search = tensor_hypothesis.float()
#         _tensor = _tensor.float()
#         _mmd_values_obj = mmd_estimator.forward(
#             _tensor_beam_search.to(torch_device), 
#             _tensor.to(torch_device))

#         _tau_sequence.append(temp_sample.temperature)
#         _mmd_trajectory.append(_mmd_values_obj.mmd.item())
#     # end for

#     return MmdTrajectory(
#         values_index=np.array(_tau_sequence),
#         values_trajectory=np.array(_mmd_trajectory)
#     )



# ---- computing the adjacenry matrix of Hypothesis-Set ----


def _compute_distance_inner_stochastic_samples(mmd_estimator: QuadraticMmdEstimator,
                                               temperature2tensor: ty.List[SampleSet],
                                               torch_device: torch.device
                                               ) -> MmdMatrix:
    n_size_matrix = len(temperature2tensor)
    assert all([o.temperature_parameter is not None for o in temperature2tensor])
    temperature2tensor = sorted(temperature2tensor, key=lambda o: o.temperature_parameter.as_float())  # type: ignore

    index_tau = [_o.temperature_parameter.as_float() for _o in temperature2tensor if _o.temperature_parameter is not None]
    assert len(index_tau) > 0

    # computing the adjajency matrix between stochastic-sampling.
    _tensor_mmd_matrix = np.zeros(shape=(n_size_matrix, n_size_matrix))
    for _i_tau, _obj_a in enumerate(temperature2tensor):
        for _j_tau, _obj_b in enumerate(temperature2tensor):
            if isinstance(mmd_estimator.kernel_obj, StringBasedGaussianKernel):
                _sent_a: ty.List[str] = _obj_a.get_text_samples()
                _feat_a = module_preprocessing.nltk_preprocess_text(_sent_a)
            else:
                _feat_a = torch.stack(_obj_a.get_embedding_samples(), dim=0).float().to(torch_device)
            #
            if isinstance(mmd_estimator.kernel_obj, StringBasedGaussianKernel):
                _sent_b: ty.List[str] = _obj_b.get_text_samples()
                _feat_b = module_preprocessing.nltk_preprocess_text(_sent_b)
            else:
                _feat_b = torch.stack(_obj_b.get_embedding_samples(), dim=0).float().to(torch_device)
            # 

            if _i_tau == _j_tau:
                _mmd = 0.0
                _tensor_mmd_matrix[_i_tau, _j_tau] = _mmd
            else:
                if len(_feat_a) <= 1 or len(_feat_b) <= 1:
                    # MMD cannot be reliably computed on sample size <= 1 without ZeroDivisionError
                    _tensor_mmd_matrix[_i_tau, _j_tau] = 0.0
                else:
                    with torch.no_grad():
                        _mmd_values_obj = mmd_estimator.forward(_feat_a, _feat_b)
                        _tensor_mmd_matrix[_i_tau, _j_tau] = _mmd_values_obj.mmd
                    # end with
            # end if
        # end for
    # end for

    m = MmdMatrix(
        values_index=np.array(index_tau),
        values_column=np.array(index_tau),
        values_matrix=_tensor_mmd_matrix
    )

    return m


# ---- metrics on the matrix ----


def _get_coefficient_of_variation(_mmd_matrix: MmdMatrix) -> float:
    """Ref: https://en.wikipedia.org/wiki/Coefficient_of_variation
    """
    _mean = _mmd_matrix.values_matrix.mean()
    _std = _mmd_matrix.values_matrix.std()
    _variance = _mmd_matrix.values_matrix.var()
    _cv = _std / _mean if _mean != 0 else 0

    return _cv
    

def _get_entropy(_mmd_matrix: MmdMatrix) -> float:
    _entropy = entropy(_mmd_matrix.values_matrix.flatten())

    return _entropy


def _get_quantile_residuam_sum(_mmd_trajectory: MmdTrajectory,
                               _mmd_matrix: MmdMatrix,
                               percentile: int = 90) -> float:
    # computing the metric: Quantile Residual Sum
    _stack_quantile_residual_sum = []
    for _tau, _mmd in zip(_mmd_trajectory.values_index, _mmd_trajectory.values_trajectory):
        _mmd_sto_a_and_b: np.ndarray = _mmd_matrix.get_vector_mmd(_tau)
        _quantile_mmd_sto_a_and_b = np.percentile(_mmd_sto_a_and_b, percentile)

        _residual_i = max(0, (_mmd - _quantile_mmd_sto_a_and_b))
        _stack_quantile_residual_sum.append(_residual_i)
    # end for
    _average_quantile_residual = np.mean(_stack_quantile_residual_sum)

    return _average_quantile_residual.item()


def _get_outlier_ratio(_mmd_trajectory: MmdTrajectory,
                       _mmd_matrix: MmdMatrix) -> float:
    # computing the metric: Outlier Ratio
    _stack_outlier_ratio = []
    for _tau, _mmd in zip(_mmd_trajectory.values_index, _mmd_trajectory.values_trajectory):
        _mmd_sto_a_and_b: np.ndarray = _mmd_matrix.get_vector_mmd(_tau)
        _quantile_mmd_sto_a_and_b = np.percentile(_mmd_sto_a_and_b, 90)

        _is_out_mmd_beam_sto_in_distribution = 1 if _mmd > _quantile_mmd_sto_a_and_b else 0
        
        _stack_outlier_ratio.append(_is_out_mmd_beam_sto_in_distribution)
    # end for
    _ratio_outlier = sum(_stack_outlier_ratio) / len(_stack_outlier_ratio)

    return _ratio_outlier

# --------

MetricTypes = ty.Literal[
    'coefficient_of_variation',
    'qunatile_residual',
    'outlier_ratio'
]


class MmdFlaggerScoreBaseVer1(ty.NamedTuple):
    mmd_trajectory: MmdTrajectory
    mmd_matrix: MmdMatrix

    metrics: ty.List[ty.Dict]
    trajectory_shape: str
    



def score_mmd_trajectory_variability(metric_name: PossibleMetricsMmdVariability,
                                     mmd_matrix: MmdMatrix,
                                     mmd_trajectory: MmdTrajectory) -> float:
    if metric_name == 'coefficient_of_variation':
        _v = _get_coefficient_of_variation(mmd_matrix)
    elif metric_name == 'entropy':
        _v = _get_entropy(mmd_matrix)
    elif metric_name == 'outlier_ratio':
        _v = _get_outlier_ratio(mmd_trajectory, mmd_matrix)
    elif metric_name == 'qunatile_residual':
        _v = _get_quantile_residuam_sum(mmd_trajectory, mmd_matrix)
    else:
        raise NotImplementedError()
    # end if

    return _v



def mmd_flagger_score_base(mmd_estimator: QuadraticMmdEstimator,
                           mmd_flagger_trajectory: MmdErrorFlagResultVer3,
                           embedding_set_stochastic: ty.List[SampleSet],
                           torch_device: torch.device,
                           metrics: ty.Optional[ty.List[MetricTypes]] = None,
                           ) -> MmdFlaggerScoreBaseVer1:
    """The main interface of this module"""
    if metrics is None:
        metrics =  ['qunatile_residual', 'outlier_ratio']
    # end if

    mmd_estimator = mmd_estimator.to(torch_device)

    # mmd_trajectory = _compute_distance_hypothesis_and_stochastic(
    #     mmd_estimator=mmd_estimator,
    #     tensor_hypothesis=embedding_set_hypothesis,
    #     temperature2tensor=embedding_set_stochastic,
    #     torch_device=torch_device
    # )
    # mmd_trajectory = MmdTrajectory(
    #     values_index=np.array(mmd_flagger_trajectory.tau_parameter),
    #     values_trajectory=np.array(mmd_flagger_trajectory.mmd_distances)
    # )
    mmd_trajectory = MmdTrajectory.from_MmdErrorFlagResultVer3(mmd_flagger_trajectory)
    
    mmd_matrix = _compute_distance_inner_stochastic_samples(
        mmd_estimator=mmd_estimator,
        temperature2tensor=embedding_set_stochastic,
        torch_device=torch_device
    )

    # _shape_trajectory = classify_function_shape(
    #     x=np.array(mmd_flagger_trajectory.tau_parameter), 
    #     y=np.array(mmd_flagger_trajectory.mmd_distances), 
    #     type_filter='no_filter')

    seq_metric_res = []
    for _metric_name in metrics:
        _v = score_mmd_trajectory_variability(_metric_name, mmd_matrix=mmd_matrix, mmd_trajectory=mmd_trajectory)

        seq_metric_res.append(dict(
            metric=_metric_name,
            value=_v))
    # end for

    _o = MmdFlaggerScoreBaseVer1(
        mmd_trajectory=mmd_trajectory,
        mmd_matrix=mmd_matrix,
        metrics=seq_metric_res,
        trajectory_shape=mmd_flagger_trajectory.trajectory_shape
    )

    return _o


# ----- plot functions ----


# def plot_jitter(ax: Axes,
#                 mmd_trajectory: MmdTrajectory,
#                 mmd_matrix: MmdMatrix) -> Axes:
#     """Plot Jitter plot that visualizes the relation of MMD trajectory and MMD matrix."""

#     sns.lineplot(x=mmd_trajectory.values_index, y=mmd_trajectory.values_trajectory, ax=ax)
#     ax.set_xticks(mmd_trajectory.values_index)

#     # Jitter the dots for each tau
#     for i, tau in enumerate(mmd_matrix.values_index):
#         _y_values = mmd_matrix.values_matrix[i, :] # Get the 10 distance values for this tau
        
#         # Add random jitter to the x-axis
#         jitter_x = tau + np.random.uniform(-0.02, 0.02, size=len(_y_values))
#         ax.scatter(jitter_x, _y_values, c='red', s=50, alpha=0.7)
#     # end if

#     return ax