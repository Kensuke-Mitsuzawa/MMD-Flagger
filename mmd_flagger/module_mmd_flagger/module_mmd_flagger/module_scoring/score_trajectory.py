import typing as ty

import numpy as np

from ..codebase_ver3_1.models import MmdErrorFlagResultVer3


"""module to score the MMD trajectory"""

PossibleScoreMetrics = ty.Literal['tau_min_and_tauarg_min', 'tau_min_and_tau_max_and_tau_argmin']


def score(mmd_flagger_result: MmdErrorFlagResultVer3,
          metric_name: PossibleScoreMetrics) -> float:
    """

    Return:
        larger values indicates more likelihood of hallucination.
        This value can be < 0.0. Minus values indicate less likelihood more hallucination.
    """
    array_tau = np.array(mmd_flagger_result.tau_parameter)
    array_mmd = np.array(mmd_flagger_result.mmd_distances)

    mmd_at_min_tau = array_mmd[np.argmin(array_tau)]
    mmd_at_argmin_tau = np.min(array_mmd)
    mmd_at_max_tau = array_mmd[np.argmax(array_tau)]

    if metric_name == 'tau_min_and_tauarg_min':
        return mmd_at_min_tau - mmd_at_argmin_tau
    elif metric_name == 'tau_min_and_tau_max_and_tau_argmin':
        return (mmd_at_min_tau - mmd_at_argmin_tau) + (mmd_at_max_tau - mmd_at_argmin_tau)
    else:
        raise NotImplementedError(f'No named metric named {metric_name}')



