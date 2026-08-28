import typing as ty
import json
import numpy as np
from scipy.stats import entropy

import logging

from ..module_mmd_flagger.mmd_flagger import EstimateReturnObject

logger = logging.getLogger(__name__)


def detect_hallucination_masate_online(
    feat2mmd_flagger_trajectory_obj: ty.Dict[str, EstimateReturnObject]
    ) -> float:
    """
    Model-Agnostic Skewness-Adaptive Trajectory Ensemble Online (MASATE-Online)
    
    A completely standalone, zero-calibration, and single-sample online 
    hallucination detection algorithm that requires no database or reference set.
    
    INPUT:
      - feat2mmd_flagger_trajectory_obj: 
                  
    RETURNS:
      - float: The MASATE-Online uncertainty score.
    """
    active_scores = []
    
    _mmd_flagger_obj: EstimateReturnObject

    if len(feat2mmd_flagger_trajectory_obj) == 0:
        logger.warning(f"No feature and MMD-objects are given. Return 0.0 as hallucination score.")
        return 0.0
    # end if

    if len(feat2mmd_flagger_trajectory_obj) == 1:
        logger.warning(f"Only one feature is given. Return 0.0 as hallucination score.")
        return 0.0
    # end if

    for _feat, _mmd_flagger_obj in feat2mmd_flagger_trajectory_obj.items():            
        mmd_traj = _mmd_flagger_obj.mmd_traj
        mmd_matrix = _mmd_flagger_obj.mmd_matrix
        
        distances = mmd_traj.mmd_distances
        if not distances:
            logger.warning(f"No distances found for feature {_feat}")
            continue
            
        distances = [float(d) for d in distances if np.isfinite(d)]
        if len(distances) < 3:
            logger.warning(f"Less than 3 finite distances for feature {_feat}")
            continue
            
        # Unsupervised Activity Filtering:
        # Inactive domains have extremely flat trajectories (std near 0).
        # We only aggregate active domains with high-signal representations.
        std_val = np.std(distances)
        if std_val < 1e-5:
            logger.warning(f"Feature {_feat} has very small std_val: {std_val}")
            continue
            
        # 1. 1D Coefficient of Variation (Scale-free spread indicator)
        mean_val = np.mean(distances)
        cv = std_val / (mean_val + 1e-8)
        
        # 2. 2D Concentrated Energy (Entropy-based localized spike indicator)
        matrix_vals = mmd_matrix.values_matrix
        if len(matrix_vals) == 0:
            matrix_flat = np.array(matrix_vals).flatten()
            matrix_flat = matrix_flat[np.isfinite(matrix_flat)]
            matrix_flat = np.clip(matrix_flat, 0, None)
            total_sum = matrix_flat.sum()
            if total_sum > 1e-8:
                p = matrix_flat / total_sum
                mat_entropy = entropy(p)
                max_entropy = np.log(len(p))
                concentrated_energy = 1.0 - (mat_entropy / (max_entropy + 1e-8))
            else:
                concentrated_energy = 0.0
        else:
            concentrated_energy = 0.0
            
        # Unsupervised local domain score combining 1D and 2D signals
        domain_score = cv + 2.0 * concentrated_energy
        active_scores.append(domain_score)
        
    if not active_scores:
        return 0.0
        
    return float(np.mean(active_scores))

if __name__ == "__main__":
    print("MASATE-Online Algorithm loaded successfully.")
