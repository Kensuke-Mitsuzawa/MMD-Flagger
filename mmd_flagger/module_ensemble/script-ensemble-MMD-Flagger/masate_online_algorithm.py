import json
import numpy as np
from scipy.stats import entropy

def detect_hallucination_masate_online(score_aux_records: list) -> float:
    """
    Model-Agnostic Skewness-Adaptive Trajectory Ensemble Online (MASATE-Online)
    
    A completely standalone, zero-calibration, and single-sample online 
    hallucination detection algorithm that requires no database or reference set.
    
    INPUT:
      - score_aux_records (list): List of score_aux JSON strings or dictionaries 
        containing the MMD trajectory and matrix data for a single inference.
        
    RETURNS:
      - float: The MASATE-Online uncertainty score.
    """
    active_scores = []
    
    for record in score_aux_records:
        if not record:
            continue
        if isinstance(record, str):
            try:
                aux = json.loads(record)
            except Exception:
                continue
        else:
            aux = record
            
        mmd_traj = aux.get("mmd_traj", {}) or {}
        mmd_matrix = aux.get("mmd_matrix", {}) or {}
        
        distances = mmd_traj.get("mmd_distances", []) or []
        if not distances:
            continue
            
        distances = [float(d) for d in distances if np.isfinite(d)]
        if len(distances) < 3:
            continue
            
        # Unsupervised Activity Filtering:
        # Inactive domains have extremely flat trajectories (std near 0).
        # We only aggregate active domains with high-signal representations.
        std_val = np.std(distances)
        if std_val < 1e-5:
            continue
            
        # 1. 1D Coefficient of Variation (Scale-free spread indicator)
        mean_val = np.mean(distances)
        cv = std_val / (mean_val + 1e-8)
        
        # 2. 2D Concentrated Energy (Entropy-based localized spike indicator)
        matrix_vals = mmd_matrix.get("values_matrix", []) or []
        if matrix_vals:
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
