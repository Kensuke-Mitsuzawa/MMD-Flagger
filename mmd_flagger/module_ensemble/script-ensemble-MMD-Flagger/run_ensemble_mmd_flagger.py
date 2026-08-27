#!/usr/bin/env python3
"""
MASATE-Online (doc-opt-5-MMD-Flagger) Calculator

Independent Python script to compute Model-Agnostic Skewness-Adaptive Trajectory 
Ensemble Online (MASATE-Online) from doc-opt-5-MMD-Flagger on DuckDB trajectory records.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

# Add doc-opt-5-MMD-Flagger to path to import official algorithm function
DOC_OPT_5_DIR = "/home/kmitsuzawa/codes/uca-mmd-flagger/llm_decoding_comparison/scripts/preliminary_scripts/ensemble-MMD-Flagger-model/doc-opt-5-MMD-Flagger"
if DOC_OPT_5_DIR not in sys.path:
    sys.path.append(DOC_OPT_5_DIR)

from masate_online_algorithm import detect_hallucination_masate_online


def evaluate_duckdb_dataset(db_path: str, output_dir: str = None) -> dict:
    """
    Run MASATE-Online (doc-opt-5-MMD-Flagger) computation on a DuckDB database.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")
        
    conn = duckdb.connect(db_path)
    
    # Query parameters to filter MMD-Flagger feature records (excluding text embeddings)
    param_rows = conn.execute(
        "SELECT record_aggregation_key, llm_identifier, algorithm_name, feature_name, feature_aggregation_method "
        "FROM ParameterRecord"
    ).fetchall()
    
    param_dict = {}
    for row in param_rows:
        agg_key, llm, algo, feat_name, feat_agg = row
        if algo != 'MMD-Flagger':
            continue
        if 'text' in feat_name or 'TextEmbedding' in feat_name:
            continue
        param_dict[agg_key] = {
            'llm': llm,
            'feature_name': feat_name,
            'feature_agg': feat_agg
        }
        
    if not param_dict:
        conn.close()
        raise ValueError("No MMD-Flagger parameters found in ParameterRecord table.")
        
    score_rows = conn.execute(
        "SELECT target_record_primary_key, label_hallucination_level, record_aggregation_key, llm_identifier, score_aux "
        "FROM ScoringResultRecord"
    ).fetchall()
    conn.close()
    
    data_records = []
    for s_row in score_rows:
        target_key, label_level, agg_key, llm, score_aux = s_row
        if agg_key in param_dict:
            data_records.append({
                'llm': llm,
                'target_key': target_key,
                'label_level': label_level,
                'label': 1 if label_level >= 0.5 else 0,
                'score_aux': score_aux
            })
            
    df = pd.DataFrame(data_records)
    if df.empty:
        raise ValueError("No matching ScoringResultRecords found for MMD-Flagger.")
        
    llms = df['llm'].unique()
    results = []
    
    print("\n=========================================")
    print("MASATE-ONLINE BENCHMARK (DOC-OPT-5-MMD-FLAGGER)")
    print("=========================================")
    print(f"Database: {db_path}\n")
    
    for llm in llms:
        df_llm = df[df['llm'] == llm]
        scored_samples = []
        
        for target_key, group in df_llm.groupby('target_key'):
            records = group['score_aux'].tolist()
            score = detect_hallucination_masate_online(records)
            label = group['label'].iloc[0]
            scored_samples.append({
                'target_key': target_key,
                'score': score,
                'label': label
            })
            
        df_scored = pd.DataFrame(scored_samples)
        n_samples = len(df_scored)
        
        if n_samples > 1 and len(df_scored['label'].unique()) > 1:
            auc = float(roc_auc_score(df_scored['label'], df_scored['score']))
            print(f"LLM: {llm:40s} | Samples: {n_samples:4d} | ROC-AUC: {auc:.4f}")
            results.append({
                'LLM': llm,
                'Samples': n_samples,
                'ROC-AUC': auc
            })
        else:
            print(f"LLM: {llm:40s} | Samples: {n_samples:4d} | ROC-AUC: N/A")
            results.append({
                'LLM': llm,
                'Samples': n_samples,
                'ROC-AUC': float('nan')
            })
            
    summary_filepath = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_filepath = os.path.join(output_dir, "results_ensemble_mmd_flagger.txt")
        with open(summary_filepath, "w") as f:
            f.write("=========================================\n")
            f.write("MASATE-ONLINE BENCHMARK RESULTS (DOC-OPT-5)\n")
            f.write("=========================================\n")
            f.write(f"Source DB: {db_path}\n\n")
            for res in results:
                auc_str = f"{res['ROC-AUC']:.4f}" if not np.isnan(res['ROC-AUC']) else "N/A"
                f.write(f"LLM: {res['LLM']:40s} | Samples: {res['Samples']:4d} | ROC-AUC: {auc_str}\n")
        print(f"\nResults successfully written to: {summary_filepath}")
        
    return {
        'results': results,
        'summary_filepath': summary_filepath
    }


def main():
    default_db_path = "/home/kmitsuzawa/dvc-repository/dvc-mmd-flagger/Reproducing-KLE-Nikitin-2026-06-24/estimators/Reproducing-KLE-Nikitin-2026-06-24-TriviaQA-LLM-inference-for-MMD-Flagger-run-1/detection_record_db.duckdb"
    default_out_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Compute MASATE-Online (doc-opt-5) score on DuckDB database.")
    parser.add_argument("--db-path", type=str, default=default_db_path, help="Path to DuckDB database")
    parser.add_argument("--output-dir", type=str, default=default_out_dir, help="Path to save output results")
    args = parser.parse_args()
    
    evaluate_duckdb_dataset(args.db_path, args.output_dir)


if __name__ == '__main__':
    main()
