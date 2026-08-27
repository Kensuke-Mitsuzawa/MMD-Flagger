import duckdb
import json
import torch
import io
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy

# Import the evaluation function from the workspace package
import sys
sys.path.append('/root/vast-ai-setup/llm_decoding_comparison')
from llm_decoding_comparison.module_eval_metrics import evaluate_estimator_quality_lm_polygraph

from masate_online_algorithm import detect_hallucination_masate_online

def main():
    runs = {
        'run-2': 'run-2',
        'run-adhoc-top10': 'run-adhoc-additional-features-issue-205',
        'run-adhoc-top100-PCA': 'run-adhoc-additional-features-issue-205-run2'
    }

    data_records = []
    
    for run_key, run_dir in runs.items():
        db_path = f"/root/vast-ai-setup/llm_decoding_comparison/workdir/opt-MMD-flagger/midium-MMD-Flagger-2026-05-19-LLM-Inference-standard-HPC-Design/{run_dir}/detection_record_db.duckdb"
        if not os.path.exists(db_path):
            continue
        
        conn = duckdb.connect(db_path)
        param_rows = conn.execute("SELECT record_aggregation_key, llm_identifier, algorithm_name, algorithm_configuration, feature_name, feature_aggregation_method FROM ParameterRecord").fetchall()
        
        param_dict = {}
        for row in param_rows:
            agg_key, llm, algo, algo_conf_json, feat_name, feat_agg = row
            algo_conf = json.loads(algo_conf_json)
            
            if algo != 'MMD-Flagger':
                continue
            if 'text' in feat_name or 'TextEmbedding' in feat_name:
                continue
            
            param_dict[agg_key] = {
                'feature_name': feat_name,
                'run_key': run_key
            }
            
        score_rows = conn.execute("SELECT target_record_primary_key, label_hallucination_level, record_aggregation_key, llm_identifier, score_aux FROM ScoringResultRecord").fetchall()
        for s_row in score_rows:
            target_key, label_level, agg_key, llm, score_aux = s_row
            if agg_key in param_dict:
                meta = param_dict[agg_key]
                rec = {
                    'run': run_key,
                    'llm': llm,
                    'target_key': target_key,
                    'label_level': label_level,
                    'label': 1 if label_level >= 0.5 else 0,
                    'score_aux': score_aux,
                    'feat_name': meta['feature_name']
                }
                data_records.append(rec)
        conn.close()
        
    df = pd.DataFrame(data_records)
    llms = df['llm'].unique()
    
    print("\n=========================================\nFINAL MASATE-ONLINE BENCHMARK\n=========================================")
    
    results = []
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
        if len(df_scored) > 5:
            auc = evaluate_estimator_quality_lm_polygraph('ROCAUC', scores=df_scored['score'].tolist(), labels=df_scored['label'].tolist())
            print(f"LLM: {llm:36s} | Samples: {len(df_scored):3d} | ROC-AUC: {auc:.4f}")
            results.append({
                'LLM': llm,
                'ROC-AUC': auc,
                'Samples': len(df_scored)
            })
            
    # Write summary to a clean file
    out_path = '/root/vast-ai-setup/llm_decoding_comparison/workdir/doc-opt-5-MMD-Flagger/results_masate_online.txt'
    with open(out_path, 'w') as f:
        f.write("=========================================\n")
        f.write("FINAL MASATE-ONLINE BENCHMARK RESULTS\n")
        f.write("=========================================\n")
        for res in results:
            f.write(f"LLM: {res['LLM']:36s} | Samples: {res['Samples']:3d} | ROC-AUC: {res['ROC-AUC']:.4f}\n")

if __name__ == '__main__':
    main()
