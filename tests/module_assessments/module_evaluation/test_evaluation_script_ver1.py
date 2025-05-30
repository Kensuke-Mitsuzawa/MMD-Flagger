from pathlib import Path
import tempfile
import shutil

from hallucination_mt.module_assessments.module_evaluation import evaluation_script_ver1
from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset

from hallucination_mt.module_assessments.module_management_db.module_db_record import (
    DbTableRecordRaunak2021,
    DbTableRecordGuerreiro2023McDSIM
)


def test_evaluation_runner(resource_path_root: Path):
    path_dataset_tsv = resource_path_root / "test_dataset.tsv"
    path_prediction_database = resource_path_root / "management_db_script_ver1.sqlite3"

    seq_dataset_record = load_dataset(path_dataset_tsv, delimiter='\t')

    eval_runner = evaluation_script_ver1.EvaluationVer1(
        seq_dataset_record=seq_dataset_record,
        path_prediction_database=path_prediction_database)
    
    seq_eval_table_name = [
        DbTableRecordRaunak2021.__name__,
    ]

    path_dir_tmp = Path(tempfile.mkdtemp())
    path_dir_tmp.mkdir(parents=True, exist_ok=True)
 
    eval_runner.main(
        path_output_dir=path_dir_tmp,
        config_name="test", 
        seq_eval_table_name=seq_eval_table_name)

    shutil.rmtree(path_dir_tmp)