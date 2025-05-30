import typing as ty
from pathlib import Path
import toml
import tempfile
import shutil
import os

import numpy as np
import torch

from hallucination_mt.module_flagging import utils
from hallucination_mt.module_flagging.mmd_error_flagger_trajectory_ver1 import MmdErrorFlaggerTrajectoryVer1, EvaluationTargetTranslationPair
from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset


def test_MmdErrorFlaggerTrajectoryVer1(resource_path_root: Path):

    path_fairseq_model = resource_path_root / "model_guerreiro_2023"
    assert path_fairseq_model.exists()

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=Path(path_fairseq_model / "wmt18_de-en"),
        path_fairseq_model_file=Path(path_fairseq_model / "checkpoint_best.pt"),
        path_sentencepiece_model=Path(path_fairseq_model / "sentencepiece_models/sentencepiece.joint.bpe.model")
    )    

    path_tmp_dir = Path(tempfile.mkdtemp())
    path_tmp_dir.mkdir(parents=True, exist_ok=True)

    path_dataset_tsv = Path(resource_path_root / "eval_datasets/annotated_corpus_checkpoint_2025_03_03_14h.tsv")
    seq_dataset = load_dataset(path_dataset_tsv, delimiter="\t")
    seq_correct_translation = [__record for __record in seq_dataset if __record.error_type == "correct"]
    
    seq_calibration_record = seq_correct_translation[:200]
    # seq_eval_record = seq_correct_translation[100:105]

    seq_calibration_text = [__r.translation for __r in seq_calibration_record ]

    mmd_flagger = MmdErrorFlaggerTrajectoryVer1(
        model_encoder_decoder_mt=model_encoder_decoder_mt,
        path_cache_dir=path_tmp_dir,
        seq_calibration_text=seq_calibration_text)
    
    input_obj = EvaluationTargetTranslationPair(
        source="Freundlichkeit des Frühstückpersonals und Qualität des Frühstück erstklassig, alles ausreichend da.",
        target="The staff were very friendly and helpful. The room was clean and comfortable.",
        sentence_id=str(3044)
    )
    candidate_temperature_parameters = np.arange(0.1, 0.8, 0.05)
    result_flag = mmd_flagger.flag_hallucination_one_record(eval_target=input_obj,
                                              candidate_temperature_parameters=candidate_temperature_parameters, 
                                              n_sampling=25)
    # check the cache db
    assert (Path(mmd_flagger.path_cache_dir) / mmd_flagger.path_cache_database).exists()

    result_flag_second = mmd_flagger.flag_hallucination_one_record(
        eval_target=input_obj,
        candidate_temperature_parameters=candidate_temperature_parameters, 
        n_sampling=25)
    
    assert result_flag.trajectory_shape == "saddle-point", f"result_flag.trajectory_shape={result_flag.trajectory_shape}"

    shutil.rmtree(path_tmp_dir)