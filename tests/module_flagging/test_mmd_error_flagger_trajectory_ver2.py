import typing as ty
from pathlib import Path
import toml
import tempfile
import shutil
import os

import numpy as np
import torch

from hallucination_mt.module_flagging import utils
from hallucination_mt.module_flagging.mmd_error_flagger_trajectory_ver2 import (
    MmdErrorFlaggerTrajectoryVer2,
    EvaluationTargetTranslationPair)
from hallucination_mt.module_translation_handler.ver1 import (
    FaiseqTranslationModelHandler,
    TransformersTranslationModelHandler
)
from hallucination_mt.module_hidden_vector_extractor.ver1 import (
    FairSeqVectorExtractor,
    TransformerVectorExtractor
)


from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset
from hallucination_mt import dale_2023_halomi




def test_MmdErrorFlaggerTrajectoryVer2_fairseq(resource_path_root: Path):
    path_config = resource_path_root / "config.toml"
    assert path_config.exists(), f"path_config={path_config}"

    with open(path_config, "r") as f:
        config_obj = toml.load(f)
        assert "path_fairseq_model" in config_obj
        assert "path_dataset" in config_obj
    # end with

    config_obj_fairseq_model = config_obj["path_fairseq_model"]
    config_obj_dataset = config_obj["path_dataset"]

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=Path(config_obj_fairseq_model["path_fairseq_model_dir"]),
        path_fairseq_model_file=Path(config_obj_fairseq_model["path_fairseq_model_file"]),
        path_sentencepiece_model=Path(config_obj_fairseq_model["path_sentencepiece_model"])
    )    

    path_tmp_dir = Path(tempfile.mkdtemp())
    path_tmp_dir.mkdir(parents=True, exist_ok=True)

    path_dataset_tsv = Path(config_obj_dataset["path_dataset_tsv"])
    seq_dataset = load_dataset(path_dataset_tsv, delimiter="\t")
    seq_correct_translation = [__record for __record in seq_dataset if __record.error_type == "correct"]
    
    seq_calibration_record = seq_correct_translation[:200]
    # seq_eval_record = seq_correct_translation[100:105]

    seq_calibration_text = [__r.translation for __r in seq_calibration_record ]

    translation_handler = FaiseqTranslationModelHandler(model_encoder_decoder_mt=model_encoder_decoder_mt)
    vector_extractor = FairSeqVectorExtractor(model=model_encoder_decoder_mt)


    mmd_flagger = MmdErrorFlaggerTrajectoryVer2(
        translation_handler=translation_handler,
        vector_extractor=vector_extractor,
        path_cache_dir=path_tmp_dir,
        seq_calibration_text=seq_calibration_text
    )
    
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
    assert (Path(mmd_flagger.path_cache_dir) / mmd_flagger.path_cache_database_translation).exists()
    assert (Path(mmd_flagger.path_cache_dir) / mmd_flagger.path_cache_database_embedding).exists()

    result_flag_second = mmd_flagger.flag_hallucination_one_record(
        eval_target=input_obj,
        candidate_temperature_parameters=candidate_temperature_parameters, 
        n_sampling=25)
    
    assert result_flag.trajectory_shape == "saddle-point", f"result_flag.trajectory_shape={result_flag.trajectory_shape}"

    shutil.rmtree(path_tmp_dir)


def test_MmdErrorFlaggerTrajectoryVer2_transformer(resource_path_root: Path):


    path_tmp_dir = Path(tempfile.mkdtemp())
    path_tmp_dir.mkdir(parents=True, exist_ok=True)

    src_lang = "deu_Latn"
    target_lang = "eng_Latn"

    path_dataset_tsv = resource_path_root / 'eval_datasets' / 'halomi_core.tsv'
    assert path_dataset_tsv.exists(), f"path_dataset_tsv={path_dataset_tsv}"
    seq_dataset = dale_2023_halomi.load_dataset.load_dataset(path_dataset_tsv)
    seq_correct_translation = [
        __record for __record in seq_dataset 
        if __record.error_type == "correct" and __record.src_lang == src_lang and __record.tgt_lang == target_lang]
    assert len(seq_correct_translation) > 0, f"len(seq_correct_translation)={len(seq_correct_translation)}"
    seq_calibration_record = seq_correct_translation[:100]

    seq_calibration_text = [__r.tgt_text for __r in seq_calibration_record ]
    translation_handler = TransformersTranslationModelHandler(src_lang=src_lang, target_lang=target_lang, model_name='facebook/nllb-200-distilled-600M')
    vector_extractor = TransformerVectorExtractor(translation_handler)


    mmd_flagger = MmdErrorFlaggerTrajectoryVer2(
        translation_handler=translation_handler,
        vector_extractor=vector_extractor,
        path_cache_dir=path_tmp_dir,
        seq_calibration_text=seq_calibration_text
    )
    
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
    assert (Path(mmd_flagger.path_cache_dir) / mmd_flagger.path_cache_database_translation).exists()
    assert (Path(mmd_flagger.path_cache_dir) / mmd_flagger.path_cache_database_embedding).exists()

    result_flag_second = mmd_flagger.flag_hallucination_one_record(
        eval_target=input_obj,
        candidate_temperature_parameters=candidate_temperature_parameters, 
        n_sampling=25)
    
    shutil.rmtree(path_tmp_dir)
