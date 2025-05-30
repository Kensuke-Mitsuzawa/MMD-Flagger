from pathlib import Path

from hallucination_mt.module_flagging import mmd_error_flagger_ver1
from hallucination_mt.module_flagging import utils

from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset

import pandas as pd
import pytest
import toml
import torch
import tempfile

from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator


import logzero
logger = logzero.logger


def func_case_one(model_encoder_decoder_mt,
                  seq_calibration_text,
                  seq_eval_record):
    """In case one, the detection result should be all no-error.
    The temperature is set to 0.2 and 1.5.
    So, the population-a should be always closer to the given translation text.
    """
    path_cache_dir = Path(tempfile.mkdtemp())
    path_cache_dir.mkdir(parents=True, exist_ok=True)

    mmd_flagger = mmd_error_flagger_ver1.MmdErrorFlaggerVer1(
        model_encoder_decoder_mt=model_encoder_decoder_mt,
        n_sampling=50,
        temperature_low=0.2,
        temperature_high=1.5,
        seq_calibration_text=seq_calibration_text,
        path_cache_dir=path_cache_dir)
    seq_input_eval = [
        mmd_error_flagger_ver1.EvaluationTargetTranslationPair(
            target=__record.translation,
            source=__record.source,
            sentence_id=str(__record.sentence_id)
        ) for __record in seq_eval_record]
    flag_result = mmd_flagger.flag_hallucination(seq_input_eval)
    
    n_error_flag = 0
    for __record in flag_result:
        if __record.is_hallucination:
            n_error_flag += 1
        # end if
    # end for

    ratio_error_flag = n_error_flag / len(flag_result)
    assert ratio_error_flag < 0.05, f"ratio_error_flag: {ratio_error_flag}"
    
    # check if there are files at cache.
    assert (path_cache_dir / 'temperature_0.2').exists()
    assert (path_cache_dir / 'temperature_1.5').exists()

    assert len(list((path_cache_dir / 'temperature_0.2').rglob('*json'))) == len(seq_eval_record), f"len: {len(list((path_cache_dir / 'temperature_0.2').rglob('*json')))}"
    assert len(list((path_cache_dir / 'temperature_1.5').rglob('*json'))) == len(seq_eval_record), f"len: {len(list((path_cache_dir / 'temperature_1.5').rglob('*json')))}"

    assert len(list((path_cache_dir / 'temperature_0.2').rglob('*pt'))) == len(seq_eval_record), f"len: {len(list((path_cache_dir / 'temperature_0.2').rglob('*json')))}"
    assert len(list((path_cache_dir / 'temperature_1.5').rglob('*pt'))) == len(seq_eval_record), f"len: {len(list((path_cache_dir / 'temperature_1.5').rglob('*json')))}"

    import shutil
    shutil.rmtree(path_cache_dir)

def func_case_two(model_encoder_decoder_mt,
                  seq_calibration_text,
                  seq_eval_record):
    """In case two, the detection result should be with a lot of errors.
    The temperature is set to 1.5 and 0.2.
    """
    path_cache_dir = Path(tempfile.mkdtemp())
    path_cache_dir.mkdir(parents=True, exist_ok=True)

    mmd_flagger = mmd_error_flagger_ver1.MmdErrorFlaggerVer1(
        model_encoder_decoder_mt=model_encoder_decoder_mt,
        n_sampling=50,
        temperature_low=1.5,
        temperature_high=0.15,
        seq_calibration_text=seq_calibration_text,
        path_cache_dir=path_cache_dir)
    seq_input_eval = [
        mmd_error_flagger_ver1.EvaluationTargetTranslationPair(
            target=__record.translation,
            source=__record.source,
            sentence_id=str(__record.sentence_id)
        ) for __record in seq_eval_record]
    flag_result = mmd_flagger.flag_hallucination(seq_input_eval)
    
    n_error_flag = 0
    for __record in flag_result:
        if __record.is_hallucination:
            n_error_flag += 1
        # end if
    # end for

    ratio_error_flag = n_error_flag / len(flag_result)
    logger.debug(f"ratio_error_flag: {ratio_error_flag}")

    import shutil
    shutil.rmtree(path_cache_dir)


def test_mmd_error_flagger_ver1(resource_path_root: Path):
    path_fairseq_model = resource_path_root / "model_guerreiro_2023"
    assert path_fairseq_model.exists()

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=Path(path_fairseq_model / "wmt18_de-en"),
        path_fairseq_model_file=Path(path_fairseq_model / "checkpoint_best.pt"),
        path_sentencepiece_model=Path(path_fairseq_model / "sentencepiece_models/sentencepiece.joint.bpe.model")
    )    
    

    # loading the dataset (for test)
    path_dataset_tsv = Path(resource_path_root / "eval_datasets/annotated_corpus_checkpoint_2025_03_03_14h.tsv")
    seq_dataset = load_dataset(path_dataset_tsv, delimiter="\t")
    seq_correct_translation = [__record for __record in seq_dataset if __record.error_type == "correct"]
    
    seq_calibration_record = seq_correct_translation[:200]
    seq_eval_record = seq_correct_translation[100:105]

    seq_calibration_text = [__r.translation for __r in seq_calibration_record ]

    func_case_one(model_encoder_decoder_mt, seq_calibration_text, seq_eval_record)
    func_case_two(model_encoder_decoder_mt, seq_calibration_text, seq_eval_record)


def test_without_kernel_calibration(resource_path_root: Path):
    path_fairseq_model = resource_path_root / "model_guerreiro_2023"
    assert path_fairseq_model.exists()

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=Path(path_fairseq_model / "wmt18_de-en"),
        path_fairseq_model_file=Path(path_fairseq_model / "checkpoint_best.pt"),
        path_sentencepiece_model=Path(path_fairseq_model / "sentencepiece_models/sentencepiece.joint.bpe.model")
    )    
    

    # loading the dataset (for test)
    path_dataset_tsv = Path(resource_path_root / "eval_datasets/annotated_corpus_checkpoint_2025_03_03_14h.tsv")
    
    seq_dataset = load_dataset(path_dataset_tsv, delimiter="\t")
    seq_correct_translation = [__record for __record in seq_dataset if __record.error_type == "correct"]
    
    seq_calibration_record = seq_correct_translation[:100]
    # seq_eval_record = seq_correct_translation[100:102]
    seq_eval_record = [__obj for __obj in seq_dataset if __obj.error_type == "hallucination"][:10]

    n_dim_size = 512

    kernel_obj = QuadraticKernelGaussianKernel(
        ard_weights=torch.nn.Parameter(torch.ones(n_dim_size), requires_grad=True),
        bandwidth=torch.nn.Parameter(torch.ones(n_dim_size), requires_grad=True)
    )
    mmd_estimator = QuadraticMmdEstimator(kernel_obj, variance_term='sutherland_2017')

    mmd_flagger = mmd_error_flagger_ver1.MmdErrorFlaggerVer1(
        model_encoder_decoder_mt=model_encoder_decoder_mt,
        n_sampling=50,
        temperature_low=0.2,
        temperature_high=1.5,
        mmd_estimator=mmd_estimator)
    seq_input_eval = [
        mmd_error_flagger_ver1.EvaluationTargetTranslationPair(
            target=__record.translation,
            source=__record.source,
            sentence_id=str(__record.sentence_id)
        ) for __record in seq_eval_record]
    flag_result = mmd_flagger.flag_hallucination(seq_input_eval)
    
    n_error_flag = 0
    for __record in flag_result:
        if __record.is_hallucination:
            n_error_flag += 1
        # end if
    # end for

    ratio_error_flag = n_error_flag / len(flag_result)
    assert ratio_error_flag < 0.05, f"ratio_error_flag: {ratio_error_flag}"
