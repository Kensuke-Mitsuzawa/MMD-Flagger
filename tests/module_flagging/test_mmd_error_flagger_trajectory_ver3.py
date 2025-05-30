import typing as ty
from pathlib import Path
import toml
import tempfile
import shutil
import os
import pickle
import zlib

import numpy as np
import torch

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator

from hallucination_mt.module_flagging import utils
from hallucination_mt.module_flagging.mmd_error_flagger_trajectory_ver3 import (
    MmdErrorFlaggerTrajectoryVer3,
    EvaluationTargetTranslationPair)
from hallucination_mt.module_flagging.module_mmd_error_flagger_trajectory_ver3 import (
    TensorPreprocessorVer1,
    MmdEstimatorInitialiserVer1
)
from hallucination_mt.module_translation_handler.ver2 import (
    FairSeqTranslationModelHandlerVer2,
    TransformersTranslationModelHandlerVer2,
    TranslationResultContainer
)
from hallucination_mt.module_hidden_vector_extractor.ver2 import (
    TransformerVectorExtractorVer2,
    FairSeqVectorExtractorVer2
)


from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset
from hallucination_mt import dale_2023_halomi


def pre_routine_fairseq(resource_path_root: Path):
    path_dir_mode = resource_path_root / 'model_guerreiro_2023'
    assert path_dir_mode.exists()

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=path_dir_mode / 'wmt18_de-en',
        path_fairseq_model_file=path_dir_mode / 'checkpoint_best.pt',
        path_sentencepiece_model=path_dir_mode / 'sentencepiece_models/sentencepiece.joint.bpe.model'
    )    

    path_dataset_tsv = resource_path_root / 'eval_datasets/lfan_hall_subset.tsv'
    seq_dataset = load_dataset(path_dataset_tsv, delimiter="\t")
    seq_correct_translation = [__record for __record in seq_dataset if __record.error_type == "correct"]

    return model_encoder_decoder_mt, seq_correct_translation


def pre_routine_transformer(resource_path_root: Path):
    src_lang = "deu_Latn"
    target_lang = "eng_Latn"


    path_dataset_tsv = resource_path_root / 'eval_datasets/halomi_deu_eng_subset.tsv'
    assert path_dataset_tsv.exists(), f"path_dataset_tsv={path_dataset_tsv}"
    seq_dataset = dale_2023_halomi.load_dataset.load_dataset(path_dataset_tsv)
    seq_correct_translation = [
        __record for __record in seq_dataset 
        if __record.error_type == "correct" and __record.src_lang == src_lang and __record.tgt_lang == target_lang]

    return seq_correct_translation, src_lang, target_lang


def base_procedure_fairseq_based(resource_path_root: Path,
                                 mode_vector_preprocess: str,
                                 mode_target_embedding_layer: str,
                                 mode_max_token_length_vector_concat: str = 'max_calibration'):
    path_tmp_dir = Path(tempfile.mkdtemp())
    path_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    path_dir_mode = resource_path_root / 'model_guerreiro_2023'

    model_encoder_decoder_mt, seq_correct_translation = pre_routine_fairseq(resource_path_root)

    # seq_calibration_record = seq_correct_translation[:200]
    seq_calibration_record = seq_correct_translation[100:105]

    seq_calibration_text = [
        EvaluationTargetTranslationPair(
            source=_r.source,
            target=_r.translation,
            sentence_id=str(_r.sentence_id)
        )
        for _r in seq_calibration_record
    ]

    translation_handler = FairSeqTranslationModelHandlerVer2(
        path_dir_fairseq_model=path_dir_mode / 'wmt18_de-en',
        path_model_checkpoint=path_dir_mode / 'checkpoint_best.pt',
        path_sentencepiece_model=path_dir_mode / 'sentencepiece_models/sentencepiece.joint.bpe.model'
        )
    vector_extractor = FairSeqVectorExtractorVer2(translation_handler)

    tensor_preprocessor = TensorPreprocessorVer1(
        mode_vector_preprocess=mode_vector_preprocess,
        mode_max_token_length_vector_concat=mode_max_token_length_vector_concat
    )
    mmd_init_executor = MmdEstimatorInitialiserVer1(
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer='decoder.word_embedding',
        vector_extractor=vector_extractor
    )
    mmd_estimator = mmd_init_executor.init_gaussian_kernel_mmd_estimator(
        seq_calibration_text=seq_calibration_text,
    )
    
    mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
        vector_extractor=vector_extractor,
        path_cache_dir=path_tmp_dir,
        mmd_estimator=mmd_estimator,
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer=mode_target_embedding_layer,
        option_is_sampling_in_iteration=True,
        option_translation_max_a=1.0,
        option_translation_max_b=10
    )
    
    input_obj = EvaluationTargetTranslationPair(
        source="Freundlichkeit des Frühstückpersonals und Qualität des Frühstück erstklassig, alles ausreichend da.",
        target="The staff were very friendly and helpful. The room was clean and comfortable.",
        sentence_id=str(3044)
    )
    candidate_temperature_parameters = np.array([0.1, 0.3, 0.5, 0.6])
    result_flag = mmd_flagger.flag_hallucination_one_record(eval_target=input_obj,
                                              candidate_temperature_parameters=candidate_temperature_parameters, 
                                              n_sampling=5)
    # check the cache db
        
    result_flag_second = mmd_flagger.flag_hallucination_one_record(
        eval_target=input_obj,
        candidate_temperature_parameters=candidate_temperature_parameters, 
        n_sampling=25)
    
    assert result_flag_second.trajectory_shape == "saddle-point", f"result_flag.trajectory_shape={result_flag.trajectory_shape}"

    shutil.rmtree(path_tmp_dir)    


def base_procedure_transformers(resource_path_root: Path,
                                mode_max_token_length_vector_concat: ty.Union[str, int] = 'max_calibration'):
    pass


# -------------------------------------------------
# Test of basic usage

def test_MmdErrorFlaggerTrajectoryVer3_fairseq_avg_decoder_embedding(resource_path_root: Path):
    base_procedure_fairseq_based(
        resource_path_root,
        mode_vector_preprocess='avg',
        mode_target_embedding_layer='decoder.word_embedding'
    )


def test_MmdErrorFlaggerTrajectoryVer3_transformer(resource_path_root: Path):
    code_name_nllb = "facebook/nllb-200-distilled-600M"
    seq_correct_translation, src_lang, target_lang = pre_routine_transformer(resource_path_root)

    path_tmp_dir = Path(tempfile.mkdtemp())
    path_tmp_dir.mkdir(parents=True, exist_ok=True)

    path_dir_translation_cache = path_tmp_dir / 'cache'
    path_dir_translation_cache.mkdir(parents=True, exist_ok=True)

    assert len(seq_correct_translation) > 0, f"len(seq_correct_translation)={len(seq_correct_translation)}"
    # seq_calibration_record = seq_correct_translation[:100]
    seq_calibration_record = seq_correct_translation[100:105]

    seq_calibration_text = [
        EvaluationTargetTranslationPair(
            source=_r.src_text,
            target=_r.tgt_text,
            sentence_id=str(_r.key_unique)
        )
        for _r in seq_calibration_record
    ]
    translation_handler = TransformersTranslationModelHandlerVer2(
        src_lang=src_lang, 
        target_lang=target_lang, 
        model_name=code_name_nllb,
        path_cache_dir=path_tmp_dir)
    vector_extractor = TransformerVectorExtractorVer2(translation_handler)

    mode_vector_preprocess='avg'
    tensor_preprocessor = TensorPreprocessorVer1(
        mode_vector_preprocess=mode_vector_preprocess,
    )
    mmd_init_executor = MmdEstimatorInitialiserVer1(
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer='decoder.word_embedding',
        vector_extractor=vector_extractor
    )
    mmd_estimator = mmd_init_executor.init_gaussian_kernel_mmd_estimator(
        seq_calibration_text=seq_calibration_text,
    )


    mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
        mmd_estimator=mmd_estimator,
        vector_extractor=vector_extractor,
        tensor_preprocessor=tensor_preprocessor,
        path_cache_dir=path_tmp_dir,
        mode_target_embedding_layer='decoder.word_embedding',
        option_translation_max_a=1.0,
        option_translation_max_b=10
    )
    
    input_obj = EvaluationTargetTranslationPair(
        source="Freundlichkeit des Frühstückpersonals und Qualität des Frühstück erstklassig, alles ausreichend da.",
        target="The staff were very friendly and helpful. The room was clean and comfortable.",
        sentence_id=str(3044)
    )
    candidate_temperature_parameters = np.array([0.1, 0.2, 0.3])
    result_flag = mmd_flagger.flag_hallucination_one_record(eval_target=input_obj,
                                              candidate_temperature_parameters=candidate_temperature_parameters, 
                                              n_sampling=25)
    # check the cache db
    assert Path(mmd_flagger.path_cache_dir)

    result_flag_second = mmd_flagger.flag_hallucination_one_record(
        eval_target=input_obj,
        candidate_temperature_parameters=candidate_temperature_parameters, 
        n_sampling=25)
    
    shutil.rmtree(path_tmp_dir)


# # -------------------------------------------------
# # Test cases: vector concat


# def test_MmdErrorFlaggerTrajectoryVer3_vector_cocats_max_calibration(resource_path_root: Path):
#     options_test_targets = dict(
#             mode_vector_preprocess='concat',
#             mode_target_embedding_layer='word_embedding',
#             mode_max_token_length_vector_concat='max_calibration',
#             kernel_type='gaussian',
#             kernel_length_scale_median_option='dimensionwise'        
#     )

#     def _test_fairseq():
#         path_tmp_dir = Path(tempfile.mkdtemp())
#         path_tmp_dir.mkdir(parents=True, exist_ok=True)
        
#         model_encoder_decoder_mt, seq_correct_translation = pre_routine_fairseq(resource_path_root)

#         seq_calibration_record = seq_correct_translation[:10]
#         # seq_eval_record = seq_correct_translation[100:105]

#         seq_calibration_text = [__r.translation for __r in seq_calibration_record ]

#         translation_handler = FaiseqTranslationModelHandler(model_encoder_decoder_mt=model_encoder_decoder_mt)
#         vector_extractor = FairSeqVectorExtractor(model=model_encoder_decoder_mt)

#         mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
#             translation_handler=translation_handler,
#             vector_extractor=vector_extractor,
#             path_cache_dir=path_tmp_dir,
#             seq_calibration_text=seq_calibration_text,
#             **options_test_targets
#         )
#         # check the gaussian kernel's length scale
#         assert isinstance(mmd_flagger.mmd_estimator.kernel_obj.bandwidth, torch.Tensor)
        
#         input_obj = EvaluationTargetTranslationPair(
#             source="Freundlichkeit des Frühstückpersonals und Qualität des Frühstück erstklassig, alles ausreichend da.",
#             target="The staff were very friendly and helpful. The room was clean and comfortable.",
#             sentence_id=str(3044)
#         )

#         # test the flagger works
#         res = mmd_flagger.flag_hallucination_one_record(input_obj,
#                                                   candidate_temperature_parameters=np.array([0.1, 0.2, 0.3]),
#                                                   n_sampling=5)
#     # end def

#     def _test_transformers():
#         code_name_nllb = "facebook/nllb-200-distilled-600M"
#         seq_correct_translation, src_lang, target_lang = pre_routine_transformer(resource_path_root)

#         path_tmp_dir = Path(tempfile.mkdtemp())
#         path_tmp_dir.mkdir(parents=True, exist_ok=True)

#         assert len(seq_correct_translation) > 0, f"len(seq_correct_translation)={len(seq_correct_translation)}"
#         seq_calibration_record = seq_correct_translation[:110]

#         seq_calibration_text = [__r.tgt_text for __r in seq_calibration_record ]
#         translation_handler = TransformersTranslationModelHandler(src_lang=src_lang, target_lang=target_lang, model_name=code_name_nllb)
#         vector_extractor = TransformerVectorExtractor(translation_handler)


#         mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
#             translation_handler=translation_handler,
#             vector_extractor=vector_extractor,
#             path_cache_dir=path_tmp_dir,
#             seq_calibration_text=seq_calibration_text,
#             **options_test_targets
#         )
        
#         input_obj = EvaluationTargetTranslationPair(
#             source="Freundlichkeit des Frühstückpersonals und Qualität des Frühstück erstklassig, alles ausreichend da.",
#             target="The staff were very friendly and helpful. The room was clean and comfortable.",
#             sentence_id=str(3044)
#         )
#         candidate_temperature_parameters = np.array([0.1, 0.2])
#         result_flag = mmd_flagger.flag_hallucination_one_record(eval_target=input_obj,
#                                                 candidate_temperature_parameters=candidate_temperature_parameters, 
#                                                 n_sampling=10)
#     # end def

#     _test_fairseq()
#     _test_transformers()


# def test_MmdErrorFlaggerTrajectoryVer3_vector_cocats_fixed_token_size(resource_path_root: Path):
#     options_test_targets = dict(
#             mode_vector_preprocess='concat',
#             mode_target_embedding_layer='word_embedding',
#             mode_max_token_length_vector_concat=5,
#             kernel_type='gaussian',
#             kernel_length_scale_median_option='dimensionwise'        
#     )

#     def _test_fairseq():
#         path_tmp_dir = Path(tempfile.mkdtemp())
#         path_tmp_dir.mkdir(parents=True, exist_ok=True)
        
#         model_encoder_decoder_mt, seq_correct_translation = pre_routine_fairseq(resource_path_root)

#         seq_calibration_record = seq_correct_translation[:10]
#         # seq_eval_record = seq_correct_translation[100:105]

#         seq_calibration_text = [__r.translation for __r in seq_calibration_record ]

#         translation_handler = FaiseqTranslationModelHandler(model_encoder_decoder_mt=model_encoder_decoder_mt)
#         vector_extractor = FairSeqVectorExtractor(model=model_encoder_decoder_mt)

#         mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
#             translation_handler=translation_handler,
#             vector_extractor=vector_extractor,
#             path_cache_dir=path_tmp_dir,
#             seq_calibration_text=seq_calibration_text,
#             **options_test_targets
#         )
#         # check the gaussian kernel's length scale
#         assert isinstance(mmd_flagger.mmd_estimator.kernel_obj.bandwidth, torch.Tensor)
#         assert len(mmd_flagger.mmd_estimator.kernel_obj.bandwidth) == options_test_targets['mode_max_token_length_vector_concat'] * 512  # the model's word embedding vector is 512.

#         input_obj = EvaluationTargetTranslationPair(
#             source="Freundlichkeit des Frühstückpersonals und Qualität des Frühstück erstklassig, alles ausreichend da.",
#             target="The staff were very friendly and helpful. The room was clean and comfortable.",
#             sentence_id=str(3044)
#         )

#         # test the flagger works
#         res = mmd_flagger.flag_hallucination_one_record(input_obj,
#                                                   candidate_temperature_parameters=np.array([0.1, 0.2, 0.3]),
#                                                   n_sampling=10)
#     # end def

#     def _test_transformers():
#         code_name_nllb = "facebook/nllb-200-distilled-600M"
#         seq_correct_translation, src_lang, target_lang = pre_routine_transformer(resource_path_root)

#         path_tmp_dir = Path(tempfile.mkdtemp())
#         path_tmp_dir.mkdir(parents=True, exist_ok=True)

#         assert len(seq_correct_translation) > 0, f"len(seq_correct_translation)={len(seq_correct_translation)}"
#         seq_calibration_record = seq_correct_translation[:110]

#         seq_calibration_text = [__r.tgt_text for __r in seq_calibration_record ]
#         translation_handler = TransformersTranslationModelHandler(src_lang=src_lang, target_lang=target_lang, model_name=code_name_nllb)
#         vector_extractor = TransformerVectorExtractor(translation_handler)


#         mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
#             translation_handler=translation_handler,
#             vector_extractor=vector_extractor,
#             path_cache_dir=path_tmp_dir,
#             seq_calibration_text=seq_calibration_text,
#             **options_test_targets
#         )
        
#         input_obj = EvaluationTargetTranslationPair(
#             source="Freundlichkeit des Frühstückpersonals und Qualität des Frühstück erstklassig, alles ausreichend da.",
#             target="The staff were very friendly and helpful. The room was clean and comfortable.",
#             sentence_id=str(3044)
#         )
#         candidate_temperature_parameters = np.array([0.1, 0.2])
#         result_flag = mmd_flagger.flag_hallucination_one_record(eval_target=input_obj,
#                                                 candidate_temperature_parameters=candidate_temperature_parameters, 
#                                                 n_sampling=25)
#     # end def

#     _test_fairseq()
#     _test_transformers()
