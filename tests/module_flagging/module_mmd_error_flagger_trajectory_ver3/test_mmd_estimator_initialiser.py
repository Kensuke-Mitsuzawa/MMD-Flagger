import typing as ty
import shutil
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from pathlib import Path
import tempfile

from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset
from hallucination_mt.module_flagging.module_mmd_error_flagger_trajectory_ver3 import mmd_estimator_initialiser
from hallucination_mt.module_flagging.module_mmd_error_flagger_trajectory_ver3.tensor_preprocessor import TensorPreprocessorVer1
from hallucination_mt.module_translation_handler.ver2.module_base import EvaluationTargetTranslationPair
from hallucination_mt.module_translation_handler.ver2.module_fairseq_handler import FairSeqTranslationModelHandlerVer2
from hallucination_mt.module_hidden_vector_extractor.ver2 import FairSeqVectorExtractorVer2

from hallucination_mt.module_flagging.module_mmd_error_flagger_trajectory_ver3 import (
    MmdEstimatorInitialiserVer1,
)
from hallucination_mt.module_flagging.module_mmd_error_flagger_trajectory_ver3.mmd_estimator_initialiser import get_median_heuristic_single
from hallucination_mt.module_flagging.module_mmd_error_flagger_trajectory_ver3.dot_product_kernel import DotProductKernel

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator


def pre_routine_fairseq(resource_path_root: Path):
    path_dir_mode = resource_path_root / 'model_guerreiro_2023'
    assert path_dir_mode.exists()

    path_dataset_tsv = resource_path_root / 'eval_datasets/lfan_hall_subset.tsv'
    seq_dataset = load_dataset(path_dataset_tsv, delimiter="\t")
    seq_correct_translation = [__record for __record in seq_dataset if __record.error_type == "correct"]

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

    return seq_calibration_text, vector_extractor


# def base_procedure_fairseq_based(resource_path_root: Path,
#                                  mode_vector_preprocess: str,
#                                  mode_target_embedding_layer: str,
#                                  mode_max_token_length_vector_concat: str = 'max_calibration'):
#     path_tmp_dir = Path(tempfile.mkdtemp())
#     path_tmp_dir.mkdir(parents=True, exist_ok=True)
    
#     path_dir_mode = resource_path_root / 'model_guerreiro_2023'

#     model_encoder_decoder_mt, seq_correct_translation = pre_routine_fairseq(resource_path_root)

#     # seq_calibration_record = seq_correct_translation[:200]
#     seq_calibration_record = seq_correct_translation[100:105]

#     seq_calibration_text = [
#         EvaluationTargetTranslationPair(
#             source=_r.source,
#             target=_r.translation,
#             sentence_id=str(_r.sentence_id)
#         )
#         for _r in seq_calibration_record
#     ]

#     translation_handler = FairSeqTranslationModelHandlerVer2(
#         path_dir_fairseq_model=path_dir_mode / 'wmt18_de-en',
#         path_model_checkpoint=path_dir_mode / 'checkpoint_best.pt',
#         path_sentencepiece_model=path_dir_mode / 'sentencepiece_models/sentencepiece.joint.bpe.model'
#         )
#     vector_extractor = FairSeqVectorExtractorVer2(translation_handler)

#     tensor_preprocessor = TensorPreprocessorVer1(
#         mode_vector_preprocess=mode_vector_preprocess,
#         mode_max_token_length_vector_concat=mode_max_token_length_vector_concat
#     )
#     mmd_init_executor = MmdEstimatorInitialiserVer1(
#         tensor_preprocessor=tensor_preprocessor,
#         mode_target_embedding_layer='decoder.word_embedding',
#         vector_extractor=vector_extractor
#     )
#     mmd_estimator = mmd_init_executor.init_gaussian_kernel_mmd_estimator(
#         seq_calibration_text=seq_calibration_text,
#     )
    
#     mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
#         vector_extractor=vector_extractor,
#         path_cache_dir=path_tmp_dir,
#         mmd_estimator=mmd_estimator,
#         tensor_preprocessor=tensor_preprocessor,
#         mode_target_embedding_layer=mode_target_embedding_layer,
#         option_is_sampling_in_iteration=True,
#         option_translation_max_a=1.0,
#         option_translation_max_b=10
#     )
    
#     input_obj = EvaluationTargetTranslationPair(
#         source="Freundlichkeit des Frühstückpersonals und Qualität des Frühstück erstklassig, alles ausreichend da.",
#         target="The staff were very friendly and helpful. The room was clean and comfortable.",
#         sentence_id=str(3044)
#     )
#     candidate_temperature_parameters = np.array([0.1, 0.3, 0.5, 0.6])
#     result_flag = mmd_flagger.flag_hallucination_one_record(eval_target=input_obj,
#                                               candidate_temperature_parameters=candidate_temperature_parameters, 
#                                               n_sampling=5)
#     # check the cache db
        
#     result_flag_second = mmd_flagger.flag_hallucination_one_record(
#         eval_target=input_obj,
#         candidate_temperature_parameters=candidate_temperature_parameters, 
#         n_sampling=25)
    
#     assert result_flag_second.trajectory_shape == "saddle-point", f"result_flag.trajectory_shape={result_flag.trajectory_shape}"

#     shutil.rmtree(path_tmp_dir)   


def test_MmdEstimatorInitialiserVer1_gaussian_dimensionwise(resource_path_root: Path):
    seq_calibration_record, vector_extractor  = pre_routine_fairseq(resource_path_root)

    mode_vector_preprocess = 'avg'
    kernel_length_scale_median_option = 'dimensionwise'

    tensor_preprocessor = TensorPreprocessorVer1(
        mode_vector_preprocess=mode_vector_preprocess,
        mode_max_token_length_vector_concat=None
    )
    mmd_init_executor = MmdEstimatorInitialiserVer1(
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer='decoder.word_embedding',
        vector_extractor=vector_extractor,
        kernel_length_scale_median_option=kernel_length_scale_median_option
    )
    mmd_estimator = mmd_init_executor.init_gaussian_kernel_mmd_estimator(
        seq_calibration_text=seq_calibration_record,
    )
    path_model = mmd_init_executor.save_mmd_estimator()
    assert path_model.exists()

    # check: the Kernel's length scale must be multi-dimension
    assert len(mmd_estimator.kernel_obj.bandwidth) > 1

    path_model.unlink()
    shutil.rmtree(mmd_init_executor.path_cache_dir)


def test_MmdEstimatorInitialiserVer1_gaussian_dimensionwise_percentile(resource_path_root: Path):
    seq_calibration_record, vector_extractor  = pre_routine_fairseq(resource_path_root)

    mode_vector_preprocess = 'avg'
    kernel_length_scale_median_option = 'dimensionwise'
    percentile_value = 5

    tensor_preprocessor = TensorPreprocessorVer1(
        mode_vector_preprocess=mode_vector_preprocess,
        mode_max_token_length_vector_concat=None
    )
    mmd_init_executor = MmdEstimatorInitialiserVer1(
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer='decoder.word_embedding',
        vector_extractor=vector_extractor,
        kernel_length_scale_median_option=kernel_length_scale_median_option,
        kernel_length_scale_percentile=percentile_value
    )
    mmd_estimator = mmd_init_executor.init_gaussian_kernel_mmd_estimator(
        seq_calibration_text=seq_calibration_record,
    )
    path_model = mmd_init_executor.save_mmd_estimator()
    assert path_model.exists()

    # check: the Kernel's length scale must be multi-dimension
    assert len(mmd_estimator.kernel_obj.bandwidth) > 1

    # check: the length scale value are smaller than percentile=50 overall.
    mmd_init_executor_compare = MmdEstimatorInitialiserVer1(
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer='decoder.word_embedding',
        vector_extractor=vector_extractor,
        kernel_length_scale_median_option=kernel_length_scale_median_option,
        kernel_length_scale_percentile=50
    )
    mmd_estimator_compare = mmd_init_executor_compare.init_gaussian_kernel_mmd_estimator(
        seq_calibration_text=seq_calibration_record,
    )
    binary_tensor = mmd_estimator.kernel_obj.bandwidth < mmd_estimator_compare.kernel_obj.bandwidth
    ratio_true = binary_tensor.float().mean()
    assert ratio_true > 0.7, "Test failed. Most of bandwidth values are supposed to be smaller compared with percentile=50."

    path_model.unlink()
    shutil.rmtree(mmd_init_executor.path_cache_dir)
    

def test_MmdEstimatorInitialiserVer1_gaussian_single(resource_path_root: Path):
    seq_calibration_record, vector_extractor  = pre_routine_fairseq(resource_path_root)

    mode_vector_preprocess = 'avg'
    kernel_length_scale_median_option = 'single'

    tensor_preprocessor = TensorPreprocessorVer1(
        mode_vector_preprocess=mode_vector_preprocess,
        mode_max_token_length_vector_concat=None
    )
    mmd_init_executor = MmdEstimatorInitialiserVer1(
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer='decoder.word_embedding',
        vector_extractor=vector_extractor,
        kernel_length_scale_median_option=kernel_length_scale_median_option
    )
    mmd_estimator = mmd_init_executor.init_gaussian_kernel_mmd_estimator(
        seq_calibration_text=seq_calibration_record,
    )
    path_model = mmd_init_executor.save_mmd_estimator()
    assert path_model.exists()

    # check: the Kernel's length scale must be single value
    assert mmd_estimator.kernel_obj.bandwidth is not None
    assert isinstance(mmd_estimator.kernel_obj.bandwidth.item(), float)

    path_model.unlink()
    shutil.rmtree(mmd_init_executor.path_cache_dir)


def test_MmdEstimatorInitialiserVer1_gaussian_single_percentile(resource_path_root: Path):
    seq_calibration_record, vector_extractor  = pre_routine_fairseq(resource_path_root)

    mode_vector_preprocess = 'avg'
    kernel_length_scale_median_option = 'single'

    tensor_preprocessor = TensorPreprocessorVer1(
        mode_vector_preprocess=mode_vector_preprocess,
        mode_max_token_length_vector_concat=None
    )
    mmd_init_executor = MmdEstimatorInitialiserVer1(
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer='decoder.word_embedding',
        vector_extractor=vector_extractor,
        kernel_length_scale_median_option=kernel_length_scale_median_option,
        kernel_length_scale_percentile=90
    )
    mmd_estimator = mmd_init_executor.init_gaussian_kernel_mmd_estimator(
        seq_calibration_text=seq_calibration_record,
    )
    path_model = mmd_init_executor.save_mmd_estimator()
    assert path_model.exists()

    # check: the Kernel's length scale must be single value
    assert mmd_estimator.kernel_obj.bandwidth.item()

    # check: the length scale value are smaller than percentile=50 overall.
    mmd_init_executor_compare = MmdEstimatorInitialiserVer1(
        tensor_preprocessor=tensor_preprocessor,
        mode_target_embedding_layer='decoder.word_embedding',
        vector_extractor=vector_extractor,
        kernel_length_scale_median_option=kernel_length_scale_median_option,
        kernel_length_scale_percentile=50
    )
    mmd_estimator_compare = mmd_init_executor_compare.init_gaussian_kernel_mmd_estimator(
        seq_calibration_text=seq_calibration_record,
    )
    binary_tensor = mmd_estimator.kernel_obj.bandwidth > mmd_estimator_compare.kernel_obj.bandwidth
    ratio_true = binary_tensor.float().mean()
    assert ratio_true > 0.7, "Test failed. Most of bandwidth values are supposed to be smaller compared with percentile=50."


    path_model.unlink()
    shutil.rmtree(mmd_init_executor.path_cache_dir)


from mmd_tst_variable_detector.kernels.gaussian_kernel import L2Distance


def test_get_median_heuristic_single():
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(100, 20))
    y = rng.normal(0.0, 3.0, size=(100, 20))    

    value_median = get_median_heuristic_single(distance_module=L2Distance(coordinate_size=1), x=torch.from_numpy(x), y=torch.from_numpy(y), percentile=50)

    def euclidean_distances_squared(X, Y=None):
        if Y is None:
            Y = X
        # Compute squared norms of X and Y
        X_norm = (X**2).sum(dim=1).unsqueeze(1)  # shape: (n_samples_X, 1)
        Y_norm = (Y**2).sum(dim=1).unsqueeze(0)  # shape: (1, n_samples_Y)
        
        # Compute the squared Euclidean distance matrix
        distances_squared = X_norm + Y_norm - 2.0 * torch.matmul(X, Y.T)
        
        # Clamp to zero to avoid negative distances due to numerical issues
        distances_squared = torch.clamp(distances_squared, min=0.0)
        return distances_squared

    sample_concat = torch.cat([torch.from_numpy(x), torch.from_numpy(y)])
    d2_torch = euclidean_distances_squared(sample_concat)
    # matrix_shape_torch = d2_torch.shape
    # distance_matrix_torch = d2_torch[torch.triu_indices(matrix_shape_torch[0], matrix_shape_torch[0], 1)]
    med_sqdist_torch = torch.quantile(d2_torch, q=(50 / 100))
    bandwidth_torch = torch.sqrt(med_sqdist_torch / 2)

    # code by numpy and scikit learn
    samp = torch.cat([torch.from_numpy(x), torch.from_numpy(y)])
    np_reps = samp.detach().cpu().numpy()
    d2 = euclidean_distances(np_reps, squared=True)
    # distance_matrix = d2[np.triu_indices_from(d2, k=1)]
    med_sqdist = np.percentile(d2, q=50)
    bandwidth = np.sqrt(med_sqdist / 2)

    assert np.abs(bandwidth_torch.numpy().item() - bandwidth.item()) < 0.1


def test_MmdEstimatorInitialiserVer1_dot_product():
    rng = np.random.default_rng(42)
    x = torch.from_numpy(rng.normal(1.0, 1.0, size=(100, 20)))
    y = torch.from_numpy(rng.normal(10.0, 3.0, size=(100, 20)))

    mmd_estimator = QuadraticMmdEstimator(kernel_obj=DotProductKernel(ard_weight_shape=(1,)))
    mmd_computed_obj = mmd_estimator.forward(x, y)
    assert mmd_computed_obj.mmd.item()