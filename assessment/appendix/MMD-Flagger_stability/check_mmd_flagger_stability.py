from pathlib import Path
import typing as ty
import logzero
import random
from tqdm import tqdm
import tempfile
import itertools
import torch
import json
import warnings

import numpy as np
import numpy.typing as npt

from omegaconf import UnsupportedValueType

from fairseq.hub_utils import GeneratorHubInterface

from mmd_tst_variable_detector import QuadraticMmdEstimator

from sklearn.metrics import confusion_matrix

import toml
import dataclasses
import dacite
import logging

import itertools

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from hallucination_mt.exceptions import ParameterSettingException
from hallucination_mt.guerreiro_2023_wmt.utils_models import utils
from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset
from hallucination_mt.module_translation_handler.ver1.module_fairseq_handler import FaiseqTranslationModelHandler
from hallucination_mt.module_translation_handler.ver2.module_base import (TranslationResultContainer, EvaluationTargetTranslationPair)

from hallucination_mt.module_hidden_vector_extractor.ver2.module_base import (
    BaseVectorExtractorVer2,
    TranslationResultContainer,
)
from hallucination_mt.module_hidden_vector_extractor.ver2.module_fairseq import FairSeqVectorExtractorVer2CustomTranslationHandlerVer1
from hallucination_mt.module_flagging.module_mmd_error_flagger_trajectory_ver3 import (
    TensorPreprocessorVer1,
    MmdEstimatorInitialiserVer1
)
from hallucination_mt.module_flagging.mmd_error_flagger_trajectory_ver3 import MmdErrorFlaggerTrajectoryVer3
from hallucination_mt.module_flagging.module_classify_trajectory.module_classify_rule_base import classify_function_shape
from hallucination_mt.module_flagging.mmd_error_flagger_ver1 import MmdErrorFlaggerVer1
from hallucination_mt.logger_module import formatter

# warnings.filterwarnings("error")

# logzero.setup_logger(level=logzero.logging.DEBUG)
# logger = logzero.logger


"""A script to analyze the effect of sample size on MMD Trajectory Flagger.
"""


default_seq_n_samples: ty.List[int] = [10, 25, 50, 100]
default_temperature_values: ty.List[float] = [0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86, 0.91, 0.96, 1.0]  # must be Python's pure float. Otherwise, omegaconf raises an exception.

@dataclasses.dataclass
class AlgorithmParameterConfig:
    vector_preprocess: str # avg or concat
    option_max_token_length: int = -1

    max_token_length_vector_concat: ty.Optional[str] = "max_calibration"
    kernel_type: str = "gaussian"
    kernel_length_scale_percentile: int = 25
    kernel_length_scale_median_option: str = "single"

    option_translation_max_a: float = 0.0
    option_translation_max_b: int = 10



@dataclasses.dataclass
class ConfigRoot:
    path_dataset_tsv: Path
    path_fairseq_model_checkpoint: Path
    path_fairseq_model_dir: Path
    path_sentencepiece_model: Path
    
    path_output_dir: Path
    path_translation_cache_root: Path

    algorithm_parameter: AlgorithmParameterConfig

    dir_name_log: str = "logs"

    # execution parameters
    seq_n_samples: ty.Sequence[int] = tuple(default_seq_n_samples)
    temperature_values: ty.Sequence[float] = tuple(default_temperature_values)

    


# Internal Data Objects

class _FlagResult(ty.NamedTuple):
    n_sample: int
    i_iteration: int
    sentence_id: str
    flag: int
    ground_truth: int


class _EvaluationResultContainer(ty.NamedTuple):
    n_total: int
    n_target_label: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    precision: float
    recall: float
    f1: float


class _AggregationFlagSampleSizeAndIteration(ty.NamedTuple):
    n_sample: int
    i_iteration: int
    array_flag: npt.NDArray[np.int8]
    array_ground_truth: npt.NDArray[np.int8]
    trajectory_rule: str
    trajectory_rule_smoothing: str
    trajectory_rule_smoothing_window: int
    evaluation_result: _EvaluationResultContainer



def _get_mmd_estimator(
        model_fairseq_interface: GeneratorHubInterface,
        seq_calibration_text: ty.List[str],
        ) -> QuadraticMmdEstimator:
    mmd_error_flagger = MmdErrorFlaggerVer1(
        model_fairseq_interface,
        n_sampling=1,  # dummy value
        temperature_low=0.1, # dummy value
        temperature_high=1.0, # dummy value
        seq_calibration_text=seq_calibration_text)
    return mmd_error_flagger.mmd_estimator


def _func_get_evaluation(
        array_label_prediction: npt.NDArray[np.int8],
        array_label_ground_truth: npt.NDArray[np.int8]
        ) -> _EvaluationResultContainer:
    """An abstract function to get the evaluation metric."""
    # confusion matrix
    # computing elements of a confusion matrix.
    matrix_confusion = confusion_matrix(array_label_ground_truth, array_label_prediction)
    tn, fp, fn, tp = matrix_confusion.ravel()

    # computing p, r, f1
    # evaluation metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)

    if np.isnan(precision):
        precision = 0.0
    else:
        precision = float(precision)
    if np.isnan(recall):
        recall = 0.0
    else:
        recall = float(recall)
    if np.isnan(f_score):
        f_score = 0.0
    else:
        f_score = float(f_score)
    # end if

    return _EvaluationResultContainer(
        n_total=len(array_label_ground_truth),
        n_target_label=int(np.sum(array_label_ground_truth)),
        true_positive=int(tp),
        false_positive=int(fp),
        true_negative=int(tn),
        false_negative=int(fn),
        precision=precision,
        recall=recall,
        f1=f_score)


def main_translation(
        path_dataset_tsv: Path,
        path_fairseq_model_checkpoint: Path,
        path_fairseq_model_dir: Path,
        path_sentencepiece_model: Path,
        path_output_dir: Path,
        temperature_values: ty.List[float],
        seq_n_samples: ty.List[int],
        target_sentence_ids: ty.Union[str, ty.List[int]],
        path_cache_translation_root: Path,        
        random_seed: int = 42,
        n_iteration: int = 5):
    # path_output_dir.mkdir(exist_ok=True, parents=True)
    # path_cache_dir = path_output_dir / "cache"


    def obtain_word_embedding(model_encoder_decoder_mt,
                              seq_container_translation: ty.List[TranslationResultContainer]) -> ty.List[TranslationResultContainer]:
        # seq_translation_text = [container.translation_text for container in seq_container_translation]
        # _h_enb_unfixed = extract_word_embeddings_batch(
        #                 model_encoder_decoder_mt, 
        #                 seq_translation_text)        
        seq_container_updated = []
        for _container in seq_container_translation:
            with torch.no_grad():
                token_tensor = torch.tensor(_container.target_tensor_tokens)  # Batch size of 1
                model_encoder_decoder_mt = model_encoder_decoder_mt.to(torch.device('cpu'))
                embeddings = model_encoder_decoder_mt.models[0].decoder.embed_tokens(token_tensor)
            # end with
            dict_container = _container._asdict()
            dict_container['dict_layer_embeddings'] = {'decoder.word_embedding': embeddings}
            _container_updated = TranslationResultContainer(**dict_container)
            seq_container_updated.append(_container_updated)
        # end for
        return seq_container_updated
    # end def

    path_result_dir = path_output_dir / "result"
    path_result_dir.mkdir(exist_ok=True, parents=True)
    
    assert path_dataset_tsv.exists(), f"{path_dataset_tsv} does not exist"
    assert path_fairseq_model_checkpoint.exists(), f"{path_fairseq_model_checkpoint} does not exist"
    assert path_fairseq_model_dir.exists(), f"{path_fairseq_model_dir} does not exist"
    assert path_sentencepiece_model.exists(), f"{path_sentencepiece_model} does not exist"
    assert path_output_dir.exists(), f"{path_output_dir} does not exist"

    # loading the dataset
    root_logger.debug(f"Load dataset from {path_dataset_tsv}")
    dataset_records = load_dataset(path_dataset_tsv, delimiter="\t")
    root_logger.debug(f"Loaded {len(dataset_records)} records")
    if isinstance(target_sentence_ids, list):
        seq_target_records = [
            record 
            for record in dataset_records 
            if record.sentence_id in target_sentence_ids]
        assert len(seq_target_records) > 0, f"target_sentence_ids={target_sentence_ids} does not match any records"
    elif isinstance(target_sentence_ids, str):
        if target_sentence_ids == "hallucination":
            seq_target_records = [
                record for record in dataset_records if record.error_type == "hallucination"]
        else:
            raise ValueError(f"target_sentence_ids={target_sentence_ids} is not supported")
    else:
        raise UnsupportedValueType(f"target_sentence_ids={target_sentence_ids}")
    # end if
    assert len(seq_target_records) > 0, f"No records found for target_sentence_ids={target_sentence_ids}"

    # I create random seed for the fairseq model
    __candidate_numbers = range(0, 1000)
    seq_random_seed_fairseq = random.Random(random_seed).sample(__candidate_numbers, k=n_iteration)

    loop_conditions = list(itertools.product(range(n_iteration), seq_random_seed_fairseq, seq_n_samples))

    for _t_conditions in loop_conditions:
        _i_iteration, _random_seed_iteration, _n_sampling = _t_conditions

        _path_cache_translation = path_cache_translation_root / f'iteration_{_i_iteration}'
        root_logger.info(f"Translation cache directory -> {_path_cache_translation}")
        _path_cache_translation.mkdir(parents=True, exist_ok=True)
        
        # loading the model
        root_logger.debug("Loading the fairseq model")
        model_fairseq_interface = utils.load_model(path_fairseq_model_dir, path_fairseq_model_checkpoint, path_sentencepiece_model)

        _translation_handler = FaiseqTranslationModelHandler(
            model_encoder_decoder_mt=model_fairseq_interface,
            data_format_return='ver2',
            max_len_a=0.0,
            max_len_b=10,
            random_seed=_random_seed_iteration,
            n_sampling=_n_sampling,
            path_cache_dir=_path_cache_translation,
            is_zlib_compress=False)

        for _record in seq_target_records:
            # beam-search; `n_sampling=None` in the argument, it automatically switches to beam search.
            _is_exist_beam_search = _translation_handler._is_exist_cache(sentence_id=str(_record.sentence_id), tau_param=1.0, n_sampling=None)  # TODO: I may make the round.
            if _is_exist_beam_search is None:
                root_logger.info(f"sentence-id={_record.sentence_id}. beam-search translation started")
                input_record = EvaluationTargetTranslationPair(sentence_id=str(_record.sentence_id), source=_record.source, target=_record.translation)
                try:
                    _container_beam_search = _translation_handler.translate_beam_search(input_text=input_record, temperature=1.0)  # type: ignore
                except ParameterSettingException as e:
                    root_logger.error(f"sentence-id: {_record.sentence_id} -> {e}")
                else:
                    # converting into vector and update the container
                    assert isinstance(_container_beam_search, TranslationResultContainer)
                    _container_beam_search_updated = obtain_word_embedding(model_fairseq_interface, [_container_beam_search])[0]
                    # save
                    _translation_handler._save_cache(sentence_id=str(_record.sentence_id), tau_param=1.0, translation_obj=_container_beam_search_updated, n_sampling=None)
                    root_logger.info(f"sentence-id={_record.sentence_id}. beam-search translation done")
            # end if

            for _tau_param in temperature_values:
                # check if the file existence
                # if exist -> skip, else do-translation
                _is_exists_tau = _translation_handler._is_exist_cache(
                    sentence_id=str(_record.sentence_id),
                    tau_param=_tau_param,  # TODO: I may make the round.
                    n_sampling=_n_sampling)
                if _is_exists_tau is None:
                    root_logger.info(f"sentence-id={_record.sentence_id}. stochastic translation tau={_tau_param} started.")
                    try:
                        _seq_container_tau = _translation_handler.sample_multiple_times(input_text=_record.source, temperature=_tau_param, n_sampling=_n_sampling)
                    except ParameterSettingException as e:
                        root_logger.error(f"sentence-id={_record.sentence_id} -> {e}")
                    else:
                        assert isinstance(_seq_container_tau, list)
                        _seq_container_tau_updated = obtain_word_embedding(model_fairseq_interface, _seq_container_tau)
                        # save to the disk
                        _translation_handler._save_cache(sentence_id=str(_record.sentence_id), tau_param=_tau_param, translation_obj=_seq_container_tau_updated, n_sampling=_n_sampling)
                        root_logger.info(f"sentence-id={_record.sentence_id}. stochastic translation tau={_tau_param} done.")
                # end if
            # end for
        # end for
    # end for



def main_flag(
        path_dataset_tsv: Path,
        path_fairseq_model_checkpoint: Path,
        path_fairseq_model_dir: Path,
        path_sentencepiece_model: Path,
        path_output_dir: Path,
        path_translation_cache_root: Path,
        temperature_values: ty.List[float],
        seq_n_samples: ty.List[int],
        target_sentence_ids: ty.Union[str, ty.List[int]],
        algorithm_parameter_config: AlgorithmParameterConfig,        
        random_seed: int = 42,
        n_iteration: int = 5,
        N_CALIBRATION_TEXT: int = 200,
        candidate_trajectory_rule: ty.List[str] = ["v1"],
        candidate_trajectory_rule_smoothing: ty.List[str] = ["no_filter"],
        candidate_trajectory_rule_smoothing_window: ty.List[int] = [2],):
    # path_output_dir.mkdir(exist_ok=True, parents=True)
    # path_cache_dir = path_output_dir / "cache"

    path_cache_flagger = Path(tempfile.mkdtemp())

    path_result_dir = path_output_dir / "result"
    path_result_dir.mkdir(exist_ok=True, parents=True)
    
    path_mmd_estimator_cache_root = path_output_dir / "cache_mmd_estimator"

    assert path_dataset_tsv.exists(), f"{path_dataset_tsv} does not exist"
    assert path_fairseq_model_checkpoint.exists(), f"{path_fairseq_model_checkpoint} does not exist"
    assert path_fairseq_model_dir.exists(), f"{path_fairseq_model_dir} does not exist"
    assert path_sentencepiece_model.exists(), f"{path_sentencepiece_model} does not exist"
    assert path_output_dir.exists(), f"{path_output_dir} does not exist"

    # loading the dataset
    root_logger.debug(f"Load dataset from {path_dataset_tsv}")
    dataset_records = load_dataset(path_dataset_tsv, delimiter="\t")
    root_logger.debug(f"Loaded {len(dataset_records)} records")
    if isinstance(target_sentence_ids, list):
        seq_target_records = [
            record 
            for record in dataset_records 
            if record.sentence_id in target_sentence_ids]
        assert len(seq_target_records) > 0, f"target_sentence_ids={target_sentence_ids} does not match any records"
    elif isinstance(target_sentence_ids, str):
        if target_sentence_ids == "hallucination":
            seq_target_records = [
                record for record in dataset_records if record.error_type == "hallucination"]
        else:
            raise ValueError(f"target_sentence_ids={target_sentence_ids} is not supported")
    else:
        raise UnsupportedValueType(f"target_sentence_ids={target_sentence_ids}")
    # end if
    assert len(seq_target_records) > 0, f"No records found for target_sentence_ids={target_sentence_ids}"

    # loading the model
    root_logger.debug("Loading the fairseq model")
    model_fairseq_interface = utils.load_model(path_fairseq_model_dir, path_fairseq_model_checkpoint, path_sentencepiece_model)

    # -----------------------------------------------------------------
    # randomly select the text for the calibration
    root_logger.debug(f"I am selecting the calibration text. N={N_CALIBRATION_TEXT}")
    seq_calibration = random.Random(random_seed).sample(dataset_records, k=N_CALIBRATION_TEXT)
    if isinstance(target_sentence_ids, list):
        seq_calibration = [__r for __r in seq_calibration if int(__r.sentence_id) not in target_sentence_ids]
    elif isinstance(target_sentence_ids, str):
        seq_calibration = [__r for __r in seq_calibration if __r.error_type == "correct"]
    else:
        raise UnsupportedValueType(f"target_sentence_ids={target_sentence_ids}")
    # end if
    assert len(seq_calibration) > 0, f"No records found for calibration"
    seq_calibration_records = [EvaluationTargetTranslationPair(source=_r.source, target=_r.translation, sentence_id=str(_r.sentence_id)) for _r in seq_calibration]
    # -----------------------------------------------------------------

    dict_sent_id2ground_truth = {str(_record.sentence_id): 1 if _record.error_type == 'hallucination' else 0 for _record in seq_target_records}

    seq_flag_sample_size_and_iteration: ty.List[_AggregationFlagSampleSizeAndIteration] = []
    # for-loop of sample-size.
    tqdm_iter = tqdm(total=len(seq_n_samples))
    for _n_sample_size in seq_n_samples:
        # I create random seed for the fairseq model
        __candidate_numbers = range(0, 1000)
        seq_random_seed_fairseq = random.Random(random_seed).sample(__candidate_numbers, k=n_iteration)

        root_logger.info(f"Executing Flagger for sample size {_n_sample_size}")

        for _iter_number, _random_seed_fairseq in zip(range(n_iteration), seq_random_seed_fairseq):
            # a directory to save MMD flagger result.
            _path_dir_save_output = path_result_dir / str(_n_sample_size) / f"iteration_{_iter_number}"
            _path_dir_save_output.mkdir(parents=True, exist_ok=True)
            # a directory to hold the translation result.
            _path_translation_cache_dir = path_translation_cache_root / f"iteration_{_iter_number}"
            _path_translation_cache_dir.mkdir(exist_ok=True, parents=True)

            # ---------------------------------------------------------
            # initializing modules
            translation_handler = FaiseqTranslationModelHandler(
                model_encoder_decoder_mt=model_fairseq_interface,
                data_format_return='ver2',
                max_len_a=0.0,
                max_len_b=10,
                random_seed=_random_seed_fairseq,
                n_sampling=_n_sample_size,
                path_cache_dir=_path_translation_cache_dir,
                is_zlib_compress=False)
            vector_extractor = FairSeqVectorExtractorVer2CustomTranslationHandlerVer1(translation_handler)

            _tensor_preprocessor = TensorPreprocessorVer1(
                mode_vector_preprocess=algorithm_parameter_config.vector_preprocess,
                mode_max_token_length_vector_concat=algorithm_parameter_config.max_token_length_vector_concat,
                option_max_token_length=algorithm_parameter_config.option_max_token_length
            )

            init_mmd_estimator = MmdEstimatorInitialiserVer1(
                tensor_preprocessor=_tensor_preprocessor,
                vector_extractor=vector_extractor,
                mode_target_embedding_layer="decoder.word_embedding",
                kernel_type=algorithm_parameter_config.kernel_type,
                kernel_length_scale_percentile=algorithm_parameter_config.kernel_length_scale_percentile, 
                kernel_length_scale_median_option=algorithm_parameter_config.kernel_length_scale_median_option,
                path_cache_dir=path_mmd_estimator_cache_root,  # a directory to save mmd-estimator. length-scale set may take several time. 
                option_translation_max_a=algorithm_parameter_config.option_translation_max_a,
                option_translation_max_b=algorithm_parameter_config.option_translation_max_b
            )
            root_logger.info(f"computing the kernel length scale")
            _mmd_estimator = init_mmd_estimator.get_mmd_estimator(
                seq_calibration_text=seq_calibration_records,
            )
            root_logger.info(f"done computing the kernel length scale")            
            # ---------------------------------------------------------


            # setting the MMD Trajectory Flagger
            _mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
                vector_extractor=vector_extractor,
                mmd_estimator=_mmd_estimator,
                tensor_preprocessor=_tensor_preprocessor,
                mode_target_embedding_layer="decoder.word_embedding",
                option_translation_max_a=0.0,
                option_translation_max_b=10,
                path_cache_dir=path_cache_flagger,  # note: not used right now.
                target_layers_extraction=["decoder.word_embedding"]
            )
            
            for _record in tqdm(seq_target_records, desc=f"Flagging records with sample-size={_n_sample_size}, iteration={_iter_number}"):
                _path_save_pt = _path_dir_save_output / f'{_record.sentence_id}.pt'
                
                if _path_save_pt.exists():
                    continue
                # end if

                root_logger.info(f"flag started sentence-id={_record.sentence_id}")
                try:
                    _result_trajectory_container = _mmd_flagger.flag_hallucination_one_record(
                        eval_target=EvaluationTargetTranslationPair(source=_record.source, target=_record.translation, sentence_id=str(_record.sentence_id)),
                        candidate_temperature_parameters=[float(_tau) for _tau in temperature_values],
                        n_sampling=_n_sample_size)
                except ParameterSettingException as e:
                    root_logger.error(f"flag error sentence-id={_record.sentence_id} by e -> {e}")
                except RuntimeError as e:
                    root_logger.error(f"flag error sentence-id={_record.sentence_id} by e -> {e}")                    
                else:
                    root_logger.info(f"done started sentence-id={_record.sentence_id}")
                    torch.save(dataclasses.asdict(_result_trajectory_container), _path_save_pt)
            # end for

        tqdm_iter.update(1)        
    # end for


    # Visualizing the result
    # X-axis: sample size, Y-axis: Recall-score.
    # Aggregation of result by a key of sample-size.

    # # I create a DataFrame.
    # seq_pd_record = []
    # for _record in seq_flag_sample_size_and_iteration:
    #     _pd_record = {
    #         "n_sample": _record.n_sample,
    #         "i_iteration": _record.i_iteration,
    #         "precision": _record.evaluation_result.precision,
    #         "recall": _record.evaluation_result.recall,
    #         "f1": _record.evaluation_result.f1,
    #         "trajectory_rule": _record.trajectory_rule,
    #         "trajectory_rule_smoothing": _record.trajectory_rule_smoothing,
    #         "trajectory_rule_smoothing_window": _record.trajectory_rule_smoothing_window,
    #         "trajectory_rule_combination_name": f"{_record.trajectory_rule}/{_record.trajectory_rule_smoothing}/{_record.trajectory_rule_smoothing_window}"
    #     }
    #     seq_pd_record.append(_pd_record)
    # # end for
    # df_eval_result = pd.DataFrame(seq_pd_record)

    # # exporting the result into a TSV file.
    # df_eval_result.to_csv(path_result_dir / "recall_score_by_sample_size.tsv", sep="\t", index=False)

    # f, ax = plt.subplots(1, 1, figsize=(15, 5))
    # sns.lineplot(data=df_eval_result, x="n_sample", y="recall", hue="trajectory_rule_combination_name", ax=ax)
    # ax.set_title("Recall score by sample size")
    # ax.set_xlabel("Sample size")
    # ax.set_ylabel("Recall score")
    # ax.set_ylim(0, 1)
    
    # f.savefig(path_result_dir / "recall_score_by_sample_size.png", dpi=300, bbox_inches="tight")



# ----------------------------------------------------------------------------------------------------






def create_interface_config(config_obj: ty.Dict) -> ConfigRoot:
    config_obj["path_dataset_tsv"] = Path(config_obj["path_dataset_tsv"])
    config_obj["path_fairseq_model_checkpoint"] = Path(config_obj["path_fairseq_model_checkpoint"])
    config_obj["path_fairseq_model_dir"] = Path(config_obj["path_fairseq_model_dir"])
    config_obj["path_sentencepiece_model"] = Path(config_obj["path_sentencepiece_model"])
    config_obj["path_output_dir"] = Path(config_obj["path_output_dir"])
    config_obj["path_translation_cache_root"] = Path(config_obj["path_translation_cache_root"])    

    return dacite.from_dict(ConfigRoot, config_obj)



def create_file_logger(path_config: Path, mode_name: str) -> logging.Logger:
    config_obj = toml.load(path_config)
    interface_config = create_interface_config(config_obj)

    (interface_config.path_output_dir / interface_config.dir_name_log).mkdir(parents=True, exist_ok=True)
    path_log_dir = interface_config.path_output_dir / interface_config.dir_name_log / f"{mode_name}.log"

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set up logging
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(path_log_dir)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    std_handler = logging.StreamHandler()
    std_handler.setLevel(logging.DEBUG)
    std_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(std_handler)

    # re-setting the log level.
    logging.getLogger('fairseq').setLevel(logging.WARNING)
    
    logger.info(f"-------- Script Start --------")

    return logger



def main(path_config: Path, mode_name: str, n_iteration: int = 10):
    # --------------------------------------------
    # for hallucination
    config_obj_dict = toml.load(path_config)
    config_obj = create_interface_config(config_obj_dict)
    config_obj.path_output_dir.mkdir(exist_ok=True, parents=True)

    target_sentence_ids_hallucination = "hallucination"

    

    if mode_name == "translation":
        main_translation(
            path_dataset_tsv=config_obj.path_dataset_tsv,
            path_fairseq_model_checkpoint=config_obj.path_fairseq_model_checkpoint,
            path_fairseq_model_dir=config_obj.path_fairseq_model_dir,
            path_sentencepiece_model=config_obj.path_sentencepiece_model,
            path_output_dir=config_obj.path_output_dir,
            temperature_values=list(config_obj.temperature_values),
            seq_n_samples=list(config_obj.seq_n_samples),
            target_sentence_ids=target_sentence_ids_hallucination,
            path_cache_translation_root=config_obj.path_translation_cache_root,
            n_iteration=n_iteration
        )
    elif mode_name == "flag":
        main_flag(
            path_dataset_tsv=config_obj.path_dataset_tsv,
            path_fairseq_model_checkpoint=config_obj.path_fairseq_model_checkpoint,
            path_fairseq_model_dir=config_obj.path_fairseq_model_dir,
            path_sentencepiece_model=config_obj.path_sentencepiece_model,
            path_output_dir=config_obj.path_output_dir,
            temperature_values=list(config_obj.temperature_values),
            seq_n_samples=list(config_obj.seq_n_samples),
            target_sentence_ids=target_sentence_ids_hallucination,
            algorithm_parameter_config=config_obj.algorithm_parameter,
            path_translation_cache_root=config_obj.path_translation_cache_root,
            random_seed=42,
            n_iteration=n_iteration,
            candidate_trajectory_rule=["v1"],
            candidate_trajectory_rule_smoothing=["no_filter"],
            candidate_trajectory_rule_smoothing_window=[-1],)
    else:
        raise NotImplementedError()




if __name__ == "__main__":
    import argparse
    _opt = argparse.ArgumentParser()
    _opt.add_argument('-m', '--mode', choices=['visualisation', 'flag', 'translation'])
    _opt.add_argument('-p', '--path_config', required=True)
    _args = _opt.parse_args()


    root_logger = create_file_logger(Path(_args.path_config), mode_name=_args.mode)
    main(Path(_args.path_config), mode_name=_args.mode)
    # _configurations_2025_05(_args.mode)