import toml
import dacite
import typing as ty
import sqlite3
import json
import subprocess
import itertools
import logging
import tqdm
import sys
import random
import importlib
import copy

import numpy as np
import torch

from pathlib import Path

import dataclasses
from dataclasses import dataclass, asdict

# dataset module
from hallucination_mt.dale_2023_halomi.load_dataset import (
    HalomiDatasetRecord,
    load_dataset as loading_halomi
)
from hallucination_mt.guerreiro_2023_wmt.data_models.utils import (
    WMTDatasetRecord,
    load_dataset as loading_lfan_hall
)

# management db module
from hallucination_mt.module_assessments.module_management_db.interface_ver3.module_db_record import (
    DbTableRecordRaunak2021,
    DbTableRecordGuerreiro2023SeqLogProb,
    DbTableRecordGuerreiro2023McDSIM,
    DbTableRecordProposalMmdFlaggerTrajectoryVer3
)
from hallucination_mt.module_assessments.module_management_db.module_sqlite3_handler import (
    create_table_from_table_definition,
    DBHandlerExp
)
# helper
from hallucination_mt.module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from hallucination_mt.guerreiro_2023_wmt.utils_models.utils import (
    load_model as loading_model_fairseq
)
from hallucination_mt.module_assessments import slack_notifier
from slack_sdk.webhook import WebhookClient
from hallucination_mt.module_assessments.slack_notifier import message_monitor

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator
# approach modules
from hallucination_mt.commons.data_models import EvaluationTargetTranslationPair
from hallucination_mt.baselines.raunak_2021 import oscillatory_detection_Raunak_2021
from hallucination_mt.baselines.seq_log_probability import (
    TransformerFlaggerSeqLogProbability,
    OutputLogProbabilityFlagger)
from hallucination_mt.baselines.mc_dropout.fairseq_handler.flagger_mc_dropout import FlaggerDisSimilarityMcDropOut, OutputDisSimilarityMcDropOut
from hallucination_mt.baselines.mc_dropout.fairseq_handler import flagger_mc_dropout
from hallucination_mt.module_flagging.mmd_error_flagger_trajectory_ver3 import (
    MmdErrorFlaggerTrajectoryVer3,
    MmdErrorFlagResultVer3
)
from hallucination_mt.module_translation_handler.ver1.module_fairseq_handler import FaiseqTranslationModelHandler
from hallucination_mt.module_hidden_vector_extractor.ver2.module_fairseq import FairSeqVectorExtractorVer2CustomTranslationHandlerVer1
from hallucination_mt.module_flagging.module_mmd_error_flagger_trajectory_ver3 import (
    load_mmd_estimator,
    MmdEstimatorInitialiserVer1,
    TensorPreprocessorVer1,
)


from hallucination_mt.module_flagging.utils import load_fairseq_model
# fairseq handler
from hallucination_mt.exceptions import ParameterSettingException
# translation module
from hallucination_mt.module_translation_handler.ver2 import (
    BaseTranslationModelHandlerVer2,
    TransformersTranslationModelHandlerVer2, 
    FairSeqTranslationModelHandlerVer2)
from hallucination_mt.module_hidden_vector_extractor.ver2 import (
    BaseTranslationModelHandlerVer2,
    TransformerVectorExtractorVer2,
    FairSeqVectorExtractorVer2
)
# evaluation module
from hallucination_mt.module_assessments.module_evaluation import evaluation_script_ver3

# logger
from hallucination_mt.logger_module import formatter

# method configs
from method_configs import (
    ModelConfigHalomi,
    ModelConfigLfanHall,
    TranslationConfig,
    MmdErrorFlaggerTrajectoryVer3Config,
    Raunak2021ApproachConfig,
    Guerreiro2023MCDSim,
    Guerreiro2023SeqLogprob
)


logging.basicConfig(level=logging.DEBUG)
# a special logger for tqdm
tqdm_logger = logging.getLogger('tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())

SCRIPT_VERSION = "0.2"

# --------------------------------------------------------------

@dataclass
class ResourceConfig:
    path_work_dir: Path
    path_dataset_tsv: Path

    dataset_name: str  # halomi or lfan_hall

    file_name_db_sqlite3: str = "management_db.sqlite3"
    dir_name_log: str = "logs"
    dir_name_cache: str = "cache"

    path_dir_cache_translation: ty.Optional[Path] = None

    # configuration about selecting calibration text
    n_calibration_records: int = 200
    method_select_calibration: str = "random"
    seed_random_seed: int = 42

    # limiting the number of evaluation dataset records. This is for debugging and testing.
    limit_dataset_record: ty.Optional[int] = None

    # filtering the dataset records. The format is ["src_lang_code", "tgt_lang_code"]
    filter_src_tgt_lang_code: ty.Optional[ty.List[str]] = None
    # filtering the label type. I can set the hallucination records only for the evaluation.
    # class label of the hallucination.
    filter_target_label: ty.Optional[ty.List[str]] = None

    # hallucination, mt-error, correct
    filter_error_type: ty.Optional[str] = None

    # list of sentence id that you want to skip the execution (e.g. a translation always fails)
    filter_exclude_sentence_id: ty.Optional[ty.List[str]] = None

    def __post_init__(self):
        assert self.path_dataset_tsv.exists(), f"File not found: {self.path_dataset_tsv}"
        if self.path_dir_cache_translation is not None:
            assert self.path_dir_cache_translation.exists(), f"Directory not found: {self.path_dir_cache_translation}"
        # end if
        
        if self.filter_src_tgt_lang_code is not None:
            self.filter_src_tgt_lang_code = [tuple(_seq_list) for _seq_list in self.filter_src_tgt_lang_code]  # type: ignore
        # end if

    def to_dict(self):
        _obj = asdict(self)
        _obj["path_work_dir"] = _obj["path_work_dir"].as_posix()
        _obj["path_dataset_tsv"] = _obj["path_dataset_tsv"].as_posix()
        return _obj



@dataclass
class EvaluationConfig:
    dir_name_output: str = "evaluation_output"
    dir_name_analysis: str = "result_analysis_output"


@dataclass
class LlmModelConfig:
    model_halomi: ModelConfigHalomi
    model_lfan_hall: ModelConfigLfanHall


@dataclass
class NotificationConfig:
    slack_wekhook: ty.Optional[str] = None


@dataclass
class InterfaceConfig:
    script_version: str
    config_name: str  # recommended to set the unique name.
    resource_config: ResourceConfig
    evaluation_config: EvaluationConfig
    flagging_approaches: ty.List[str]
    approach_configs: ty.Dict[str, ty.Any]
    llm_model_config: LlmModelConfig
    translation_config: TranslationConfig
    notification_config: ty.Optional[NotificationConfig] = None

    def __post_init__(self):
        # Convert the approach configs to the appropriate dataclass
        approach_configs = {}

        assert len(self.approach_configs) > 0, "At least one approach config should be provided"

        # collecting the config objects for each approach
        for _key, _value in self.approach_configs.items():
            assert isinstance(_value, dict), f"Expected dict, got {_value}"
            if _key == "Raunak2021ApproachConfig":
                approach_configs[_key] = dacite.from_dict(Raunak2021ApproachConfig, _value)
            elif _key == "MmdErrorFlaggerTrajectoryVer3Config":
                approach_configs[_key] = dacite.from_dict(MmdErrorFlaggerTrajectoryVer3Config, _value)
            elif _key == "Guerreiro2023MCDSim":
                approach_configs[_key] = dacite.from_dict(Guerreiro2023MCDSim, _value)
            elif _key == "Guerreiro2023SeqLogprob":
                approach_configs[_key] = dacite.from_dict(Guerreiro2023SeqLogprob, _value)
            else:
                raise Exception(f"Approach not implemented: {_key}")
            # end if
        # end for
        self.approach_configs = approach_configs

        # check flagging_approaches
        for _approach_name in self.flagging_approaches:
            assert _approach_name in self.approach_configs, f"Approach config not found for {_approach_name}. You have to define it in the toml file."
        # end for

    def to_dict(self):
        resource_config = self.resource_config.to_dict()
        approach_configs = {_k: _v.to_dict() for _k, _v in self.approach_configs.items()}

        obj = asdict(self)
        obj["resource_config"] = resource_config
        obj["approach_configs"] = approach_configs

        return obj

# --------------------------------------------------------------

def path_to_str(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")



def _select_calibration_record_and_convert(
        seq_dataset_record: ty.Sequence[ty.Union[HalomiDatasetRecord, WMTDatasetRecord]],
        n_calibration_records: int,
        method_select_calibration: str = 'random',
        seed_random_seed: int = 42
        ) -> ty.List[EvaluationTargetTranslationPair]:
    """I select the calibration text from the dataset."""
    dataset_type = classify_dataset_type(seq_dataset_record[0])

    if dataset_type == 'halomi':
        seq_correct_translation = [__record for __record in seq_dataset_record if __record.is_hallucination is False and __record.is_omission is False]
    elif dataset_type == 'lfan_hall':
        seq_correct_translation = [__record for __record in seq_dataset_record if __record.error_type == 'correct']
    else:
        raise ValueError()
    # end if        
    assert len(seq_correct_translation) > 0, f"len(seq_correct_translation)={len(seq_correct_translation)}"
    
    _selected_records: ty.List[ty.Union[HalomiDatasetRecord, WMTDatasetRecord]]
    if len(seq_correct_translation) < n_calibration_records:
        _selected_records = seq_correct_translation
    else:
        if method_select_calibration == "random":
            _random_gen = random.Random(seed_random_seed)
            _selected_records = _random_gen.sample(
                seq_correct_translation, 
                k=n_calibration_records,
            )
        else:
            raise Exception(f"Method not implemented: {method_select_calibration}")
        # end if
    # end if

    if dataset_type == 'halomi':
        seq_expected_input = [EvaluationTargetTranslationPair(source=_r.src_text, target=_r.tgt_text, sentence_id=str(_r.key_unique)) for _r in _selected_records]
    elif dataset_type == 'lfan_hall':
        seq_expected_input = [EvaluationTargetTranslationPair(source=_r.source, target=_r.translation, sentence_id=_r.sentence_id) for _r in _selected_records]
    else:
        raise ValueError()
    # end if

    return seq_expected_input
# end def


def select_record_lang_code(
        seq_dataset_record: ty.Sequence[HalomiDatasetRecord],
        src_lang_code: str,
        tgt_lang_code: str) -> ty.Sequence[HalomiDatasetRecord]:
    """I select the records from the dataset based on the source and target language code."""
    seq_selected_records = [__record for __record in seq_dataset_record if __record.src_lang == src_lang_code and __record.tgt_lang == tgt_lang_code]
    assert len(seq_selected_records) > 0, f"len(seq_selected_records)={len(seq_selected_records)}"

    return seq_selected_records


def making_language_pair(resource_config: ResourceConfig,
                         seq_dataset: ty.Sequence[HalomiDatasetRecord]
                         ) -> ty.List[ty.Tuple[str, str]]:
    # filtering records by the option of `filter_src_tgt_lang_code`.
    if resource_config.filter_src_tgt_lang_code is not None:
        __set_conditions_src_and_tgt: ty.Set[ty.Tuple[str, str]] = set([tuple(_l) for _l in resource_config.filter_src_tgt_lang_code])  # type: ignore
        seq_possible_language_pair = [tuple(_l) for _l in resource_config.filter_src_tgt_lang_code]
    else:
        # making the source and target language code.
        __seq_possible_language_pair = set([(__r.src_lang, __r.tgt_lang) for __r in seq_dataset])
        seq_possible_language_pair = list(__seq_possible_language_pair)
    # end if

    return seq_possible_language_pair



def setup_tokenizer_and_model_halomi(source_lang_code: str,
                                     target_lang_code: str,
                                     model_name: str,
                                     path_dir_translation_cache: Path,
                                     translation_config: TranslationConfig) -> TransformerVectorExtractorVer2:

    # adding the language codes.
    path_dir_translation_cache_language_code = path_dir_translation_cache / f'{source_lang_code}_{target_lang_code}'
    # assert path_dir_translation_cache_language_code.exists(), f'No translation cache directory found. Hint: execute the `do_translation` first or manually make the directory at {path_dir_translation_cache_language_code}'
    
    translation_handler = TransformersTranslationModelHandlerVer2(
        src_lang=source_lang_code,
        target_lang=target_lang_code,
        model_name=model_name,
        path_cache_dir=path_dir_translation_cache_language_code,
        is_save_convert_float16=translation_config.is_save_convert_fp16
    )
    vector_extractor = TransformerVectorExtractorVer2(translation_handler)

    return vector_extractor

def setup_tokenizer_and_model_fairseq(model_config: ModelConfigLfanHall,
                                      path_cache_dir: Path,
                                      translation_config: TranslationConfig,
                                      ) -> ty.Union[FairSeqVectorExtractorVer2CustomTranslationHandlerVer1, FairSeqVectorExtractorVer2]:

    if translation_config.option_embedding_layer is not None and translation_config.option_embedding_layer == ["decoder.word_embedding"]:
        model_encoder_decoder_mt = loading_model_fairseq(
            path_fairseq_model_dir=model_config.path_fairseq_model_dir,
            path_fairseq_model_file=model_config.path_fairseq_model_file,
            model_sentencepiece=model_config.path_sentencepiece_model
        )
        translation_handler = FaiseqTranslationModelHandler(
            model_encoder_decoder_mt=model_encoder_decoder_mt,
            n_sampling=-1,  # dummy value
            data_format_return='ver2',
            path_cache_dir=path_cache_dir,
            is_use_cache=True,
            is_zlib_compress=True
        )
        vector_extractor = FairSeqVectorExtractorVer2CustomTranslationHandlerVer1(translation_handler)

        return vector_extractor
    else:
        translation_handler = FairSeqTranslationModelHandlerVer2(
            path_dir_fairseq_model=model_config.path_fairseq_model_dir,
            path_model_checkpoint=model_config.path_fairseq_model_file,
            path_sentencepiece_model=model_config.path_sentencepiece_model,
            path_cache_dir=path_cache_dir,
            is_use_cache=True,
            is_zlib_compress=True
        )
        vector_extractor = FairSeqVectorExtractorVer2(translation_handler)

        return vector_extractor



# --------------------------------------------------------------


def classify_dataset_type(record: ty.Union[HalomiDatasetRecord, WMTDatasetRecord]) -> str:
    if isinstance(record, HalomiDatasetRecord):
        return 'halomi'
    elif isinstance(record, WMTDatasetRecord):
        return 'lfan_hall'
    else:
        raise ValueError()
# end def


def proposal_mmd_flagging_trajectory_ver3(
        config_name: str,
        dataset_name: str,
        llm_model_config: LlmModelConfig,
        translation_config: TranslationConfig,
        resource_config: ResourceConfig,
        algorithm_config_obj: MmdErrorFlaggerTrajectoryVer3Config,
        seq_dataset_records: ty.Sequence[ty.Union[HalomiDatasetRecord, WMTDatasetRecord]],
        management_db_handler: DBHandlerExp,
        path_flagger_cache_dir: Path):
    
    path_subdir_flagger_halomi = path_flagger_cache_dir / 'halomi'
    path_subdir_flagger_lfan_hall = path_flagger_cache_dir / 'lfan_hall'
    path_subdir_flagger_halomi.mkdir(parents=True, exist_ok=True)
    path_subdir_flagger_lfan_hall.mkdir(parents=True, exist_ok=True)

    path_translation_cache_halomi = translation_config.path_cache_translator / 'halomi'
    path_translation_cache_lfan_hall = translation_config.path_cache_translator / 'lfan_hall'
    path_translation_cache_halomi.mkdir(parents=True, exist_ok=True)
    path_translation_cache_lfan_hall.mkdir(parents=True, exist_ok=True)
    
    @dataclass
    class FunctionArgsFlagMain:
        """A function argument object.
        I want to pack the arguments into a container."""
        source_language_code: str
        target_language_code: str
        n_sampling: int
        tau_sequence: ty.List[float]
        target_embedding_layer: str
        vector_preprocess: str  # 'avg', 'concat'
        max_token_length_vector_concat: str
        mode_max_token_length: int
        kernel_type: str
        kernel_gaussian_length_scale: int
        kernel_gaussian_length_scale_computation: str
        trajectory_rule: str
        trajectory_rule_smoothing: str
        trajectory_rule_smoothing_window: ty.Optional[int]
        path_subdir_flagger_cache: Path
        mmd_estimator: QuadraticMmdEstimator
        tensor_preprocessor: TensorPreprocessorVer1
        # option_is_sampling_in_iteration: bool = False        
    # end


    def __pack_args_fields_db(args: FunctionArgsFlagMain) -> ty.Tuple[str, str, str]:
        """I make a pack of arguments into a dict. The DB record requires the dict object having the argument parameters."""
        args_kernel_options = dict(
            kernel_type=args.kernel_type,
            kernel_gaussian_length_scale=args.kernel_gaussian_length_scale,
            kernel_gaussian_length_scale_computation=args.kernel_gaussian_length_scale_computation
        )
        args_trajectory_options = dict(
            trajectory_rule=args.trajectory_rule,
            trajectory_rule_smoothing=args.trajectory_rule_smoothing,
            trajectory_rule_smoothing_window=args.trajectory_rule_smoothing_window
        )
        args_translation_options = asdict(translation_config)

        return (
            json.dumps(args_kernel_options, default=path_to_str),
            json.dumps(args_trajectory_options, default=path_to_str), 
            json.dumps(args_translation_options, default=path_to_str)
        )

    def _generate_db_eval_record(
            seq_dataset_reocrd: ty.List[EvaluationTargetTranslationPair],
            args: FunctionArgsFlagMain) -> ty.List[DbTableRecordProposalMmdFlaggerTrajectoryVer3]:
        """I generate the db record object. At the moment, the evaluation result is not ready yet.
        So, I set algorithm parameter only. I can set the algorithm result later."""
        seq_db_record = []
        args_kernel_options, args_trajectory_options, args_translation_options = __pack_args_fields_db(args)
        for _dataset_record in seq_dataset_reocrd:
            tau_sequence_json = json.dumps([float(_v) for _v in args.tau_sequence])
            _db_record = DbTableRecordProposalMmdFlaggerTrajectoryVer3(
                approach_name=MmdErrorFlaggerTrajectoryVer3.__name__,
                dataset_name=dataset_name,
                sentence_id=_dataset_record.sentence_id,
                source_language_code=args.source_language_code,
                target_language_code=args.target_language_code,
                n_sampling=args.n_sampling,
                tau_sequence=tau_sequence_json,
                mode_vector_preprocess=args.vector_preprocess,
                mode_target_embedding_layer=args.target_embedding_layer,
                mode_max_token_length_vector_concat=str(args.max_token_length_vector_concat),
                mode_max_token_length=args.mode_max_token_length,
                args_kernel_options_json=args_kernel_options,
                args_trajectory_options_json=args_trajectory_options,
                args_translation_options_json=args_translation_options,
                flagging_label=None,  # type: ignore , place holder
                flagging_argument_json=None  # type: ignore , place holder
            )
            seq_db_record.append(_db_record)
        # end for
        return seq_db_record

    def _filter_db_record(args: FunctionArgsFlagMain,
                          seq_db_record_all: ty.List[DbTableRecordProposalMmdFlaggerTrajectoryVer3]
                          ) -> ty.List[DbTableRecordProposalMmdFlaggerTrajectoryVer3]:
        """Filter out db records that already processed."""
        seq_db_record = []
        for _record in seq_db_record_all:
            assert _record.record_id is not None
            _is_exist = management_db_handler.is_record_exists(
                table_name=DbTableRecordProposalMmdFlaggerTrajectoryVer3.__name__.__str__(),
                exp_key=_record.record_id,
                is_partially_search=False,
                primary_key='record_id')
            if _is_exist is False:
                seq_db_record.append(_record)
            # end if
        # end for
        return seq_db_record

    def _execute_flagging(args: FunctionArgsFlagMain,
                          seq_dataset_reocrd: ty.List[EvaluationTargetTranslationPair],
                          vector_extractor: ty.Union[TransformerVectorExtractorVer2, FairSeqVectorExtractorVer2]
                          ):
        """Main procedure of flagging."""
        assert len(args.tau_sequence) > 0, "tau_sequence should not be empty."
        _seq_db_records_all = _generate_db_eval_record(seq_dataset_reocrd=seq_dataset_reocrd, 
                                                       args=args)
        seq_db_processing_records = _filter_db_record(args, _seq_db_records_all)
        
        if len(seq_db_processing_records) == 0:
            root_logger.info(f"All records already all processed.")
            return
        # end if

        # set the flagger object
        mmd_flagger = MmdErrorFlaggerTrajectoryVer3(
            vector_extractor=vector_extractor,
            tensor_preprocessor=args.tensor_preprocessor,
            mmd_estimator=args.mmd_estimator,
            mode_target_embedding_layer=args.target_embedding_layer,
            option_translation_max_a=translation_config.max_len_a,
            option_translation_max_b=translation_config.max_len_b,
            path_cache_dir=args.path_subdir_flagger_cache,
            trajectory_rule=args.trajectory_rule,
            trajectory_rule_smoothing=args.trajectory_rule_smoothing,
            trajectory_rule_smoothing_window=args.trajectory_rule_smoothing_window,
            )
        
        __d_sent_id2record = {_r.sentence_id: _r for _r in seq_dataset_reocrd}
        
        _db_record: DbTableRecordProposalMmdFlaggerTrajectoryVer3
        for _db_record in seq_db_processing_records:
            _dataset_record = __d_sent_id2record[_db_record.sentence_id]
            
            # execution.
            root_logger.debug(f"Processing sentence_id: {_dataset_record.sentence_id}")

            try:
                _flag_result = mmd_flagger.flag_hallucination_one_record(
                    eval_target=_dataset_record,
                    candidate_temperature_parameters=args.tau_sequence,
                    n_sampling=args.n_sampling)
                root_logger.debug(f"Done: {_dataset_record.sentence_id}")
            except (AssertionError, ValueError, ParameterSettingException, RuntimeError) as e:
                # I assume an AssertionError involving trajectory rule.
                # TODO: I refine the try-except block: e.g. putting a specific Exception Class.
                root_logger.error(f"Encountering an AssertionError. Skip this execution: {e}")
                continue
            else:
                # saving into the db.
                # _arg_obj = asdict(_flag_result)
                # I delete unnecessary fields (translations.)
                _flag_result.hypothesis_translation = []  # the translation takes too much size. Refer to the translation module cache file if you want to see the translation.

                root_logger.debug(f"Saving the result into the db.")

                # updating the db records.
                _db_record.flagging_label = _flag_result.is_hallucination
                _db_record.flagging_argument_json = json.dumps(asdict(_flag_result))

                record_dict = asdict(_db_record)
                management_db_handler.insert(table_name=DbTableRecordProposalMmdFlaggerTrajectoryVer3.__name__, data=record_dict)
                root_logger.debug(f"Done saving the result into the db.")
            # end try
        # end for
    # end def

    def _function_loop_over_language_pair_halomi(args: FunctionArgsFlagMain, 
                                                 seq_dataset: ty.Sequence[HalomiDatasetRecord]):
        """Designed for the Halomi Dataset.
        For the halomi dataset, I have to set a loop for a language pair.
        For each language pair, I initialise a tokenizer, model, selecting records from the dataset.
        """
        seq_possible_language_pair = making_language_pair(resource_config=resource_config, seq_dataset=seq_dataset)

        # I make a sub-directory to save MMD estimator
        path_flagger_cache_subdir = path_subdir_flagger_halomi / f'{args.target_embedding_layer}_{args.vector_preprocess}_{args.max_token_length_vector_concat}_{args.kernel_type}_{args.kernel_gaussian_length_scale}'
        path_flagger_cache_subdir.mkdir(parents=True, exist_ok=True)

        # making the pair of src, tgt lang.
        for __t_source_target_lang_code in seq_possible_language_pair:
            _source_lang_code, _target_lang_code = __t_source_target_lang_code

            path_dir_translation_cache_language_code = path_translation_cache_halomi / f'{_source_lang_code}_{_target_lang_code}' 
            assert path_dir_translation_cache_language_code.exists(), f'No translation cache directory found. Hint: execute the `do_translation` first or manually make the directory at {path_dir_translation_cache_language_code}'

            assert isinstance(llm_model_config.model_halomi, ModelConfigHalomi)
            # setting the tokenizer and model.
            __vector_extractor = setup_tokenizer_and_model_halomi(
                translation_config=translation_config,
                source_lang_code=_source_lang_code,
                target_lang_code=_target_lang_code,
                model_name=llm_model_config.model_halomi.nllb_model_name,
                path_dir_translation_cache=path_translation_cache_halomi)

            # selecting the records.
            _seq_target_record = select_record_lang_code(seq_dataset_record=seq_dataset, 
                                                         src_lang_code=_source_lang_code, 
                                                         tgt_lang_code=_target_lang_code)

            # selecting the calibration text.
            _seq_calibration_record = _select_calibration_record_and_convert(
                seq_dataset_record=_seq_target_record,
                n_calibration_records=resource_config.n_calibration_records,
                method_select_calibration=resource_config.method_select_calibration,
                seed_random_seed=resource_config.seed_random_seed)
            
            _seq_dataset_eval = _seq_target_record

            # I do filter the dataset records by the option of `filter_target_label`.
            if resource_config.filter_target_label is not None:
                _seq_dataset_eval = [_record for _record in _seq_dataset_eval if _record.class_hall in resource_config.filter_target_label]
                assert len(_seq_dataset_eval) > 0, f"len(_seq_dataset_eval)={len(_seq_dataset_eval)}. The filter_target_label option is not correct. Incorrect hallucination labels: {resource_config.filter_target_label}"
            # end if
            
            # limiting the number of records for debugging and testing.
            if resource_config.limit_dataset_record is not None:
                _seq_dataset_eval = _seq_dataset_eval[:resource_config.limit_dataset_record]
            # end if

            # halomi records -> `EvaluationTargetTranslationPair`
            _seq_dataset_eval_type = [
                EvaluationTargetTranslationPair(source=_r.src_text, target=_r.tgt_text, sentence_id=_r.key_unique)  # type: ignore
                for _r in _seq_dataset_eval]

            # ----------------------------------------------------------------
            # quick check of DB records. I want to run quick check before starting the length scale computation.
            _seq_db_records_all = _generate_db_eval_record(
                seq_dataset_reocrd=_seq_dataset_eval_type, 
                args=args)
            seq_db_processing_records = _filter_db_record(args, _seq_db_records_all)
            if len(seq_db_processing_records) == 0:
                root_logger.info(f'All end. Skip it.')
                continue
            # end if
            # ----------------------------------------------------------------

            # load the MMD estimator if the model file is available.
            # I use this directory for saving MMD estimator.
            path_flagger_cache_subdir_lang_pair = path_flagger_cache_subdir / f'{_source_lang_code}_{_target_lang_code}'
            path_flagger_cache_subdir_lang_pair.mkdir(parents=True, exist_ok=True)

            # _t_cached_mmd_estimator = load_mmd_estimator(path_flagger_cache_subdir_lang_pair)
            # if _t_cached_mmd_estimator is None:
            init_mmd_estimator = MmdEstimatorInitialiserVer1(
                tensor_preprocessor=args.tensor_preprocessor,
                vector_extractor=__vector_extractor,
                mode_target_embedding_layer=args.target_embedding_layer,
                kernel_type=args.kernel_type,
                kernel_length_scale_percentile=args.kernel_gaussian_length_scale, 
                kernel_length_scale_median_option=args.kernel_gaussian_length_scale_computation,
                path_cache_dir=path_flagger_cache_subdir_lang_pair,
                option_translation_max_a=translation_config.max_len_a,
                option_translation_max_b=translation_config.max_len_b
            )
            root_logger.info(f"computing the kernel length scale")
            _mmd_estimator = init_mmd_estimator.get_mmd_estimator(
                seq_calibration_text=_seq_calibration_record,
            )
            root_logger.info(f"done computing the kernel length scale")            

            init_mmd_estimator.save_mmd_estimator()
            # else:
            #     _mmd_estimator, _mmd_config_obj = _t_cached_mmd_estimator
            #     assert 'option_max_token_length' in _mmd_config_obj
            #     args.tensor_preprocessor.option_max_token_length = _mmd_config_obj['option_max_token_length']
            # # end if

            # replacing the function args.
            _local_args = copy.deepcopy(args)  # copy the original object, not to update it.
            _local_args.source_language_code = _source_lang_code
            _local_args.target_language_code = _target_lang_code
            _local_args.mmd_estimator = _mmd_estimator
            _local_args.path_subdir_flagger_cache = path_flagger_cache_subdir_lang_pair

            _execute_flagging(_local_args,
                              seq_dataset_reocrd=_seq_dataset_eval_type,
                              vector_extractor=__vector_extractor)
        # end for
    # end def

    def _function_loop_lfan_hall(args: FunctionArgsFlagMain, seq_dataset: ty.Sequence[WMTDatasetRecord]):
        """Designed for the LFAN-Hall Dataset.
        """
        # making the pair of src, tgt lang.
        assert isinstance(llm_model_config.model_lfan_hall, ModelConfigLfanHall)
        __vector_extractor = setup_tokenizer_and_model_fairseq(
            llm_model_config.model_lfan_hall,
            path_cache_dir=path_translation_cache_lfan_hall,
            translation_config=translation_config)
        
        # selecting the calibration text.
        _seq_calibration_record = _select_calibration_record_and_convert(
            seq_dataset_record=seq_dataset,
            n_calibration_records=resource_config.n_calibration_records,
            method_select_calibration=resource_config.method_select_calibration,
            seed_random_seed=resource_config.seed_random_seed)

        
        # load the MMD estimator if the model file is available.
        # I use this directory for saving MMD estimator.
        # 2025-05-06 tmp commenting out
        # _t_cached_mmd_estimator = load_mmd_estimator(args.path_subdir_flagger_cache)
        # if _t_cached_mmd_estimator is None:
        init_mmd_estimator = MmdEstimatorInitialiserVer1(
            tensor_preprocessor=args.tensor_preprocessor,
            vector_extractor=__vector_extractor,
            mode_target_embedding_layer=args.target_embedding_layer,
            kernel_type=args.kernel_type,
            kernel_length_scale_percentile=args.kernel_gaussian_length_scale,
            kernel_length_scale_median_option=args.kernel_gaussian_length_scale_computation,
            path_cache_dir=path_subdir_flagger_lfan_hall,
            option_translation_max_a=translation_config.max_len_a,
            option_translation_max_b=translation_config.max_len_b
        )
        root_logger.info(f"computing the kernel length scale")
        _mmd_estimator = init_mmd_estimator.get_mmd_estimator(
            seq_calibration_text=_seq_calibration_record,
        )

        root_logger.info(f"done the kernel length scale")

        init_mmd_estimator.save_mmd_estimator()
        # else:
        #     _mmd_estimator, _mmd_config_obj = _t_cached_mmd_estimator
        #     assert 'option_max_token_length' in _mmd_config_obj
        #     args.tensor_preprocessor.option_max_token_length = _mmd_config_obj['option_max_token_length']
        # end if


        seq_dataset = filter_dataset_record(translation_config, resource_config, seq_dataset)
        # limiting the number of records for debugging and testing.
        if resource_config.limit_dataset_record is not None:
            _seq_dataset_eval = seq_dataset[:resource_config.limit_dataset_record]
        else:
            _seq_dataset_eval = seq_dataset
        # end if

        _seq_dataset_eval_type = [
            EvaluationTargetTranslationPair(source=_r.source, target=_r.translation, sentence_id=str(_r.sentence_id)) for _r in _seq_dataset_eval]

        _local_args = copy.deepcopy(args)
        _local_args.source_language_code = 'deu_Latn'
        _local_args.target_language_code = 'eng_Latn'
        _local_args.mmd_estimator = _mmd_estimator
        _local_args.path_subdir_flagger_cache = path_subdir_flagger_lfan_hall

        _execute_flagging(_local_args,
                          seq_dataset_reocrd=_seq_dataset_eval_type,
                          vector_extractor=__vector_extractor)

    # end def


    def _update_option_combinations(option_combinations: ty.List[ty.Tuple[ty.Any,...]]) -> ty.List[ty.Tuple[ty.Any,...]]:
        """defining adhoc rules."""
        # update rule.
        def rule_update_vector_preprocess_concat(t_option: ty.Tuple):
            # [Rule] when the `option_vector_preprocess` = 'avg', then `option_max_token_length_vector_concat` is None.
            list_options = list(t_option)
            _option_embedding_layer = list_options[3]
            _option_vector_preprocess = list_options[4]
            
            if _option_embedding_layer == 'avg':
                _option_vector_preprocess = None
                list_options[4] = _option_vector_preprocess
            # end if 
            return list_options
        # end def

        def rule_update_smoothing_window_size(t_option):
            # [Rule] when the trajectory filter is no_filter, then smoothing window size = 1
            list_options = list(t_option)
            _option_filter_name = list_options[7]
            _option_window_size = list_options[8]
            
            if _option_filter_name == 'no_filter':
                list_options[8] = 1
            # end if 
            return list_options
        # end def

        def rule_update_dot_product_kernel(t_option):
            # [Rule] when the kernel type is dot-product kernel, then set `option_kernel_gaussian_length_scale` and `option_kernel_gaussian_length_scale_computation`
            list_options = list(t_option)
            _option_kernel_type: str = list_options[9]
            _option_kernel_gaussian_length_scale: int = list_options[10]
            _option_kernel_gaussian_length_scale_computation: str = list_options[11]

            if _option_kernel_type == 'dot':
                _option_kernel_gaussian_length_scale = -1
                list_options[10] = _option_kernel_gaussian_length_scale
                _option_kernel_gaussian_length_scale_computation = 'no-gaussian'
                list_options[11] = _option_kernel_gaussian_length_scale_computation
            else:
                pass
            # end if
            return list_options

        def apply_rules(list_option):
            list_option = rule_update_vector_preprocess_concat(list_option)
            list_option = rule_update_smoothing_window_size(list_option)
            list_option = rule_update_dot_product_kernel(list_option)

            return tuple(list_option)
        # end def

        # I just wanna remove duplicated args
        seq_rule_applied = [apply_rules(l) for l in option_combinations]
        seq_json_no_dumplication = list(set([json.dumps(_t, ensure_ascii=False) for _t in seq_rule_applied]))
        # sorting the order
        seq_json_no_dumplication = list(sorted(seq_json_no_dumplication))
        seq_updated_args = [json.loads(_string_json) for _string_json in seq_json_no_dumplication]
        return seq_updated_args
    # end def

    def _function_loop_options(algorithm_config_obj: MmdErrorFlaggerTrajectoryVer3Config):
        """A closure of the main loop."""
        # parameter combinations
        # set option parameters
        option_n_sampling: ty.List[int] = algorithm_config_obj.option_n_translation_sampling
        option_temperature_sequence: ty.List = algorithm_config_obj.option_temperature_sequence
        option_embedding_layer: ty.List[str] = algorithm_config_obj.option_embedding_layer

        # WARNING! wHEN YOU CHANGE THE ARGUMENT ORDER, DOUBLE-CHECK `_update_option_combinations`.
        option_combinations = list(itertools.product(
            option_n_sampling, 
            option_temperature_sequence, 
            option_embedding_layer,
            algorithm_config_obj.option_vector_preprocess,
            algorithm_config_obj.option_max_token_length_vector_concat,
            algorithm_config_obj.option_max_token_length,
            algorithm_config_obj.option_trajectory_rule_versions,
            algorithm_config_obj.option_trajectory_rule_smoothing,
            algorithm_config_obj.option_trajectory_rule_smoothing_window,
            algorithm_config_obj.option_kernel_type,
            algorithm_config_obj.option_kernel_gaussian_length_scale,
            algorithm_config_obj.option_kernel_gaussian_length_scale_computation
            ))
        option_combinations = _update_option_combinations(option_combinations)
        root_logger.info(f"List of executed arguments -> {json.dumps(option_combinations, ensure_ascii=False)}")
        
        for _t in tqdm.tqdm(option_combinations, desc="Option Combination"):
            _n_sampling, _tau_sequence, _target_embedding_layer, _vector_preprocess, _max_token_length_vector_concat, _option_max_token_length, _trajectroy_rule, _trajectory_rule_smoothing, _trajectory_rule_smoothing_window, _kernel_type, _kernel_gaussian_length_scale, _kernel_gaussian_length_scale_computation = _t
            # updating `trajectory_rule_smoothing_window` when `_trajectory_rule_smoothing` is no_filter
            if _trajectory_rule_smoothing == "no_filter":
                _trajectory_rule_smoothing_window = None
            # end if

            if _max_token_length_vector_concat == "max_calibration":
                _option_max_token_length = -1
            elif _max_token_length_vector_concat == "fixed":
                _option_max_token_length = _option_max_token_length
            elif _max_token_length_vector_concat is None:
                _option_max_token_length = -1
            else:
                raise ValueError()
            # end if

            _msg_logging = f'Started option combination -> {_t}'
            root_logger.info(_msg_logging)
            if is_use_slack:
                assert slack_client is not None
                try:
                    slack_notifier.send_message(webhook=slack_client, message=_msg_logging)
                except Exception as e:
                    root_logger.error(f"Failed sending message to slack -> {e}")
                    continue
            # end if

            _tensor_preprocessor = TensorPreprocessorVer1(
                mode_vector_preprocess=_vector_preprocess,
                mode_max_token_length_vector_concat=_max_token_length_vector_concat,
                option_max_token_length=_option_max_token_length
            )

            if dataset_type == 'halomi':
                # I switch the loop by the dataset type.                
                args = FunctionArgsFlagMain(
                    source_language_code=None,  # type: ignore , place holder  
                    target_language_code=None,  # type: ignore , place holder 
                    n_sampling=_n_sampling,
                    tau_sequence=_tau_sequence,
                    target_embedding_layer=_target_embedding_layer,
                    vector_preprocess=_vector_preprocess,
                    max_token_length_vector_concat=_max_token_length_vector_concat,
                    mode_max_token_length=_option_max_token_length,
                    kernel_type=_kernel_type,
                    kernel_gaussian_length_scale=_kernel_gaussian_length_scale,
                    kernel_gaussian_length_scale_computation=_kernel_gaussian_length_scale_computation,
                    trajectory_rule=_trajectroy_rule,
                    trajectory_rule_smoothing=_trajectory_rule_smoothing,
                    trajectory_rule_smoothing_window=_trajectory_rule_smoothing_window,
                    path_subdir_flagger_cache=path_subdir_flagger_halomi,
                    mmd_estimator=None,  # type: ignore , place holder
                    tensor_preprocessor=_tensor_preprocessor,
                ) # type: ignore
                assert all(isinstance(_r, HalomiDatasetRecord) for _r in seq_dataset_records)
                _function_loop_over_language_pair_halomi(args=args, seq_dataset=seq_dataset_records)
            elif dataset_type == 'lfan_hall':

                # making a subdirectory for flagger cache directory.
                _path_cache_sudir = path_subdir_flagger_lfan_hall / f'{_target_embedding_layer}-{_vector_preprocess}-{_max_token_length_vector_concat}-{_option_max_token_length}' 
                _path_cache_sudir.mkdir(parents=True, exist_ok=True)


                args = FunctionArgsFlagMain(
                    source_language_code='deu_Latn',  # place holder
                    target_language_code='eng_Latn',  # place holder
                    n_sampling=_n_sampling,
                    tau_sequence=_tau_sequence,
                    target_embedding_layer=_target_embedding_layer,
                    vector_preprocess=_vector_preprocess,
                    max_token_length_vector_concat=_max_token_length_vector_concat,
                    mode_max_token_length=_option_max_token_length,
                    kernel_type=_kernel_type,
                    kernel_gaussian_length_scale=_kernel_gaussian_length_scale,
                    kernel_gaussian_length_scale_computation=_kernel_gaussian_length_scale_computation,
                    trajectory_rule=_trajectroy_rule,
                    trajectory_rule_smoothing=_trajectory_rule_smoothing,
                    trajectory_rule_smoothing_window=_trajectory_rule_smoothing_window,
                    path_subdir_flagger_cache=_path_cache_sudir,
                    mmd_estimator=None,  # type: ignore , place holder
                    tensor_preprocessor=_tensor_preprocessor,
                ) # type: ignore

                _function_loop_lfan_hall(args=args, seq_dataset=seq_dataset_records)
            # end if

            _msg_logging = f'End option combination -> {_t}'
            root_logger.info(_msg_logging)
            if is_use_slack:
                assert slack_client is not None
                try:
                    slack_notifier.send_message(webhook=slack_client, message=_msg_logging)
                except Exception as e:
                    root_logger.error(f"Failed sending message to slack -> {e}")
                    continue
            # end if

        # end for
    # end def

    # function main parts    
    _dataset_type = set([classify_dataset_type(_r) for _r in seq_dataset_records])
    assert len(_dataset_type) == 1
    dataset_type: str = list(_dataset_type)[0]
    
    # -----------------------------------------------------------------------------

    _function_loop_options(algorithm_config_obj)
# end def




# def oscillatory_detection_Raunak_2021_flagging(
#         config_name: str,
#         resource_config: ResourceConfig,
#         raunak_2021_config_obj: Raunak2021ApproachConfig,
#         seq_dataset_eval: ty.List[HalomiDatasetRecord],
#         management_db_handler: DBHandlerExp,
#         file_logger: logging.Logger):
#     module_script_path = oscillatory_detection_Raunak_2021.__file__
#     path_python37_exec = raunak_2021_config_obj.path_python37_exec

#     assert Path(module_script_path).exists(), f"Module script not found at {module_script_path}"

#     _table_name = DbTableRecordRaunak2021.__name__

#     def execute_flagging(source: str, output: str) -> ty.Dict:
#         args_json = json.dumps(
#             {"source": source, 
#              "output": output,
#              "ngram_size": raunak_2021_config_obj.ngram_size,
#              "count_threshold": raunak_2021_config_obj.count_threshold,
#              "difference_threshold": raunak_2021_config_obj.difference_threshold,
#              "min_length_threshold": raunak_2021_config_obj.min_length_threshold             
#         })
#         result = subprocess.run(
#             [
#                 path_python37_exec, 
#                 module_script_path,
#                 "--args",
#                 args_json
#             ],
#             capture_output=True,
#             text=True
#         )
#         try:
#             return_obj = json.loads(result.stdout)  # Assuming Project B outputs JSON
#             return return_obj
#         except json.JSONDecodeError:
#             return {"status": "error", "error": result.stderr}
#     # end def

#     # I create the combination of src-target language.
#     seq_language_pair = making_language_pair(resource_config=resource_config, seq_dataset=seq_dataset_eval)


#     # main loop
#     # For each src-target language pair, I compute the log-probability.
#     for _tuple_lang_pair in seq_language_pair:
#         _src_lang_code, _tgt_lang_code = _tuple_lang_pair

#         # I select records having the src-target language.
#         # selecting the records.
#         _seq_target_record = select_record_lang_code(seq_dataset_record=seq_dataset_eval, 
#                                                         src_lang_code=_src_lang_code, 
#                                                         tgt_lang_code=_tgt_lang_code)
#         assert len(_seq_target_record) > 0, f"len(_seq_target_record)={len(_seq_target_record)}"

#         for _record in _seq_target_record:
#         # for _record in seq_dataset_record:
#             _sentence_id = _record.key_unique
#             _db_record_id = DbTableRecordRaunak2021.get_record_id(config_name, raunak_2021_config_obj.approach_name, str(_sentence_id))

#             _is_exist = management_db_handler.is_record_exists(
#                 table_name=_table_name,
#                 exp_key=_db_record_id,
#                 is_partially_search=False,
#                 primary_key='record_id')

#             if _is_exist:
#                 continue
#             # end if

#             source = _record.src_text
#             output = _record.tgt_text

#             result = execute_flagging(source, output)
#             assert 'status' in result, f"Expected 'status' in the result, got {result}"
#             if result['status'] == 'success':
#                 record_ = DbTableRecordRaunak2021(
#                     config_name=config_name,
#                     approach_name=raunak_2021_config_obj.approach_name,
#                     sentence_id=str(_sentence_id),
#                     flagging_label=result['result'],
#                     flagging_argument_json=json.dumps(result)
#                 )
#                 record_dict = asdict(record_)
#                 management_db_handler.insert(table_name=DbTableRecordRaunak2021.__name__, 
#                                             data=record_dict)
#             else:
#                 root_logger.error(f"{raunak_2021_config_obj.approach_name} failed for sentence_id: {_sentence_id}")
#             # end if
#     # end for


# def seq_log_prob_Guerreiro_2023_flagging(
#         config_name: str,
#         resource_config: ResourceConfig,
#         algorithm_config_obj: Guerreiro2023SeqLogprob,
#         seq_dataset_eval: ty.List[HalomiDatasetRecord],
#         management_db_handler: DBHandlerExp,
#         file_logger: logging.Logger,
#         path_cache_dir: Path):
#     # I set the cache directory.
#     # I create the combination of src-target language.
#     seq_language_pair = making_language_pair(resource_config=resource_config, seq_dataset=seq_dataset_eval)

#     # For each src-target language pair, I compute the log-probability.
#     for _tuple_lang_pair in seq_language_pair:
#         _src_lang_code, _tgt_lang_code = _tuple_lang_pair

#         path_cache_dir_algorithm = path_cache_dir / Guerreiro2023SeqLogprob.__name__ / f"{_src_lang_code}_{_tgt_lang_code}"
#         path_cache_dir_algorithm.mkdir(parents=True, exist_ok=True)
        
#         # setting the tokenizer and model.
#         __translation_module, __vector_extractor = setup_tokenizer_and_model(
#             source_lang_code=_src_lang_code,
#             target_lang_code=_tgt_lang_code,
#             embedding_layer="decoder.embed_tokens",  # Note: I may change this later. But, I fix it for now.
#             model_name=algorithm_config_obj.nllb_model_name,
#             max_len_a=algorithm_config_obj.max_len_a,
#             max_len_b=algorithm_config_obj.max_len_b)

#         # I select records having the src-target language.
#         # selecting the records.
#         _seq_target_record = select_record_lang_code(seq_dataset_record=seq_dataset_eval, 
#                                                         src_lang_code=_src_lang_code, 
#                                                         tgt_lang_code=_tgt_lang_code)
#         assert len(_seq_target_record) > 0, f"len(_seq_target_record)={len(_seq_target_record)}"

#         seq_eval_records = [
#             EvaluationTargetTranslationPair(
#                 source=__record.src_text, 
#                 target=__record.tgt_text, 
#                 sentence_id=str(__record.key_unique))
#                 for __record in _seq_target_record]
#         assert len(seq_eval_records) > 0, f"len(seq_eval_records)={len(seq_eval_records)}"

#         # I initialise the flagger
#         __flagger = TransformerFlaggerSeqLogProbability(
#             translation_handler=__translation_module,
#             path_dir_cache=path_cache_dir_algorithm,)

#         # I set the threshold for flagging.
#         _seq_log_prob = __flagger.compute_dataset_log_probability(seq_eval_records,)
#         threshold_flag = __flagger.get_flag_threshold(
#             seq_log_probability=_seq_log_prob, 
#             percentile=algorithm_config_obj.percentile_threshold)

#         # I execute the flagging.
#         for _record in tqdm.tqdm(seq_eval_records, desc=f"Processing {algorithm_config_obj.approach_name}", file=sys.stdout):
#             # I check the result DB. Skip if the record is already processed.
#             _exp_primary_key = DbTableRecordGuerreiro2023SeqLogProb.get_record_id(
#                 config_name=config_name,
#                 _sentence_id=str(_record.sentence_id))
#             _is_exists = management_db_handler.is_record_exists(
#                 table_name=DbTableRecordGuerreiro2023SeqLogProb.__name__,
#                 exp_key=_exp_primary_key,
#                 is_partially_search=False,
#                 primary_key='record_id')
#             if _is_exists:
#                 continue
#             # end if

#             # I flag the record.
#             _log_prob_obj = __flagger.flag(source_text=_record.source, 
#                                            threshold=threshold_flag,
#                                            sentence_id=_record.sentence_id,)
#             _db_record = DbTableRecordGuerreiro2023SeqLogProb(
#                 config_name=config_name,
#                 sentence_id=str(_record.sentence_id),
#                 log_probability=_log_prob_obj.log_probability,
#                 log_probability_threshold=threshold_flag,
#                 flagging_label=_log_prob_obj.is_hallucination,
#                 flagging_argument_json=json.dumps(asdict(_log_prob_obj))
#             )
#             _record_dict = asdict(_db_record)
#             management_db_handler.insert(table_name=DbTableRecordGuerreiro2023SeqLogProb.__name__, data=_record_dict)
#     # end for


# def mcd_sim_Guerreiro_2023_flagging(
#         config_name: str,
#         algorithm_config_obj: Guerreiro2023MCDSim,
#         seq_dataset_eval: ty.List[HalomiDatasetRecord],
#         management_db_handler: DBHandlerExp,
#         file_logger: logging.Logger,
#         path_cache_dir: Path):
#     # I set the cache directory.
#     path_cache_dir_algorithm = path_cache_dir / Guerreiro2023MCDSim.__name__
#     path_cache_dir_algorithm.mkdir(parents=True, exist_ok=True)

#     model_encoder_decoder_mt = load_fairseq_model(
#         path_fairseq_model_dir=algorithm_config_obj.path_fairseq_model_dir,
#         path_fairseq_model_file=algorithm_config_obj.path_fairseq_model_file,
#         path_sentencepiece_model=algorithm_config_obj.path_sentencepiece_model)

#     # I fix the method with METEOR.    
#     # file_logger.debug(f"Initialising the metric object.")    
#     # try:
#     #     metric_obj_class = getattr(flagger_mc_dropout, algorithm_config_obj.dissimilarity_metric_class_name)
#     # except AttributeError:
#     #     raise Exception(f"Metric class not found: {algorithm_config_obj.dissimilarity_metric_class_name}")
#     # # end try

#     flagger_obj = FlaggerDisSimilarityMcDropOut(
#         fairseq_interface=model_encoder_decoder_mt,
#         metric_obj=flagger_mc_dropout.MeteorMetric(),
#         path_dir_cache=path_cache_dir_algorithm,
#         num_samples=algorithm_config_obj.num_samples,
#         temperature_value=1.0,
#         max_len_a=algorithm_config_obj.max_len_a,
#         max_len_b=algorithm_config_obj.max_len_b)
    
#     seq_eval_records = [
#         EvaluationTargetTranslationPair(source=__record.source, target=__record.translation, sentence_id=str(__record.sentence_id))
#         for __record in seq_dataset_eval]

#     root_logger.info(f"Computing DSIM-MC-Drop over records of a dataset.")
#     # I compute the log-probability for each record and obtain the distribution of a dataset.
#     # [NOTE] I use the whole dataset for computing the distribution. That follows the description of Guerreiro, 2023, Appendix G.
#     # I compute the threshold for the log-probability.
#     seq_stats = flagger_obj.compute_dataset_statistics(seq_eval_records)
#     threshold_flag = flagger_obj.get_flag_threshold(seq_stats, algorithm_config_obj.percentile)
#     root_logger.info(f"Threshold for flagging: {threshold_flag}")

#     # I flag the records.
#     # seq_hallucination_records = [__r for __r in seq_dataset_eval if __r.error_type == "hallucination"]
#     for _record in tqdm.tqdm(seq_dataset_eval, desc=f"Processing {algorithm_config_obj.approach_name}", file=sys.stdout):
#         # I check the result DB. Skip if the record is already processed.
#         _exp_primary_key = DbTableRecordGuerreiro2023McDSIM.get_record_id(
#             config_name=config_name,
#             _sentence_id=str(_record.sentence_id))
#         _is_exists = management_db_handler.is_record_exists(
#             table_name=DbTableRecordGuerreiro2023McDSIM.__name__,
#             exp_key=_exp_primary_key,
#             is_partially_search=False,
#             primary_key='record_id')
#         if _is_exists:
#             continue
#         # end if

#         # I flag the record.
#         _res_obj_dsim_mc_dropout = flagger_obj.flag(
#             evaluation_target=EvaluationTargetTranslationPair(\
#                 source=_record.source,
#                 target=_record.translation,
#                 sentence_id=str(_record.sentence_id)), 
#             threshold=threshold_flag,
#             is_use_cache=True)

#         _db_record = DbTableRecordGuerreiro2023McDSIM(
#             config_name=config_name,
#             sentence_id=str(_record.sentence_id),
#             score=_res_obj_dsim_mc_dropout.avg_dissimilarity,
#             score_threshold=threshold_flag,
#             flagging_label=_res_obj_dsim_mc_dropout.is_hallucination,
#             flagging_argument_json=json.dumps(asdict(_res_obj_dsim_mc_dropout))
#         )
#         _record_dict = asdict(_db_record)
#         management_db_handler.insert(table_name=DbTableRecordGuerreiro2023McDSIM.__name__, data=_record_dict)
#     # end for



# --------------------------------------------------------------
# heler functions

def create_interface_config(config_dict: ty.Dict) -> InterfaceConfig:
    """
    Preprocessing before creating the dataclass instance
    """    
    def convert_paths_in_dict(d: ty.Dict) -> ty.Dict:
        """
        Recursively convert values to Path objects if the key starts with 'path'.
        """
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = convert_paths_in_dict(value)
            elif isinstance(value, str) and key.startswith("path"):
                d[key] = Path(value)
        return d
    # end def

    config_dict_path_converted = convert_paths_in_dict(config_dict)

    config_obj = dacite.from_dict(InterfaceConfig, config_dict_path_converted)
    
    # creating the root directory
    config_obj.resource_config.path_work_dir.mkdir(parents=True, exist_ok=True)
    (config_obj.resource_config.path_work_dir / config_obj.resource_config.dir_name_cache).mkdir(parents=True, exist_ok=True)
    (config_obj.resource_config.path_work_dir / config_obj.resource_config.dir_name_log).mkdir(parents=True, exist_ok=True)    

    return config_obj


def setup_slack_wekhook(path_config: Path, mode_name: str) -> ty.Tuple[ty.Optional[WebhookClient], bool]:
    config_obj = toml.load(path_config)
    interface_config = create_interface_config(config_obj)
    
    if interface_config.notification_config is None:
        return None, False
    elif interface_config.notification_config.slack_wekhook is None:
        return None, False
    else:
        client = WebhookClient(interface_config.notification_config.slack_wekhook)
        slack_notifier.send_message(client, f'---\n\n start script. mode_name -> {mode_name}, path_config -> {path_config.as_posix()}')
        return client, True


def create_file_logger(path_config: Path, mode_name: str) -> logging.Logger:
    config_obj = toml.load(path_config)
    interface_config = create_interface_config(config_obj)
    resouce_config = interface_config.resource_config

    (resouce_config.path_work_dir / resouce_config.dir_name_log).mkdir(parents=True, exist_ok=True)
    path_log_dir = resouce_config.path_work_dir / resouce_config.dir_name_log / f"{mode_name}.log"

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


def filter_dataset_record(translation_config: TranslationConfig,
                          resource_config: ResourceConfig,
                          seq_dataset_records: ty.Sequence[ty.Union[WMTDatasetRecord, HalomiDatasetRecord]]
                          ) -> ty.Sequence[ty.Union[WMTDatasetRecord, HalomiDatasetRecord]]:
    dataset_name = classify_dataset_type(seq_dataset_records[0])
    root_logger.info(f'Before the filtering N(record) = {len(seq_dataset_records)}')

    # filtering by the hallucination label.
    if dataset_name == 'lfan_hall':
        if resource_config.filter_error_type is not None:
            if resource_config.filter_error_type == 'hallucination':
                seq_dataset_records = [_r for _r in seq_dataset_records if _r.error_type == 'hallucination']
            # end if
        # end if
    elif dataset_name == 'halomi':
        if resource_config.filter_error_type is not None:
            if resource_config.filter_error_type == 'hallucination':
                seq_dataset_records = [_r for _r in seq_dataset_records if _r.error_type == 'hallucination']
            # end if
        # end if
    # end if

    if translation_config.option_target_sentence_id is not None:
        if dataset_name == 'lfan_hall':
            seq_dataset_records = [_r for _r in seq_dataset_records if str(_r.sentence_id) in translation_config.option_target_sentence_id]
        elif dataset_name == 'halomi':
            seq_dataset_records = [_r for _r in seq_dataset_records if _r.key_unique in translation_config.option_target_sentence_id]
        else:
            raise ValueError()
        # end if
    # end if

    if resource_config.filter_exclude_sentence_id is not None:
        if dataset_name == 'lfan_hall':
            seq_dataset_records = [_r for _r in seq_dataset_records if str(_r.sentence_id) not in resource_config.filter_exclude_sentence_id]
        elif dataset_name == 'halomi':
            seq_dataset_records = [_r for _r in seq_dataset_records if _r.key_unique not in resource_config.filter_exclude_sentence_id]
        else:
            raise ValueError()
        # end if
    # end if

    assert len(seq_dataset_records) > 0
    root_logger.info(f"After the filtering N(records)={len(seq_dataset_records)}")
    return seq_dataset_records



def main(path_config_toml: Path):
    assert path_config_toml.exists(), f"File not found: {path_config_toml}"

    # loading the config file
    with open(path_config_toml, "r") as f:
        config_dict = toml.load(f)
    # end with

    interface_config = create_interface_config(config_dict)
    # check script version compatibility
    assert interface_config.script_version == SCRIPT_VERSION, f"Script version mismatch. Expected {SCRIPT_VERSION}, got {interface_config.script_version}"


    # loading the dataset tsv
    path_dataset_tsv = interface_config.resource_config.path_dataset_tsv
    if interface_config.resource_config.dataset_name == 'halomi':
        seq_dataset_records = loading_halomi(path_dataset_tsv)
    elif interface_config.resource_config.dataset_name == 'lfan_hall':
        seq_dataset_records = loading_lfan_hall(path_dataset_tsv, delimiter='\t')
    else:
        raise ValueError()
    # end if
    
    # setting the management db.
    path_management_db = interface_config.resource_config.path_work_dir / interface_config.resource_config.file_name_db_sqlite3
    db_connection = sqlite3.connect(path_management_db)
    create_table_from_table_definition(db_connection, DbTableRecordRaunak2021, primary_key="record_id")
    create_table_from_table_definition(db_connection, DbTableRecordProposalMmdFlaggerTrajectoryVer3, primary_key="record_id")
    create_table_from_table_definition(db_connection, DbTableRecordGuerreiro2023SeqLogProb, primary_key="record_id")
    create_table_from_table_definition(db_connection, DbTableRecordGuerreiro2023McDSIM, primary_key="record_id")
    db_handler = DBHandlerExp(path_management_db)

    # TODO: I need a function to route approach functions
    seq_approach_names = interface_config.flagging_approaches
    if interface_config.resource_config.path_dir_cache_translation is not None:
        path_cache_dir = interface_config.resource_config.path_dir_cache_translation
        root_logger.info(f"Cache directory: {path_cache_dir}")
    else:
        path_cache_dir = interface_config.resource_config.path_work_dir / interface_config.resource_config.dir_name_cache
        path_cache_dir.mkdir(parents=True, exist_ok=True)
    # end if

    for _approach_name in seq_approach_names:
        if _approach_name == "MmdErrorFlaggerTrajectoryVer3Config":
            algorithm_config: MmdErrorFlaggerTrajectoryVer3Config = interface_config.approach_configs["MmdErrorFlaggerTrajectoryVer3Config"]
            assert isinstance(algorithm_config, MmdErrorFlaggerTrajectoryVer3Config), f"Expected MmdErrorFlaggerTrajectoryVer3, got {algorithm_config}"
            
            _path_cache_translator = interface_config.translation_config.path_cache_translator
            assert _path_cache_translator.exists(), f"No translation handler cache directory found. You may set the wrong path. Hint: manually create the directory at {_path_cache_translator}"

            _path_cache_method_cache = path_cache_dir / _approach_name
            _path_cache_method_cache.mkdir(parents=True, exist_ok=True)

            proposal_mmd_flagging_trajectory_ver3(
                config_name=interface_config.config_name,
                dataset_name=interface_config.resource_config.dataset_name,
                llm_model_config=interface_config.llm_model_config,
                translation_config=interface_config.translation_config,
                resource_config=interface_config.resource_config,
                algorithm_config_obj=algorithm_config,
                seq_dataset_records=seq_dataset_records,
                management_db_handler=db_handler,
                path_flagger_cache_dir=path_cache_dir,
            )
        elif _approach_name == "Raunak2021ApproachConfig":
            raise NotImplementedError()
            # I want to skip records that are already flagged
            raunak_2021_config_obj = interface_config.approach_configs["Raunak2021ApproachConfig"]
            assert isinstance(raunak_2021_config_obj, Raunak2021ApproachConfig), f"Expected Raunak2021ApproachConfig, got {raunak_2021_config_obj}"
            oscillatory_detection_Raunak_2021_flagging(
                config_name=interface_config.config_name,
                raunak_2021_config_obj=raunak_2021_config_obj,
                seq_dataset_eval=seq_dataset_records,
                management_db_handler=db_handler,
                file_logger=root_logger,
                resource_config=interface_config.resource_config
            )
        elif _approach_name == "Guerreiro2023MCDSim":
            raise NotImplementedError()
            # TODO. I did not make the MC-DSIM for the transformer package.
            algorithm_config = interface_config.approach_configs["Guerreiro2023MCDSim"]
            assert isinstance(algorithm_config, Guerreiro2023MCDSim), f"Expected Guerreiro2023MCDSim, got {algorithm_config}"
            mcd_sim_Guerreiro_2023_flagging(
                config_name=interface_config.config_name,
                algorithm_config_obj=algorithm_config,
                seq_dataset_eval=seq_dataset_eval,
                management_db_handler=db_handler,
                file_logger=root_logger,
                path_cache_dir=path_cache_dir
            )
        elif _approach_name == "Guerreiro2023SeqLogprob":
            raise NotImplementedError()
            algorithm_config = interface_config.approach_configs["Guerreiro2023SeqLogprob"]
            assert isinstance(algorithm_config, Guerreiro2023SeqLogprob), f"Expected Guerreiro2023SeqLogprob, got {algorithm_config}"
            seq_log_prob_Guerreiro_2023_flagging(
                config_name=interface_config.config_name,
                algorithm_config_obj=algorithm_config,
                seq_dataset_eval=seq_dataset_records,
                management_db_handler=db_handler,
                file_logger=root_logger,
                path_cache_dir=path_cache_dir,
                resource_config=interface_config.resource_config
            )            
        else:
            raise Exception(f"Approach not implemented: {_approach_name}")
        # end if
    # end for

    interface_config_obj = interface_config.to_dict()
    with open(interface_config.resource_config.path_work_dir / "interface_config.toml", "w") as f:
        toml.dump(interface_config_obj, f)
    # end with

    db_connection.close()
    root_logger.info(f"-------- Script End --------")


def do_translation(path_config_toml: Path):
    assert path_config_toml.exists(), f"File not found: {path_config_toml}"

    # loading the config file
    with open(path_config_toml, "r") as f:
        config_dict = toml.load(f)
    # end with

    interface_config = create_interface_config(config_dict)

    path_cache_translation = interface_config.translation_config.path_cache_translator
    path_cache_translation.mkdir(parents=True, exist_ok=True)

    def _translation_procedure_halomi(seq_dataset: ty.Sequence[HalomiDatasetRecord]):
        path_subdir_halomi = path_cache_translation / 'halomi'
        path_subdir_halomi.mkdir(parents=True, exist_ok=True)

        seq_possible_language_pair = making_language_pair(resource_config=interface_config.resource_config, 
                                                          seq_dataset=seq_dataset)

        # making the pair of src, tgt lang.
        for __t_source_target_lang_code in tqdm.tqdm(seq_possible_language_pair, desc=f"Processing Halomi records", file=sys.stdout):
            _source_lang_code, _target_lang_code = __t_source_target_lang_code

            # _path_subdir_language_pair = path_subdir_halomi / f'{_source_lang_code}_{_target_lang_code}'
            # _path_subdir_language_pair.mkdir(parents=True, exist_ok=True)

            assert isinstance(interface_config.llm_model_config.model_halomi, ModelConfigHalomi)
            # setting the tokenizer and model.
            __vector_extractor = setup_tokenizer_and_model_halomi(
                source_lang_code=_source_lang_code,
                target_lang_code=_target_lang_code,
                model_name=interface_config.llm_model_config.model_halomi.nllb_model_name,
                path_dir_translation_cache=path_subdir_halomi,
                translation_config=interface_config.translation_config)

            # selecting the records.
            _seq_target_record = select_record_lang_code(seq_dataset_record=seq_dataset, 
                                                         src_lang_code=_source_lang_code, 
                                                         tgt_lang_code=_target_lang_code)

            # filtering target records.
            seq_dataset: ty.Sequence[HalomiDatasetRecord] = filter_dataset_record(interface_config.translation_config, interface_config.resource_config, seq_dataset)

            # halomi records -> `EvaluationTargetTranslationPair`
            _seq_dataset_eval_type = [
                EvaluationTargetTranslationPair(source=_r.src_text, target=_r.tgt_text, sentence_id=_r.key_unique) 
                for _r in _seq_target_record]
            
            _translation_handler = __vector_extractor.translation_handler
            
            _parameter_combination_stochastic = list(itertools.product(
                interface_config.translation_config.candidates_temperature_parameter,
                interface_config.translation_config.candidates_n_sampling_stochastic
            ))

            __, seq_decoder = _translation_handler.get_all_possible_layers()
            seq_decoder.append(_translation_handler._get_decoder_word_embedding_layer_name())

            for _record in _seq_dataset_eval_type:
                # Do the beam search translation
                _translation_handler.translate_beam_search(
                    input_text=_record,
                    temperature=interface_config.translation_config.temperature_beam,
                    max_len_a=interface_config.translation_config.max_len_a,
                    max_len_b=interface_config.translation_config.max_len_b,
                    target_layers_extraction=seq_decoder
                )
                root_logger.info(f'Beam translation done. Halomi. Record-id: {_record.sentence_id}')

                for _tau_param, _n_sampling in _parameter_combination_stochastic:
                    _translation_handler.translate_sample_multiple_times(
                        input_text=_record,
                        temperature=_tau_param,
                        n_sampling=_n_sampling,
                        max_len_a=interface_config.translation_config.max_len_a,
                        max_len_b=interface_config.translation_config.max_len_b,
                        target_layers_extraction=seq_decoder,
                        n_max_attempts=2,
                        batch_size=10)
                    root_logger.info(f'One Iteration done. Halomi. Record-id: {_record.sentence_id}, tau: {_tau_param}, n_sampling: {_n_sampling}')
                # end for
            # end for
        # end for
    # end def

    def _translation_procedure_lfan_hall(seq_dataset: ty.Sequence[WMTDatasetRecord], is_shuffle_order: bool = False):
        path_subdir_lfan_hall = path_cache_translation / 'lfan_hall'
        path_subdir_lfan_hall.mkdir(parents=True, exist_ok=True)

        assert isinstance(interface_config.llm_model_config.model_lfan_hall, ModelConfigLfanHall)
        # setting the tokenizer and model.
        __vector_extractor = setup_tokenizer_and_model_fairseq(
            model_config=interface_config.llm_model_config.model_lfan_hall,
            path_cache_dir=path_subdir_lfan_hall,
            translation_config=interface_config.translation_config)

        # filtering target records.
        seq_dataset: ty.Sequence[WMTDatasetRecord] = filter_dataset_record(interface_config.translation_config, interface_config.resource_config, seq_dataset)

        # lfan_hall records -> `EvaluationTargetTranslationPair`
        _seq_dataset_eval_type = [
            EvaluationTargetTranslationPair(source=_r.source, target=_r.translation, sentence_id=str(_r.sentence_id)) 
            for _r in seq_dataset]
        
        if is_shuffle_order:
            # Note: shuffling the order.
            # The Fairseq Translation handler is heavy IO process (due to modification of hidden state extraction).
            # Thus, I intend to launch multiple script at the same time. Random shuffling makes it possible.
            # gen_random = random.Random(interface_config.resource_config.seed_random_seed)
            gen_random = random.Random()  # I use the random seed.
            gen_random.shuffle(_seq_dataset_eval_type)  
        # end if
        
        _translation_handler = __vector_extractor.translation_handler

        _parameter_combination_stochastic = list(itertools.product(
            interface_config.translation_config.candidates_temperature_parameter,
            interface_config.translation_config.candidates_n_sampling_stochastic
        ))

        __, seq_decoder = _translation_handler.get_all_possible_layers()
        seq_decoder.append(_translation_handler._get_decoder_word_embedding_layer_name())

        for _record in tqdm.tqdm(_seq_dataset_eval_type, desc=f"Processing Lfan-Hall records", file=sys.stdout):
            # Do the beam search translation
            _is_exist_beam = _translation_handler._is_exist_cache(_record.sentence_id, interface_config.translation_config.temperature_beam, n_sampling=None)
            if _is_exist_beam is None:
                try:
                    _translation_handler.translate_beam_search(
                        input_text=_record,
                        temperature=interface_config.translation_config.temperature_beam,
                        max_len_a=interface_config.translation_config.max_len_a,
                        max_len_b=interface_config.translation_config.max_len_b,
                        target_layers_extraction=seq_decoder
                    )
                    root_logger.info(f'Beam translation done. Lfan-Hall. Record-id: {_record.sentence_id}')
                except AssertionError as e:
                    root_logger.error(f'Assertion Error by the length setting. e={e}')
                # end try
            # end if

            for _tau_param, _n_sampling in _parameter_combination_stochastic:
                _is_exist_sampling = _translation_handler._is_exist_cache(_record.sentence_id, _tau_param, n_sampling=_n_sampling)
                if _is_exist_sampling is None:
                    try:
                        _translation_handler.translate_sample_multiple_times(
                            input_text=_record,
                            temperature=_tau_param,
                            n_sampling=_n_sampling,
                            max_len_a=interface_config.translation_config.max_len_a,
                            max_len_b=interface_config.translation_config.max_len_b,
                            target_layers_extraction=seq_decoder,
                            n_max_attempts=5,
                            batch_size=interface_config.llm_model_config.model_lfan_hall.batch_size)
                        root_logger.info(f'One Iteration done. Lfan-Hall. Record-id: {_record.sentence_id}, tau: {_tau_param}, n_sampling: {_n_sampling}')
                    except AssertionError as e:
                        root_logger.error(f'Assertion Error by the length setting. Error message={e}')
                    # end try
                    except ParameterSettingException as e:
                        root_logger.error(f'Assertion Error. Nothing we can do. Error message={e}')                        
                # end if
            # end for
        # end for
    # end def

    path_dataset_tsv = interface_config.resource_config.path_dataset_tsv
    if interface_config.resource_config.dataset_name == 'halomi':
        _seq_dataset_records = loading_halomi(path_dataset_tsv)
        _translation_procedure_halomi(_seq_dataset_records)
    elif interface_config.resource_config.dataset_name == 'lfan_hall':
        _seq_dataset_records = loading_lfan_hall(path_dataset_tsv, delimiter='\t')
        _translation_procedure_lfan_hall(_seq_dataset_records)
    else:
        raise ValueError()        
    # end if


def evaluate_main(path_config_toml: Path):
    assert path_config_toml.exists(), f"File not found: {path_config_toml}"

    # loading the config file
    with open(path_config_toml, "r") as f:
        config_dict = toml.load(f)
    # end with
    interface_config = create_interface_config(config_dict)
    # check script version compatibility
    assert interface_config.script_version == SCRIPT_VERSION, f"Script version mismatch. Expected {SCRIPT_VERSION}, got {interface_config.script_version}"

    path_output_dir = interface_config.resource_config.path_work_dir / interface_config.evaluation_config.dir_name_output
    path_output_dir.mkdir(parents=True, exist_ok=True)

    if interface_config.resource_config.dataset_name == 'halomi':
        seq_dataset_record = loading_halomi(interface_config.resource_config.path_dataset_tsv)
    elif interface_config.resource_config.dataset_name == 'lfan_hall':
        seq_dataset_record = loading_lfan_hall(interface_config.resource_config.path_dataset_tsv, delimiter='\t')
    else:
        raise ValueError()
    # end if

    # db file
    path_prediction_database = interface_config.resource_config.path_work_dir / interface_config.resource_config.file_name_db_sqlite3
    assert path_prediction_database.exists(), f"File not found: {path_prediction_database}"

    eval_runner = evaluation_script_ver3.EvaluationVer3(
        seq_dataset_record=seq_dataset_record,
        path_prediction_database=path_prediction_database,
        dataset_type=interface_config.resource_config.dataset_name)
    
    seq_eval_table_name = [
        # DbTableRecordRaunak2021.__name__,
        DbTableRecordProposalMmdFlaggerTrajectoryVer3.__name__,
        # DbTableRecordGuerreiro2023McDSIM.__name__,
        # DbTableRecordGuerreiro2023SeqLogProb.__name__
    ]
    if interface_config.resource_config.path_dir_cache_translation is None:
        path_dir_cache_translation = interface_config.translation_config.path_cache_translator
    else:
        path_dir_cache_translation = interface_config.resource_config.path_dir_cache_translation
    # end if
    assert path_dir_cache_translation is not None

    eval_runner.main(
        path_output_dir=path_output_dir,
        config_name=interface_config.config_name, 
        seq_eval_table_name=seq_eval_table_name,
        path_dir_cache_translation=path_dir_cache_translation
    )

    # result analysis runner
    path_analysis = interface_config.resource_config.path_work_dir / interface_config.evaluation_config.dir_name_analysis
    path_analysis.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("-c", "--config", required=True, type=str, help="Path to the config file.")
    _parser.add_argument("-m", "--mode", required=True, type=str, help="Mode: flagging or evaluation", choices=["translation", "flagging", "flag", "evaluation", "eval"])
    _args = _parser.parse_args()

    _path_config_toml = Path(_args.config)

    # global variable definitions
    root_logger = create_file_logger(_path_config_toml, mode_name=_args.mode)
    slack_client, is_use_slack = setup_slack_wekhook(_path_config_toml, mode_name=_args.mode)

    if _args.mode == "flagging" or _args.mode == "flag":
        main(_path_config_toml)
    elif _args.mode == "translation":
        do_translation(_path_config_toml)
    elif _args.mode == "evaluation" or _args.mode == "eval":
        path_config_toml = Path(_args.config)
        evaluate_main(path_config_toml)
    else:
        raise Exception(f"Mode not implemented: {_args.mode}")
    # end if

    if is_use_slack:
        assert slack_client is not None
        slack_notifier.send_message(
            webhook=slack_client,
            message=f"flagging_interface.py done successfully. mode={_args.mode}")
    # end if