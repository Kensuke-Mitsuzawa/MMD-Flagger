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

import numpy as np

from pathlib import Path

import dataclasses
from dataclasses import dataclass, asdict

# dataset module
from hallucination_mt.dale_2023_halomi.load_dataset import (
    HalomiDatasetRecord,
    load_dataset
)
# management db module
from hallucination_mt.module_assessments.module_management_db.module_db_record import (
    DbTableRecordRaunak2021,
    DbTableRecordGuerreiro2023SeqLogProb,
    DbTableRecordProposalMmdFlaggerVer1,
    DbTableRecordProposalMmdFlaggerTrajectoryVer2,
    DbTableRecordGuerreiro2023McDSIM
)
from hallucination_mt.module_assessments.module_management_db.module_sqlite3_handler import (
    create_table_from_table_definition,
    DBHandlerExp
)
# helper
from hallucination_mt.module_assessments.custom_tqdm_handler import TqdmLoggingHandler
# approach modules
from hallucination_mt.commons.data_models import EvaluationTargetTranslationPair
from hallucination_mt.baselines.raunak_2021 import oscillatory_detection_Raunak_2021
from hallucination_mt.baselines.seq_log_probability import (
    TransformerFlaggerSeqLogProbability,
    OutputLogProbabilityFlagger)
from hallucination_mt.baselines.mc_dropout.fairseq_handler.flagger_mc_dropout import FlaggerDisSimilarityMcDropOut, OutputDisSimilarityMcDropOut
from hallucination_mt.baselines.mc_dropout.fairseq_handler import flagger_mc_dropout
from hallucination_mt.module_flagging import mmd_error_flagger_ver1
from hallucination_mt.module_flagging.utils import load_fairseq_model
from hallucination_mt.module_flagging import mmd_error_flagger_trajectory_ver2
# fairseq handler
from hallucination_mt.exceptions import ParameterSettingException
# translation module
from hallucination_mt.module_translation_handler.ver1.module_transformer_handler import (
    TransformersTranslationModelHandler, 
    GeneratedTranslationObject)
from hallucination_mt.module_hidden_vector_extractor.ver1 import TransformerVectorExtractor
# evaluation module
from hallucination_mt.module_assessments.module_evaluation import evaluation_script_ver2
# from hallucination_mt.module_assessments.module_evaluation import result_analysis_ver1 


# method configs
from method_configs import (
    MmdErrorFlaggerTrajectoryVer2,
    MmdErrorFlaggerVer1,
    Raunak2021ApproachConfig,
    Guerreiro2023MCDSim,
    Guerreiro2023SeqLogprob
)


logging.basicConfig(level=logging.DEBUG)
# a special logger for tqdm
tqdm_logger = logging.getLogger('tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())

SCRIPT_VERSION = "0.1"

# --------------------------------------------------------------

@dataclass
class ResourceConfig:
    path_work_dir: Path
    path_dataset_tsv: Path

    dataset_name: str

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


    def __post_init__(self):
        assert self.path_dataset_tsv.exists(), f"File not found: {self.path_dataset_tsv}"
        if self.path_dir_cache_translation is not None:
            assert self.path_dir_cache_translation.exists(), f"Directory not found: {self.path_dir_cache_translation}"
        # end if
        
        if self.filter_src_tgt_lang_code is not None:
            self.filter_src_tgt_lang_code = [tuple(_seq_list) for _seq_list in self.filter_src_tgt_lang_code]
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
class InterfaceConfig:
    script_version: str
    config_name: str  # recommended to set the unique name.
    resource_config: ResourceConfig
    evaluation_config: EvaluationConfig
    flagging_approaches: ty.List[str]
    approach_configs: ty.Dict[str, ty.Any]

    def __post_init__(self):
        # Convert the approach configs to the appropriate dataclass
        approach_configs = {}

        assert len(self.approach_configs) > 0, "At least one approach config should be provided"

        # collecting the config objects for each approach
        for _key, _value in self.approach_configs.items():
            assert isinstance(_value, dict), f"Expected dict, got {_value}"
            if _key == "Raunak2021ApproachConfig":
                approach_configs[_key] = dacite.from_dict(Raunak2021ApproachConfig, _value)
            elif _key == "MmdErrorFlaggerVer1":
                approach_configs[_key] = dacite.from_dict(MmdErrorFlaggerVer1, _value)
            elif _key == "MmdErrorFlaggerTrajectoryVer2":
                approach_configs[_key] = dacite.from_dict(MmdErrorFlaggerTrajectoryVer2, _value)
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


def _select_non_calibration_record(seq_calibration_record: ty.List[HalomiDatasetRecord],
                                   seq_dataset_record: ty.List[HalomiDatasetRecord]) -> ty.List[HalomiDatasetRecord]:
    seq_calibration_text_id = [__r.key_unique for __r in seq_calibration_record]
    seq_non_calibration_record = [__record for __record in seq_dataset_record if __record.key_unique not in seq_calibration_text_id]
    return seq_non_calibration_record
# end def


def _select_calibration_text(seq_dataset_record: ty.List[HalomiDatasetRecord],
                             n_calibration_records: int,
                             method_select_calibration: str = 'random',
                             seed_random_seed: int = 42,
                             ) -> ty.List[HalomiDatasetRecord]:
    """I select the calibration text from the dataset."""
    seq_correct_translation = [__record for __record in seq_dataset_record if __record.is_hallucination is False and __record.is_omission is False]
    assert len(seq_correct_translation) > 0, f"len(seq_correct_translation)={len(seq_correct_translation)}"
    
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
    return _selected_records
# end def


def select_record_lang_code(
        seq_dataset_record: ty.List[HalomiDatasetRecord],
        src_lang_code: str,
        tgt_lang_code: str) -> ty.List[HalomiDatasetRecord]:
    """I select the records from the dataset based on the source and target language code."""
    seq_selected_records = [__record for __record in seq_dataset_record if __record.src_lang == src_lang_code and __record.tgt_lang == tgt_lang_code]
    assert len(seq_selected_records) > 0, f"len(seq_selected_records)={len(seq_selected_records)}"
    return seq_selected_records


def making_language_pair(resource_config: ResourceConfig,
                         seq_dataset: ty.List[HalomiDatasetRecord]
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



def setup_tokenizer_and_model(source_lang_code: str,
                              target_lang_code,
                              embedding_layer: str,
                              model_name: str,
                              max_len_a: float,
                              max_len_b: int,
                              ) -> ty.Tuple[TransformersTranslationModelHandler, TransformerVectorExtractor]:
    translation_handler = TransformersTranslationModelHandler(
        src_lang=source_lang_code,
        target_lang=target_lang_code,
        model_name=model_name,
        max_len_a=max_len_a,
        max_len_b=max_len_b
    )
    vector_extractor = TransformerVectorExtractor(translation_handler)

    return translation_handler, vector_extractor


# --------------------------------------------------------------


def proposal_mmd_flagging_trajectory_ver2(
        config_name: str,
        resource_config: ResourceConfig,
        algorithm_config_obj: MmdErrorFlaggerTrajectoryVer2,
        seq_dataset: ty.List[HalomiDatasetRecord],
        management_db_handler: DBHandlerExp,
        file_logger: logging.Logger,
        path_cache_dir: Path):

    def _get_record_processed(seq_dataset_reocrd: ty.List[HalomiDatasetRecord], 
                              db_handler: DBHandlerExp,
                              n_sample: int,
                              tau_sequence: ty.List[float],
                              embedding_option: str,
                              trajectory_rule: str,
                              trajectory_rule_smoothing: str,
                              trajectory_rule_smoothing_window: ty.Optional[int],
                              ) -> ty.List[mmd_error_flagger_trajectory_ver2.EvaluationTargetTranslationPair]:
        seq_record_id_existing = db_handler.get_all_keys(
            table_name=DbTableRecordProposalMmdFlaggerTrajectoryVer2.__name__,
            primary_key_field='record_id')
        
        seq_record_processing = []
        for __record in seq_dataset_reocrd:
            _db_record_id = DbTableRecordProposalMmdFlaggerTrajectoryVer2.get_record_id(
                config_name=config_name,
                approach_name=algorithm_config_obj.approach_name,
                n_sampling=n_sample,
                embedding_option=embedding_option,
                tau_sequence=tau_sequence,
                _sentence_id=str(__record.key_unique),
                trajectory_rule=trajectory_rule,
                trajectory_rule_smoothing=trajectory_rule_smoothing,
                trajectory_rule_smoothing_window=trajectory_rule_smoothing_window)
            
            if _db_record_id in seq_record_id_existing:
                continue
            else:
                _record_eval_input = mmd_error_flagger_trajectory_ver2.EvaluationTargetTranslationPair(
                    target=__record.tgt_text,
                    source=__record.src_text,
                    sentence_id=str(__record.key_unique))
                seq_record_processing.append(_record_eval_input)
            # end if
        # end for

        return seq_record_processing
    # end def


    def _execute_flagging(seq_dataset_reocrd: ty.List[HalomiDatasetRecord], 
                          seq_calibration_text: ty.List[str],
                          translation_handler: TransformersTranslationModelHandler,
                          vector_extractor: TransformerVectorExtractor,
                          source_language_code: str,
                          target_language_code: str,
                          n_sampling: int,
                          tau_sequence: ty.List[float],
                          embedding_layer: str,
                          trajectory_rule: str,
                          trajectory_rule_smoothing: str,
                          trajectory_rule_smoothing_window: ty.Optional[int],
                          db_handler: DBHandlerExp,
                          ):
        assert len(tau_sequence) > 0, "tau_sequence should not be empty."
        seq_processing_records = _get_record_processed(
            seq_dataset_reocrd=seq_dataset_reocrd,
            n_sample=n_sampling,
            tau_sequence=tau_sequence,
            embedding_option=embedding_layer,
            db_handler=db_handler,
            trajectory_rule=trajectory_rule,
            trajectory_rule_smoothing=trajectory_rule_smoothing,
            trajectory_rule_smoothing_window=trajectory_rule_smoothing_window)
        
        if len(seq_processing_records) == 0:
            file_logger.info(f"All records already all processed.")
            return
        # end if

        # set the flagger object
        mmd_flagger = mmd_error_flagger_trajectory_ver2.MmdErrorFlaggerTrajectoryVer2(
            translation_handler=translation_handler,
            vector_extractor=vector_extractor,
            seq_calibration_text=seq_calibration_text,
            path_cache_dir=path_cache_dir,
            trajectory_rule=trajectory_rule,
            trajectory_rule_smoothing=trajectory_rule_smoothing,
            trajectory_rule_smoothing_window=trajectory_rule_smoothing_window)
        
        _dataset_record: mmd_error_flagger_trajectory_ver2.EvaluationTargetTranslationPair
        for _dataset_record in tqdm.tqdm(seq_processing_records, 
                                         desc=f"Processing {algorithm_config_obj.approach_name}", 
                                         file=sys.stdout):
            # execution.
            file_logger.debug(f"Processing sentence_id: {_dataset_record.sentence_id}")
            try:
                _flag_result = mmd_flagger.flag_hallucination_one_record(
                    eval_target=_dataset_record,
                    candidate_temperature_parameters=np.array(tau_sequence, dtype=np.float32),
                    n_sampling=n_sampling)
                file_logger.debug(f"Done: {_dataset_record.sentence_id}")
            except (AssertionError, ValueError) as e:
                # I assume an AssertionError involving trajectory rule.
                # TODO: I refine the try-except block: e.g. putting a specific Exception Class.
                file_logger.error(f"Encountering an AssertionError. Skip this execution: {e}")
                continue
            else:
                # saving into the db.
                _arg_obj = asdict(_flag_result)
                # I delete unnecessary fields (torch Tensor objects)
                del _arg_obj["tensor_given_translation"]
                del _arg_obj["tensor_hypothesis_translation"]

                file_logger.debug(f"Saving the result into the db.")

                record_ = DbTableRecordProposalMmdFlaggerTrajectoryVer2(
                    config_name=config_name,
                    dataset_name=resource_config.dataset_name,
                    sentence_id=str(_dataset_record.sentence_id),
                    source_language_code=source_language_code,
                    target_language_code=target_language_code,
                    n_sampling=n_sampling,
                    approach_name=algorithm_config_obj.approach_name,
                    tau_sequence=json.dumps(tau_sequence),
                    embedding_option=embedding_layer,
                    flagging_label=_flag_result.is_hallucination,
                    flagging_argument_json=json.dumps(_arg_obj),
                    trajectory_rule=trajectory_rule,
                    trajectory_rule_smoothing=trajectory_rule_smoothing,
                    trajectory_rule_smoothing_window=trajectory_rule_smoothing_window,
                    trajectory_rule_options_json=json.dumps({}))
                record_dict = asdict(record_)
                db_handler.insert(table_name=DbTableRecordProposalMmdFlaggerTrajectoryVer2.__name__, data=record_dict)
                file_logger.debug(f"Done saving the result into the db.")
            # end try
        # end for
    # end def
    
    # making the cache directory.
    path_cache_dir = path_cache_dir / mmd_error_flagger_trajectory_ver2.__name__
    path_cache_dir.mkdir(parents=True, exist_ok=True)

    # set option parameters
    option_n_sampling: ty.List[int] = algorithm_config_obj.option_n_translation_sampling
    option_temperature_sequence: ty.List = algorithm_config_obj.option_temperature_sequence
    option_embedding_layer: ty.List[str] = algorithm_config_obj.option_embedding_layer

    seq_possible_language_pair = making_language_pair(resource_config=resource_config, seq_dataset=seq_dataset)
    # # filtering records by the option of `filter_src_tgt_lang_code`.
    # if resource_config.filter_src_tgt_lang_code is not None:
    #     __set_conditions_src_and_tgt: ty.Set[ty.Tuple[str, str]] = set([tuple(_l) for _l in resource_config.filter_src_tgt_lang_code])  # type: ignore
    #     # filtering the dataset records.
    #     seq_dataset = [__r for __r in seq_dataset if (__r.src_lang, __r.tgt_lang) in __set_conditions_src_and_tgt]
    #     assert len(seq_dataset) > 0, f"len(seq_dataset)={len(seq_dataset)}"
    #     seq_possible_language_pair = [tuple(_l) for _l in resource_config.filter_src_tgt_lang_code]
    # else:
    #     # making the source and target language code.
    #     seq_possible_language_pair = set([(__r.src_lang, __r.tgt_lang) for __r in seq_dataset])
    # # end if

    # parameter combinations
    option_combinations = itertools.product(
        option_n_sampling, 
        option_temperature_sequence, 
        option_embedding_layer,
        algorithm_config_obj.option_trajectory_rule_versions,
        algorithm_config_obj.option_trajectory_rule_smoothing,
        algorithm_config_obj.option_trajectory_rule_smoothing_window)
    
    for _n_sampling, tau_sequence, _embedding_layer, _trajectroy_rule, _trajectory_rule_smoothing, trajectory_rule_smoothing_window in option_combinations:
        # updating `trajectory_rule_smoothing_window` when `_trajectory_rule_smoothing` is no_filter
        if _trajectory_rule_smoothing == "no_filter":
            trajectory_rule_smoothing_window = None
        # end if

        # making the pair of src, tgt lang.
        for __t_source_target_lang_code in seq_possible_language_pair:
            _source_lang_code, _target_lang_code = __t_source_target_lang_code

            # setting the tokenizer and model.
            __translation_module, __vector_extractor = setup_tokenizer_and_model(
                source_lang_code=_source_lang_code,
                target_lang_code=_target_lang_code,
                embedding_layer=_embedding_layer,
                model_name=algorithm_config_obj.nllb_model_name,
                max_len_a=algorithm_config_obj.max_len_a,
                max_len_b=algorithm_config_obj.max_len_b)

            # selecting the records.
            _seq_target_record = select_record_lang_code(seq_dataset_record=seq_dataset, 
                                                         src_lang_code=_source_lang_code, 
                                                         tgt_lang_code=_target_lang_code)

            # selecting the calibration text.
            _seq_calibration_record = _select_calibration_text(
                seq_dataset_record=_seq_target_record,
                n_calibration_records=resource_config.n_calibration_records,
                method_select_calibration=resource_config.method_select_calibration,
                seed_random_seed=resource_config.seed_random_seed)
            
            # Note: I updated the code, 2025-04-23. I do not filter on Halomi dataset. Regardless of the calibration text, I decided using all for the assessment.
            # # selecting the non-calibration records.
            # _seq_dataset_eval = _select_non_calibration_record(
            #     seq_calibration_record=_seq_calibration_record,
            #     seq_dataset_record=_seq_target_record)
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

            _execute_flagging(
                seq_dataset_reocrd=_seq_dataset_eval,
                seq_calibration_text=[__r.tgt_text for __r in _seq_calibration_record],
                source_language_code=_source_lang_code,
                target_language_code=_target_lang_code,
                translation_handler=__translation_module,
                vector_extractor=__vector_extractor,
                n_sampling=_n_sampling,
                tau_sequence=tau_sequence,
                embedding_layer=_embedding_layer,
                db_handler=management_db_handler,
                trajectory_rule=_trajectroy_rule,
                trajectory_rule_smoothing=_trajectory_rule_smoothing,
                trajectory_rule_smoothing_window=trajectory_rule_smoothing_window,)
        # end for
    # end for




def proposal_mmd_flagging_ver1(
        config_name: str,
        mmd_flagger_ver1_obj: MmdErrorFlaggerVer1,
        seq_dataset_eval: ty.List[HalomiDatasetRecord],
        seq_calibration_record: ty.List[HalomiDatasetRecord],
        management_db_handler: DBHandlerExp,
        file_logger: logging.Logger,
        path_cache_dir: Path):
    
    def _get_record_processing(seq_dataset_reocrd: ty.List[HalomiDatasetRecord], 
                               temperature_low: float,
                               temperature_high: float,
                               db_handler: DBHandlerExp) -> ty.List[mmd_error_flagger_ver1.EvaluationTargetTranslationPair]:
        seq_record_id_existing = db_handler.get_all_keys(
            table_name=DbTableRecordProposalMmdFlaggerVer1.__name__,
            primary_key_field='record_id')
        
        seq_record_processing = []
        for __record in seq_dataset_reocrd:
            _db_record_id = DbTableRecordProposalMmdFlaggerVer1.get_record_id(
                config_name=config_name,
                approach_name=mmd_flagger_ver1_obj.approach_name,
                temperature_low=temperature_low,
                temperature_high=temperature_high,
                _sentence_id=str(__record.sentence_id))
            if _db_record_id in seq_record_id_existing:
                continue
            else:
                _record_eval_input = mmd_error_flagger_ver1.EvaluationTargetTranslationPair(
                    target=__record.translation,
                    source=__record.source,
                    sentence_id=str(__record.sentence_id))
                seq_record_processing.append(_record_eval_input)
            # end if
        # end for

        return seq_record_processing
    # end def
    
    def _execute_flagging(seq_dataset_reocrd: ty.List[HalomiDatasetRecord], 
                          seq_calibration_text: ty.List[str],
                          temperature_low: float,
                          temperature_high: float,
                          db_handler: DBHandlerExp):
        seq_processing_records = _get_record_processing(seq_dataset_reocrd=seq_dataset_reocrd,
                               temperature_low=temperature_low,
                               temperature_high=temperature_high,
                               db_handler=db_handler)
        if len(seq_processing_records) == 0:
            file_logger.info(f"Records already all processed for temperature_low: {temperature_low}, temperature_high: {temperature_high}")
            return
        # end if

        # set the flagger object
        mmd_flagger = mmd_error_flagger_ver1.MmdErrorFlaggerVer1(
            model_encoder_decoder_mt=model_encoder_decoder_mt,
            n_sampling=mmd_flagger_ver1_obj.n_translation_sampling,
            temperature_low=temperature_low,
            temperature_high=temperature_high,
            seq_calibration_text=seq_calibration_text,
            path_cache_dir=path_cache_dir,
            max_len_a=mmd_flagger_ver1_obj.max_len_a,
            max_len_b=mmd_flagger_ver1_obj.max_len_b)

        file_logger.info(f"{len(seq_processing_records)} Records to be processed for temperature_low: {temperature_low}, temperature_high: {temperature_high}")
        for _dataset_record in tqdm.tqdm(seq_processing_records, desc=f"Processing {mmd_flagger_ver1_obj.approach_name}", file=sys.stdout):            
            # execution.
            try:
                file_logger.debug(f"Processing sentence_id: {_dataset_record.sentence_id}")
                flag_result = mmd_flagger._flag_hallucination_one_record(_dataset_record)
                file_logger.debug(f"Done: {_dataset_record.sentence_id}")
            except ParameterSettingException as e:
                file_logger.error(f"Error: {e}")
                _arg_obj = dict(message="Error in the temperature parameter setting.")
                record_ = DbTableRecordProposalMmdFlaggerVer1(
                    config_name=config_name,
                    sentence_id=str(_dataset_record.sentence_id),
                    approach_name=mmd_flagger_ver1_obj.approach_name,
                    temperature_low=temperature_low,
                    temperature_high=temperature_high,
                    flagging_label=None,
                    flagging_argument_json=json.dumps(_arg_obj),
                    is_success=False)
            else:        
                # saving into the db.
                _arg_obj = flag_result._asdict()
                # I delete unnecessary fields
                del _arg_obj["translation_population_a"]
                del _arg_obj["translation_population_b"]
                del _arg_obj["tensor_word_embedding_population_a"]
                del _arg_obj["tensor_word_embedding_population_b"]
                del _arg_obj["embedding_word_target"]

                file_logger.debug(f"Saving the result into the db.")
                record_ = DbTableRecordProposalMmdFlaggerVer1(
                    config_name=config_name,
                    sentence_id=str(_dataset_record.sentence_id),
                    approach_name=mmd_flagger_ver1_obj.approach_name,
                    temperature_low=temperature_low,
                    temperature_high=temperature_high,
                    flagging_label=flag_result.is_hallucination,
                    flagging_argument_json=json.dumps(_arg_obj),
                    is_success=True)
            # end if
            record_dict = asdict(record_)
            db_handler.insert(table_name=DbTableRecordProposalMmdFlaggerVer1.__name__, data=record_dict)
            file_logger.debug(f"Done saving the result into the db.")
        # end for
    # end def
    
    model_encoder_decoder_mt = load_fairseq_model(
        path_fairseq_model_dir=mmd_flagger_ver1_obj.path_fairseq_model_dir,
        path_fairseq_model_file=mmd_flagger_ver1_obj.path_fairseq_model_file,
        path_sentencepiece_model=mmd_flagger_ver1_obj.path_sentencepiece_model)
    
    # making the cache directory.
    path_cache_dir = path_cache_dir / mmd_error_flagger_ver1.__name__
    path_cache_dir.mkdir(parents=True, exist_ok=True)

    seq_calibration_text = [__r.translation for __r in seq_calibration_record]

    # TODO: consider paralle. processing
    # exec. grid execution
    seq_temp_low = mmd_flagger_ver1_obj.set_temperature_low
    seq_temp_high = mmd_flagger_ver1_obj.set_temperature_high

    temp_combinations = itertools.product(seq_temp_low, seq_temp_high)
    for __t_low, __t_high in temp_combinations:
        _execute_flagging(
            seq_dataset_reocrd=seq_dataset_eval,
            seq_calibration_text=seq_calibration_text,
            temperature_low=__t_low,
            temperature_high=__t_high,
            db_handler=management_db_handler)
    # end for



def oscillatory_detection_Raunak_2021_flagging(
        config_name: str,
        resource_config: ResourceConfig,
        raunak_2021_config_obj: Raunak2021ApproachConfig,
        seq_dataset_eval: ty.List[HalomiDatasetRecord],
        management_db_handler: DBHandlerExp,
        file_logger: logging.Logger):
    module_script_path = oscillatory_detection_Raunak_2021.__file__
    path_python37_exec = raunak_2021_config_obj.path_python37_exec

    assert Path(module_script_path).exists(), f"Module script not found at {module_script_path}"

    _table_name = DbTableRecordRaunak2021.__name__

    def execute_flagging(source: str, output: str) -> ty.Dict:
        args_json = json.dumps(
            {"source": source, 
             "output": output,
             "ngram_size": raunak_2021_config_obj.ngram_size,
             "count_threshold": raunak_2021_config_obj.count_threshold,
             "difference_threshold": raunak_2021_config_obj.difference_threshold,
             "min_length_threshold": raunak_2021_config_obj.min_length_threshold             
        })
        result = subprocess.run(
            [
                path_python37_exec, 
                module_script_path,
                "--args",
                args_json
            ],
            capture_output=True,
            text=True
        )
        try:
            return_obj = json.loads(result.stdout)  # Assuming Project B outputs JSON
            return return_obj
        except json.JSONDecodeError:
            return {"status": "error", "error": result.stderr}
    # end def

    # I create the combination of src-target language.
    seq_language_pair = making_language_pair(resource_config=resource_config, seq_dataset=seq_dataset_eval)


    # main loop
    # For each src-target language pair, I compute the log-probability.
    for _tuple_lang_pair in seq_language_pair:
        _src_lang_code, _tgt_lang_code = _tuple_lang_pair

        # I select records having the src-target language.
        # selecting the records.
        _seq_target_record = select_record_lang_code(seq_dataset_record=seq_dataset_eval, 
                                                        src_lang_code=_src_lang_code, 
                                                        tgt_lang_code=_tgt_lang_code)
        assert len(_seq_target_record) > 0, f"len(_seq_target_record)={len(_seq_target_record)}"

        for _record in tqdm.tqdm(_seq_target_record, desc=f"Processing {raunak_2021_config_obj.approach_name}", file=sys.stdout):
        # for _record in seq_dataset_record:
            _sentence_id = _record.key_unique
            _db_record_id = DbTableRecordRaunak2021.get_record_id(config_name, raunak_2021_config_obj.approach_name, str(_sentence_id))

            _is_exist = management_db_handler.is_record_exists(
                table_name=_table_name,
                exp_key=_db_record_id,
                is_partially_search=False,
                primary_key='record_id')

            if _is_exist:
                continue
            # end if

            source = _record.src_text
            output = _record.tgt_text

            result = execute_flagging(source, output)
            assert 'status' in result, f"Expected 'status' in the result, got {result}"
            if result['status'] == 'success':
                record_ = DbTableRecordRaunak2021(
                    config_name=config_name,
                    approach_name=raunak_2021_config_obj.approach_name,
                    sentence_id=str(_sentence_id),
                    flagging_label=result['result'],
                    flagging_argument_json=json.dumps(result)
                )
                record_dict = asdict(record_)
                management_db_handler.insert(table_name=DbTableRecordRaunak2021.__name__, 
                                            data=record_dict)
            else:
                file_logger.error(f"{raunak_2021_config_obj.approach_name} failed for sentence_id: {_sentence_id}")
            # end if
    # end for


def seq_log_prob_Guerreiro_2023_flagging(
        config_name: str,
        resource_config: ResourceConfig,
        algorithm_config_obj: Guerreiro2023SeqLogprob,
        seq_dataset_eval: ty.List[HalomiDatasetRecord],
        management_db_handler: DBHandlerExp,
        file_logger: logging.Logger,
        path_cache_dir: Path):
    # I set the cache directory.
    # I create the combination of src-target language.
    seq_language_pair = making_language_pair(resource_config=resource_config, seq_dataset=seq_dataset_eval)

    # For each src-target language pair, I compute the log-probability.
    for _tuple_lang_pair in seq_language_pair:
        _src_lang_code, _tgt_lang_code = _tuple_lang_pair

        path_cache_dir_algorithm = path_cache_dir / Guerreiro2023SeqLogprob.__name__ / f"{_src_lang_code}_{_tgt_lang_code}"
        path_cache_dir_algorithm.mkdir(parents=True, exist_ok=True)
        
        # setting the tokenizer and model.
        __translation_module, __vector_extractor = setup_tokenizer_and_model(
            source_lang_code=_src_lang_code,
            target_lang_code=_tgt_lang_code,
            embedding_layer="decoder.embed_tokens",  # Note: I may change this later. But, I fix it for now.
            model_name=algorithm_config_obj.nllb_model_name,
            max_len_a=algorithm_config_obj.max_len_a,
            max_len_b=algorithm_config_obj.max_len_b)

        # I select records having the src-target language.
        # selecting the records.
        _seq_target_record = select_record_lang_code(seq_dataset_record=seq_dataset_eval, 
                                                        src_lang_code=_src_lang_code, 
                                                        tgt_lang_code=_tgt_lang_code)
        assert len(_seq_target_record) > 0, f"len(_seq_target_record)={len(_seq_target_record)}"

        seq_eval_records = [
            EvaluationTargetTranslationPair(
                source=__record.src_text, 
                target=__record.tgt_text, 
                sentence_id=str(__record.key_unique))
                for __record in _seq_target_record]
        assert len(seq_eval_records) > 0, f"len(seq_eval_records)={len(seq_eval_records)}"

        # I initialise the flagger
        __flagger = TransformerFlaggerSeqLogProbability(
            translation_handler=__translation_module,
            path_dir_cache=path_cache_dir_algorithm,)

        # I set the threshold for flagging.
        _seq_log_prob = __flagger.compute_dataset_log_probability(seq_eval_records,)
        threshold_flag = __flagger.get_flag_threshold(
            seq_log_probability=_seq_log_prob, 
            percentile=algorithm_config_obj.percentile_threshold)

        # I execute the flagging.
        for _record in tqdm.tqdm(seq_eval_records, desc=f"Processing {algorithm_config_obj.approach_name}", file=sys.stdout):
            # I check the result DB. Skip if the record is already processed.
            _exp_primary_key = DbTableRecordGuerreiro2023SeqLogProb.get_record_id(
                config_name=config_name,
                _sentence_id=str(_record.sentence_id))
            _is_exists = management_db_handler.is_record_exists(
                table_name=DbTableRecordGuerreiro2023SeqLogProb.__name__,
                exp_key=_exp_primary_key,
                is_partially_search=False,
                primary_key='record_id')
            if _is_exists:
                continue
            # end if

            # I flag the record.
            _log_prob_obj = __flagger.flag(source_text=_record.source, 
                                           threshold=threshold_flag,
                                           sentence_id=_record.sentence_id,)
            _db_record = DbTableRecordGuerreiro2023SeqLogProb(
                config_name=config_name,
                sentence_id=str(_record.sentence_id),
                log_probability=_log_prob_obj.log_probability,
                log_probability_threshold=threshold_flag,
                flagging_label=_log_prob_obj.is_hallucination,
                flagging_argument_json=json.dumps(asdict(_log_prob_obj))
            )
            _record_dict = asdict(_db_record)
            management_db_handler.insert(table_name=DbTableRecordGuerreiro2023SeqLogProb.__name__, data=_record_dict)
    # end for


def mcd_sim_Guerreiro_2023_flagging(
        config_name: str,
        algorithm_config_obj: Guerreiro2023MCDSim,
        seq_dataset_eval: ty.List[HalomiDatasetRecord],
        management_db_handler: DBHandlerExp,
        file_logger: logging.Logger,
        path_cache_dir: Path):
    # I set the cache directory.
    path_cache_dir_algorithm = path_cache_dir / Guerreiro2023MCDSim.__name__
    path_cache_dir_algorithm.mkdir(parents=True, exist_ok=True)

    model_encoder_decoder_mt = load_fairseq_model(
        path_fairseq_model_dir=algorithm_config_obj.path_fairseq_model_dir,
        path_fairseq_model_file=algorithm_config_obj.path_fairseq_model_file,
        path_sentencepiece_model=algorithm_config_obj.path_sentencepiece_model)

    # I fix the method with METEOR.    
    # file_logger.debug(f"Initialising the metric object.")    
    # try:
    #     metric_obj_class = getattr(flagger_mc_dropout, algorithm_config_obj.dissimilarity_metric_class_name)
    # except AttributeError:
    #     raise Exception(f"Metric class not found: {algorithm_config_obj.dissimilarity_metric_class_name}")
    # # end try

    flagger_obj = FlaggerDisSimilarityMcDropOut(
        fairseq_interface=model_encoder_decoder_mt,
        metric_obj=flagger_mc_dropout.MeteorMetric(),
        path_dir_cache=path_cache_dir_algorithm,
        num_samples=algorithm_config_obj.num_samples,
        temperature_value=1.0,
        max_len_a=algorithm_config_obj.max_len_a,
        max_len_b=algorithm_config_obj.max_len_b)
    
    seq_eval_records = [
        EvaluationTargetTranslationPair(source=__record.source, target=__record.translation, sentence_id=str(__record.sentence_id))
        for __record in seq_dataset_eval]

    file_logger.info(f"Computing DSIM-MC-Drop over records of a dataset.")
    # I compute the log-probability for each record and obtain the distribution of a dataset.
    # [NOTE] I use the whole dataset for computing the distribution. That follows the description of Guerreiro, 2023, Appendix G.
    # I compute the threshold for the log-probability.
    seq_stats = flagger_obj.compute_dataset_statistics(seq_eval_records)
    threshold_flag = flagger_obj.get_flag_threshold(seq_stats, algorithm_config_obj.percentile)
    file_logger.info(f"Threshold for flagging: {threshold_flag}")

    # I flag the records.
    # seq_hallucination_records = [__r for __r in seq_dataset_eval if __r.error_type == "hallucination"]
    for _record in tqdm.tqdm(seq_dataset_eval, desc=f"Processing {algorithm_config_obj.approach_name}", file=sys.stdout):
        # I check the result DB. Skip if the record is already processed.
        _exp_primary_key = DbTableRecordGuerreiro2023McDSIM.get_record_id(
            config_name=config_name,
            _sentence_id=str(_record.sentence_id))
        _is_exists = management_db_handler.is_record_exists(
            table_name=DbTableRecordGuerreiro2023McDSIM.__name__,
            exp_key=_exp_primary_key,
            is_partially_search=False,
            primary_key='record_id')
        if _is_exists:
            continue
        # end if

        # I flag the record.
        _res_obj_dsim_mc_dropout = flagger_obj.flag(
            evaluation_target=EvaluationTargetTranslationPair(\
                source=_record.source,
                target=_record.translation,
                sentence_id=str(_record.sentence_id)), 
            threshold=threshold_flag,
            is_use_cache=True)

        _db_record = DbTableRecordGuerreiro2023McDSIM(
            config_name=config_name,
            sentence_id=str(_record.sentence_id),
            score=_res_obj_dsim_mc_dropout.avg_dissimilarity,
            score_threshold=threshold_flag,
            flagging_label=_res_obj_dsim_mc_dropout.is_hallucination,
            flagging_argument_json=json.dumps(asdict(_res_obj_dsim_mc_dropout))
        )
        _record_dict = asdict(_db_record)
        management_db_handler.insert(table_name=DbTableRecordGuerreiro2023McDSIM.__name__, data=_record_dict)
    # end for



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


def create_file_logger(resouce_config: ResourceConfig) -> logging.Logger:
    (resouce_config.path_work_dir / resouce_config.dir_name_log).mkdir(parents=True, exist_ok=True)
    path_log_dir = resouce_config.path_work_dir / resouce_config.dir_name_log / "app.log"

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

    # retrive module loggers.
    logger_mmd_flagger_ver1 = logging.getLogger(mmd_error_flagger_ver1.__name__)

    # Attach main logger's handlers to submodule logger
    if not logger_mmd_flagger_ver1.handlers:  # Avoid duplicate handlers
        for h in logger.handlers:
            logger_mmd_flagger_ver1.addHandler(h)
    # end if
    logger.info(f"-------- Script Start --------")

    return logger


def main(path_config_toml: Path):
    assert path_config_toml.exists(), f"File not found: {path_config_toml}"

    # loading the config file
    with open(path_config_toml, "r") as f:
        config_dict = toml.load(f)
    # end with
    interface_config = create_interface_config(config_dict)
    # check script version compatibility
    assert interface_config.script_version == SCRIPT_VERSION, f"Script version mismatch. Expected {SCRIPT_VERSION}, got {interface_config.script_version}"

    main_logger = create_file_logger(interface_config.resource_config)

    # loading the dataset tsv
    path_dataset_tsv = interface_config.resource_config.path_dataset_tsv
    seq_dataset_records = load_dataset(path_dataset_tsv)
    
    # setting the management db.
    path_management_db = interface_config.resource_config.path_work_dir / interface_config.resource_config.file_name_db_sqlite3
    db_connection = sqlite3.connect(path_management_db)
    create_table_from_table_definition(db_connection, DbTableRecordRaunak2021, primary_key="record_id")
    create_table_from_table_definition(db_connection, DbTableRecordProposalMmdFlaggerVer1, primary_key="record_id")
    create_table_from_table_definition(db_connection, DbTableRecordProposalMmdFlaggerTrajectoryVer2, primary_key="record_id")
    create_table_from_table_definition(db_connection, DbTableRecordGuerreiro2023SeqLogProb, primary_key="record_id")
    create_table_from_table_definition(db_connection, DbTableRecordGuerreiro2023McDSIM, primary_key="record_id")
    db_handler = DBHandlerExp(path_management_db)

    # TODO: I need a function to route approach functions
    seq_approach_names = interface_config.flagging_approaches
    if interface_config.resource_config.path_dir_cache_translation is not None:
        path_cache_dir = interface_config.resource_config.path_dir_cache_translation
        main_logger.info(f"Cache directory: {path_cache_dir}")
    else:
        path_cache_dir = interface_config.resource_config.path_work_dir / interface_config.resource_config.dir_name_cache
        path_cache_dir.mkdir(parents=True, exist_ok=True)
    # end if

    for _approach_name in seq_approach_names:
        if _approach_name == "MmdErrorFlaggerVer1":
            raise NotImplementedError("This approach is not ready yet.")
            mmd_flagger_ver1_obj = interface_config.approach_configs["MmdErrorFlaggerVer1"]
            assert isinstance(mmd_flagger_ver1_obj, MmdErrorFlaggerVer1), f"Expected MmdErrorFlaggerVer1, got {mmd_flagger_ver1_obj}"
            proposal_mmd_flagging_ver1(
                config_name=interface_config.config_name,
                mmd_flagger_ver1_obj=mmd_flagger_ver1_obj,
                seq_dataset_eval=seq_dataset_eval,
                seq_calibration_record=seq_calibration_record,
                management_db_handler=db_handler,
                file_logger=main_logger,
                path_cache_dir=path_cache_dir
            )
        elif _approach_name == "MmdErrorFlaggerTrajectoryVer2":
            raise NotImplementedError("This implementation is out-dated.")            
            algorithm_config = interface_config.approach_configs["MmdErrorFlaggerTrajectoryVer2"]
            assert isinstance(algorithm_config, MmdErrorFlaggerTrajectoryVer2), f"Expected MmdErrorFlaggerTrajectoryVer2, got {algorithm_config}"
            proposal_mmd_flagging_trajectory_ver2(
                config_name=interface_config.config_name,
                resource_config=interface_config.resource_config,
                algorithm_config_obj=algorithm_config,
                seq_dataset=seq_dataset_records,
                management_db_handler=db_handler,
                file_logger=main_logger,
                path_cache_dir=path_cache_dir
            )
        elif _approach_name == "Raunak2021ApproachConfig":
            # I want to skip records that are already flagged
            raunak_2021_config_obj = interface_config.approach_configs["Raunak2021ApproachConfig"]
            assert isinstance(raunak_2021_config_obj, Raunak2021ApproachConfig), f"Expected Raunak2021ApproachConfig, got {raunak_2021_config_obj}"
            oscillatory_detection_Raunak_2021_flagging(
                config_name=interface_config.config_name,
                raunak_2021_config_obj=raunak_2021_config_obj,
                seq_dataset_eval=seq_dataset_records,
                management_db_handler=db_handler,
                file_logger=main_logger,
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
                file_logger=main_logger,
                path_cache_dir=path_cache_dir
            )
        elif _approach_name == "Guerreiro2023SeqLogprob":
            algorithm_config = interface_config.approach_configs["Guerreiro2023SeqLogprob"]
            assert isinstance(algorithm_config, Guerreiro2023SeqLogprob), f"Expected Guerreiro2023SeqLogprob, got {algorithm_config}"
            seq_log_prob_Guerreiro_2023_flagging(
                config_name=interface_config.config_name,
                algorithm_config_obj=algorithm_config,
                seq_dataset_eval=seq_dataset_records,
                management_db_handler=db_handler,
                file_logger=main_logger,
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
    main_logger.info(f"-------- Script End --------")


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

    seq_dataset_record = load_dataset(interface_config.resource_config.path_dataset_tsv)

    # db file
    path_prediction_database = interface_config.resource_config.path_work_dir / interface_config.resource_config.file_name_db_sqlite3
    assert path_prediction_database.exists(), f"File not found: {path_prediction_database}"

    eval_runner = evaluation_script_ver2.EvaluationVer2(
        seq_dataset_record=seq_dataset_record,
        path_prediction_database=path_prediction_database)
    
    seq_eval_table_name = [
        DbTableRecordRaunak2021.__name__,
        # DbTableRecordProposalMmdFlaggerVer1.__name__,
        # DbTableRecordProposalMmdFlaggerTrajectoryVer2.__name__,
        DbTableRecordGuerreiro2023McDSIM.__name__,
        DbTableRecordGuerreiro2023SeqLogProb.__name__
    ]
    eval_runner.main(
        path_output_dir=path_output_dir,
        config_name=interface_config.config_name, 
        seq_eval_table_name=seq_eval_table_name
    )

    # result analysis runner
    path_analysis = interface_config.resource_config.path_work_dir / interface_config.evaluation_config.dir_name_analysis
    path_analysis.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("-c", "--config", required=True, type=str, help="Path to the config file.")
    _parser.add_argument("-m", "--mode", required=True, type=str, help="Mode: flagging or evaluation", choices=["flagging", "flag", "evaluation", "eval"])
    _args = _parser.parse_args()

    if _args.mode == "flagging" or _args.mode == "flag":
        path_config_toml = Path(_args.config)
        main(path_config_toml)
    elif _args.mode == "evaluation" or _args.mode == "eval":
        path_config_toml = Path(_args.config)
        evaluate_main(path_config_toml)
    else:
        raise Exception(f"Mode not implemented: {_args.mode}")
    # end if
