import typing as ty
import sqlite3
import itertools
import logging
import json
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plot
import seaborn as sns
import collections
import random
import dataclasses
import re
import hashlib
import tqdm

import torch
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, PrecisionRecallDisplay

from ...guerreiro_2023_wmt.data_models.data_models import WMTDatasetRecord
from ...guerreiro_2023_wmt.data_models.utils import load_dataset as load_dataset_guerreiro_2023_wmt

from ...dale_2023_halomi.load_dataset import (
    load_dataset as load_dataset_dale_2023_halomi,
    HalomiDatasetRecord
)


from ..module_management_db.module_sqlite3_handler import DBHandlerExp
from hallucination_mt.module_assessments.module_management_db.interface_ver3.module_db_record import (
    DbTableRecordRaunak2021,
    DbTableRecordGuerreiro2023SeqLogProb,
    DbTableRecordGuerreiro2023McDSIM,
    DbTableRecordProposalMmdFlaggerTrajectoryVer3
)

from .module_evaluation_script_ver3 import (
    generate_eval_key_mmd_trajectory_flagger,
    plot_mmd_trajectory,
    get_ground_truth_label
)

from hallucination_mt.logger_module import formatter


logger = logging.getLogger(__name__)
std_handler = logging.StreamHandler()
std_handler.setLevel(logging.DEBUG)
std_handler.setFormatter(formatter)


# DEFAULT_GROUND_TRUTH_SETTINGS = (
#     "hallucination", 
#     "hallucination+mt-error", 
#     "mt-error", 
#     "error_named_entity",
#     "error_omission",
#     "error_full",
#     "error_strong",
#     "error_repetitions",
#     "2_Small_hallucination",
#     "3_Partial_hallucination",
#     "4_Full_hallucination"
# )


class EvaluationResultContainer(ty.NamedTuple):
    n_total: int
    n_target_label: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    precision: float
    recall: float
    f1: float


class AggregatedMmdFlaggerVer1(ty.NamedTuple):
    temperature_low: str
    temperature_high: str
    dict_sentid2label: ty.Dict[int, int]

    def __post_init__(self):
        assert isinstance(self.dict_sentid2label, dict), f"Invalid type: {type(self.dict_sentid2label)}"
        assert all(isinstance(_k, int) for _k in self.dict_sentid2label.keys()), "Invalid key type."
        assert all(isinstance(_v, int) for _v in self.dict_sentid2label.values()), "Invalid value type."


class EvaluationRecord(ty.NamedTuple):
    approach_name: str
    ground_truth_setting: str
    evaluation_result: EvaluationResultContainer

    def to_dict_record(self) -> ty.Dict:
        dict_eval = self.evaluation_result._asdict()
        dict_return = self._asdict()
        dict_return.update(dict_eval)
        return dict_return


# class _KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3(ty.NamedTuple):
#     tau_sequence: str
#     n_sampling: int
#     mode_vector_preprocess: str
#     mode_target_embedding_layer: str
#     mode_max_token_length_vector_concat: str
#     mode_max_token_length: int
#     args_kernel_options_json: str
#     args_trajectory_options_json: str


# # sort and aggregate the records by the field `tau_sequence` & `embedding_option` & `n_sampling` & filter options.
# def _func_aggregation_key_DbTableRecordProposalMmdFlaggerTrajectoryVer3(record: DbTableRecordProposalMmdFlaggerTrajectoryVer3) -> _KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3[str, ...]:
#     # return record['tau_sequence'], record['n_sampling'], record['embedding_option'], record['trajectory_rule'], record['trajectory_rule_smoothing'], record['trajectory_rule_smoothing_window']
#     return _KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3(
#         record.tau_sequence,
#         record.n_sampling,
#         record.mode_vector_preprocess,
#         record.mode_target_embedding_layer,
#         record.mode_max_token_length_vector_concat,
#         record.mode_max_token_length,
#         record.args_kernel_options_json,
#         record.args_trajectory_options_json
#     )    
# # end def


class EvaluationVer3(object):
    """EvaluationVer2 is a class for evaluating the performance of various models.
    It can handle the evaluations for the LFAN-HALL dataset (Guerreiro et al., 2023) and the Halomi dataset (Dale et al., 2023).
    """
    def __init__(self,
                 seq_dataset_record: ty.List[ty.Union[WMTDatasetRecord, HalomiDatasetRecord]],
                 path_prediction_database: Path,
                 dataset_type: str,
                 random_seed: int = 42) -> None:
        assert path_prediction_database.exists(), f"Database file not found: {path_prediction_database}"
        self.seq_dataset_record = seq_dataset_record
        # opening the database file.
        db_con = sqlite3.connect(str(path_prediction_database))
        self.db_handler = DBHandlerExp(path_prediction_database)
        self.db_handler.conn = db_con

        self.random_seed = random_seed  # I use the seed for the example selection.

        # -------------------------------------------------
        # distinguishing dataset type
        self.dataset_type: str = dataset_type

    @staticmethod
    def _func_get_evaluation(
            array_label_prediction: npt.NDArray[np.int8],
            array_label_ground_truth: npt.NDArray[np.int8]
            ) -> EvaluationResultContainer:
        """An abstract function to get the evaluation metric."""
        # confusion matrix
        # computing elements of a confusion matrix.
        matrix_confusion = confusion_matrix(array_label_ground_truth, array_label_prediction, labels=[0, 1])
        tn, fp, fn, tp = matrix_confusion.ravel()

        # computing p, r, f1
        # evaluation metrics
        if tp == 0:
            precision = 0.0
            recall = 0.0
            f_score = 0.0
        else:
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
        # end if

        return EvaluationResultContainer(
            n_total=len(array_label_ground_truth),
            n_target_label=int(np.sum(array_label_ground_truth)),
            true_positive=int(tp),
            false_positive=int(fp),
            true_negative=int(tn),
            false_negative=int(fn),
            precision=precision,
            recall=recall,
            f1=f_score)
    
    # -------------------------------------------------------
    # methods for writing out the results.

    def _write_out_mmd_flagger_heatmap(self,
                                       seq_evaluation_records: ty.List[EvaluationRecord],
                                       dict_approach2parameters: ty.Dict[str, ty.Tuple[float, float]],
                                       path_dir_output_heatmap: Path,
                                       target_score: str = "recall",
                                       seq_ground_truth_setting: ty.Tuple[str,...] = tuple(["hallucination", "hallucination+mt-error", "error_strong", "error_repetitions", "error_full", "error_omission"])
                                       ) -> None:
        """Writing out the heatmap.
        X: temperature-low, Y: temperature-high, Z: Recall (optional)"""

        seq_keys_mmd_flaggers = list(dict_approach2parameters.keys())
        seq_evaluation_mmd_flaggers = [rec for rec in seq_evaluation_records if rec.approach_name in seq_keys_mmd_flaggers]
        assert len(seq_evaluation_mmd_flaggers) > 0, "No records found."

        for _ground_truth_setting in seq_ground_truth_setting:
            _path_output_heatmap = path_dir_output_heatmap / f"heatmap_{target_score}_score_{_ground_truth_setting}.png"

            _seq_evaluation_target = [_obj for _obj in seq_evaluation_mmd_flaggers if _obj.ground_truth_setting == _ground_truth_setting]
            assert len(_seq_evaluation_target) > 0, f"No records found for {_ground_truth_setting}."
            seq_dict_evaluation = []
            for _eval_obj in _seq_evaluation_target:
                _approach_name = _eval_obj.approach_name
                assert _approach_name in dict_approach2parameters, f"Invalid approach name: {_approach_name}"
                _eval_score_obj = _eval_obj.evaluation_result
                if target_score == "recall":
                    _score_eval = _eval_score_obj.recall
                elif target_score == "precision":
                    _score_eval = _eval_score_obj.precision
                elif target_score == "f1":
                    _score_eval = _eval_score_obj.f1
                else:
                    raise ValueError(f"Unsupported target score: {target_score}")
                # end if
                _temperature_low, _temperature_high = dict_approach2parameters[_approach_name]

                _dict_record = dict(
                    tau_a=_temperature_low,
                    tau_b=_temperature_high,
                    score=_score_eval
                )
                seq_dict_evaluation.append(_dict_record)
            # end for

            # I want to visualise X: \tau_a, Y: \tau_b, Colour: recall.
            df_evaluation = pd.DataFrame(seq_dict_evaluation)

            # Pivot the DataFrame to get a matrix suitable for heatmap
            f, ax = plot.subplots()

            heatmap_data = df_evaluation.pivot(index='tau_a', columns='tau_b', values='score')
            sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', ax=ax)
            ax.set_xlabel('Tau B')
            ax.set_ylabel('Tau A')
            ax.set_title(f'{target_score} Score Heatmap')
            # __path_save_heatmap = path_visualisation / "heatmap_recall_score.png"
            f.savefig(_path_output_heatmap.as_posix())
            logger.info(f"Saved the heatmap to {_path_output_heatmap}")

    def _write_out_evaluation_excel_book(self,
                                         path_excel: Path,
                                         seq_evaluation_records: ty.List[EvaluationRecord]):
        """Writing out the evaluation results into an Excel file.
        
        The excel book desing is as follows:
        - Sheet name: ground-truth-setting name.

        Each sheet contains the following columns:
        | Approach Name | n_total | tp | tn | fp | fn | Precision | Recall | F1 |
        """
        def _func_key(evaluation_record: EvaluationRecord) -> str:
            return evaluation_record.ground_truth_setting
        # end def

        # sorting the evaluation records.
        iter_g_obj = itertools.groupby(sorted(seq_evaluation_records, key=_func_key), key=_func_key)

        COLUMN_HEADER_ORDER = [
            "approach_name",
            "ground_truth_setting",
            "n_total",
            "n_target_label",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1"
        ]
        with pd.ExcelWriter(path_excel) as writer:
            for _ground_truth_setting, _g_obj in iter_g_obj:
                _seq_evaluation_record: ty.List[EvaluationRecord] = list(_g_obj)
                _seq_eval_dict = [_eval_record.to_dict_record() for _eval_record in _seq_evaluation_record]
                _df_eval = pd.DataFrame(_seq_eval_dict)
                _df_eval = _df_eval[COLUMN_HEADER_ORDER]
                _df_eval.to_excel(writer, sheet_name=_ground_truth_setting, index=False)
            # end for
        # end with
        logger.info(f"Excel file is written out: {path_excel}")

    def _write_out_analysis_excel_book(self,
                                       path_excel: Path,
                                       dict_evaluations: ty.Dict[str, ty.Dict[int, int]]):

        def _process_lfan_hall_dataset(seq_dataset_record: ty.List[WMTDatasetRecord]):
            """Writing out the analysis results into an Excel file.
            
            The analysis excel book contains,
            - the sentence-id
            - source-text
            - target-text (by the model with beam-search)
            - reference-text
            -------
            - error-type
            - is-strong-detached
            - is-full-detached
            - is-repetition
            - is-named-entity-error
            - is-omission-error
            -------
            - prediction-label per approach
            """
            assert all(isinstance(obj, WMTDatasetRecord) for obj in seq_dataset_record), "Invalid type."
        
            seq_excel_record_obj = []

            # re-format the evaluation records. {sentence-id: {approach-name: prediction-label}}
            dict_sentid2approach2label = {}
            for _approach_name, _dict_sentid2label in dict_evaluations.items():
                for _sentence_id, _label in _dict_sentid2label.items():
                    if _sentence_id not in dict_sentid2approach2label:
                        _sentence_id = int(_sentence_id)
                        dict_sentid2approach2label[_sentence_id] = {}
                    # end if
                    dict_sentid2approach2label[_sentence_id][_approach_name] = _label
                # end for
            # end for

            __count_record_no_label = 0
            for _eval_record in sorted(seq_dataset_record, key=lambda x: int(x.sentence_id)):
                # record information
                _sentence_id = int(_eval_record.sentence_id)
                _source_text = _eval_record.source
                _target_text = _eval_record.translation
                _referece_text = _eval_record.reference
                # error information (ground-truth)
                _error_type = _eval_record.error_type
                _is_strong_detached = _eval_record.error_strong
                _is_full_detached = _eval_record.error_full
                _is_repetition = _eval_record.error_repetitions
                _is_named_entity_error = _eval_record.error_named_entities

                # prediction labels
                if _sentence_id not in dict_sentid2approach2label:
                    __count_record_no_label += 1
                    # logger.warning(f"No prediction labels found for sentence-id: {_sentence_id}")
                    continue
                # end if
                assert _sentence_id in dict_sentid2approach2label, f"Invalid sentence-id: {_sentence_id}"
                _dict_prediction_labels = dict_sentid2approach2label[_sentence_id]

                # creating a record.
                _record_obj = dict(
                    sentence_id=_sentence_id,
                    source_text=_source_text,
                    target_text=_target_text,
                    reference_text=_referece_text,
                    error_type=_error_type,
                    is_strong_detached=_is_strong_detached,
                    is_full_detached=_is_full_detached,
                    is_repetition=_is_repetition,
                    is_named_entity_error=_is_named_entity_error,
                    **_dict_prediction_labels)
                seq_excel_record_obj.append(_record_obj)
            # end for

            EXCEL_HEADER = [
                "sentence_id",
                "source_text",
                "target_text",
                "reference_text",
                "error_type",
                "is_strong_detached",
                "is_full_detached",
                "is_repetition",
                "is_named_entity_error",
            ]

            assert len(seq_excel_record_obj) > 0, "No records found."

            df_excel = pd.DataFrame(seq_excel_record_obj)
            df_excel = df_excel[EXCEL_HEADER]
            df_excel.to_excel(path_excel, index=False)
            logger.info(f"Excel file is written out: {path_excel}")
            logger.warning(f"N(No prediction labels) -> {__count_record_no_label}")
        # end def

        def _process_halomi_dataset(seq_dataset_record: ty.List[HalomiDatasetRecord]):
            """Writing out the analysis results into an Excel file.
            
            The analysis excel book contains,
            - the sentence-id
            - source-language
            - target-language
            - source-text
            - target-text (by the model with beam-search)
            - reference-text
            -------
            - error-type
            - class-hallucination
            - class-omission
            -------
            - prediction-label per approach
            """
            assert all(isinstance(obj, HalomiDatasetRecord) for obj in seq_dataset_record), "Invalid type."
        
            seq_excel_record_obj = []

            # re-format the evaluation records. {sentence-id: {approach-name: prediction-label}}
            dict_sentid2approach2label = {}
            for _approach_name, _dict_sentid2label in dict_evaluations.items():
                for _sentence_id, _label in _dict_sentid2label.items():
                    if _sentence_id not in dict_sentid2approach2label:
                        _sentence_id = str(_sentence_id)
                        dict_sentid2approach2label[_sentence_id] = {}
                    # end if
                    dict_sentid2approach2label[_sentence_id][_approach_name] = _label
                # end for
            # end for

            for _eval_record in sorted(seq_dataset_record, key=lambda x: str(x.key_unique)):
                # record information
                _sentence_id = str(_eval_record.key_unique)
                _source_text = _eval_record.src_text
                _target_text = _eval_record.tgt_text

                _source_lang = _eval_record.src_lang
                _target_lang = _eval_record.tgt_lang
                
                # error information (ground-truth)
                _error_type = _eval_record.error_type
                _class_hall = _eval_record.class_hall
                _class_omit = _eval_record.class_omit

                # prediction labels
                if _sentence_id not in dict_sentid2approach2label:
                    logger.warning(f"No prediction labels found for sentence-id: {_sentence_id}")
                    continue
                # end if
                assert _sentence_id in dict_sentid2approach2label, f"Invalid sentence-id: {_sentence_id}"
                _dict_prediction_labels = dict_sentid2approach2label[_sentence_id]

                # creating a record.
                _record_obj = dict(
                    sentence_id=_sentence_id,
                    source_lang=_source_lang,
                    target_lang=_target_lang,
                    source_text=_source_text,
                    target_text=_target_text,
                    error_type=_error_type,
                    class_hall=_class_hall,
                    class_omit=_class_omit,
                    **_dict_prediction_labels)
                seq_excel_record_obj.append(_record_obj)
            # end for

            EXCEL_HEADER = [
                "sentence_id",
                "source_lang",
                "target_lang",
                "source_text",
                "target_text",
                "error_type",
                "class_hall",
                "class_omit"
            ] + list(dict_evaluations.keys())

            assert len(seq_excel_record_obj) > 0, "No records found."

            df_excel = pd.DataFrame(seq_excel_record_obj)
            df_excel = df_excel[EXCEL_HEADER]
            df_excel.to_excel(path_excel, index=False)
            logger.info(f"Excel file is written out: {path_excel}")
        # end def


        if isinstance(self.seq_dataset_record[0], WMTDatasetRecord):
            assert all(isinstance(obj, WMTDatasetRecord) for obj in self.seq_dataset_record), "Invalid type."
            _process_lfan_hall_dataset(self.seq_dataset_record)  # type: ignore
        elif isinstance(self.seq_dataset_record[0], HalomiDatasetRecord):
            assert all(isinstance(obj, HalomiDatasetRecord) for obj in self.seq_dataset_record), "Invalid type."
            _process_halomi_dataset(self.seq_dataset_record)  # type: ignore
        else:
            raise ValueError(f"Unsupported dataset record type: {type(self.seq_dataset_record[0])}")
        # end if

    # -------------------------------------------------------
    
    def _fetch_language_pair_labels(self) -> ty.Dict[str, ty.Tuple[str, str]]:
        """Get a dict object. Key is the sentence-id, Value is a tuple of (src-lang, tgt-lang).

        Return:
            {sentence-id: (src-lang, tgt-lang)}
        """
        if self.dataset_type == 'lfan-hall':
            logger.debug(f'No language pair definition for the lfan-hall.')
            return {}
        # end if
        
        return_obj: ty.Dict[str, ty.Tuple[str, str]] = {}
        for _record in self.seq_dataset_record:
            assert isinstance(_record, HalomiDatasetRecord)
            assert _record.key_unique is not None
            return_obj[_record.key_unique] = (_record.src_lang, _record.tgt_lang)
        # end for

        return return_obj

    def _fetch_prediction_results(self,
                                  config_name: str,
                                  eval_table_name: str) -> ty.List[ty.Dict]:
        """Fetching prediction results from the database."""
        # selecting the records having the specified config_name and eval_table_name.
        assert self.db_handler.conn is not None, "Database connection is not established."
        db_connection = self.db_handler.conn
        db_connection.row_factory = sqlite3.Row
        db_cursor = db_connection.cursor()
        sql_query = f"SELECT * FROM {eval_table_name}"

        db_cursor.execute(sql_query,)
        seq_records = db_cursor.fetchall()

        return seq_records
    
    # custom-tailored pre-processing functions for each evaluation method.
    def _prep_mmd_trajectory_flagger_ver3(self,
                                          config_name: str,
                                          table_name: str = DbTableRecordProposalMmdFlaggerTrajectoryVer3.__name__,
                                          approach_name: str = "MmdTrajectoryFlaggerVer3"
                                          ) -> ty.Tuple[
                                               ty.Dict[str, ty.Dict[str, int]], 
                                               ty.Dict[str, generate_eval_key_mmd_trajectory_flagger._KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3],
                                               ty.Dict[str, ty.List[str]]
                                            ]:
        """
        Returns:
            - {aggregation-key: [sentence-id: binary-flag]}
            - {aggregation-key: _KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3}
            - {aggregation-key: [db-primary-key]}
        """
        # fetching db records
        seq_records = self._fetch_prediction_results(config_name=config_name, eval_table_name=table_name)
        if len(seq_records) == 0:
            logger.warning(f"No records found for {config_name} in {table_name}.")
            return {}, {}, {}
        # end if

        assert len(seq_records) > 0, f"No records found for {config_name} in {table_name}."
        
        # seq_db_record_obj = [DbTableRecordProposalMmdFlaggerTrajectoryVer3(**_t) for _t in seq_records]
        seq_db_record_obj = []
        for _t in seq_records:
            _obj = DbTableRecordProposalMmdFlaggerTrajectoryVer3(**_t)
            _obj.record_id = _t['record_id']
            seq_db_record_obj.append(_obj)
        # end for

        seq_predictions = []
        dict_aggkey2seq_db_primary_key = {}

        iter_group = itertools.groupby(
            sorted(seq_db_record_obj, key=generate_eval_key_mmd_trajectory_flagger._func_aggregation_key_DbTableRecordProposalMmdFlaggerTrajectoryVer3), 
            key=generate_eval_key_mmd_trajectory_flagger._func_aggregation_key_DbTableRecordProposalMmdFlaggerTrajectoryVer3)
        _t_key: generate_eval_key_mmd_trajectory_flagger._KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3
        for _t_key, __g_obj in iter_group:
            _seq_records = list(__g_obj)
            
            _dict_sentid2label = {str(_d.sentence_id): _d.flagging_label for _d in _seq_records}

            _obj = (_t_key, _dict_sentid2label)
            
            seq_predictions.append(_obj)
            dict_aggkey2seq_db_primary_key[f'{approach_name}-{_t_key.generate_code_name()}'] = [_obj.record_id for _obj in _seq_records]
        # end for

        dict_evaluations = {}
        _aggregated_container: ty.Tuple[generate_eval_key_mmd_trajectory_flagger._KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3, ty.Dict]
        for _aggregated_container in seq_predictions:
            dict_evaluations[f'{approach_name}-{_aggregated_container[0].generate_code_name()}'] = _aggregated_container[1]
        # end

        dict_code_name2agg = {}
        for _aggregated_container in seq_predictions:
            dict_code_name2agg[f'{approach_name}-{_aggregated_container[0].generate_code_name()}'] = _aggregated_container[0]
        # end

        return dict_evaluations, dict_code_name2agg, dict_aggkey2seq_db_primary_key

    def _prep_baselines(self, 
                          config_name: str,
                          table_name: str,
                          eval_key_name: str
                          ) -> ty.Dict[str, ty.Dict[str, int]]:
        """Preprocessing function for baselines. The function can process baselines of
        - Raunak 2021.
        - DSIM-MC DropOut
        - SeqLogProb
        """
        assert table_name in (
            DbTableRecordRaunak2021.__name__,
            DbTableRecordGuerreiro2023McDSIM.__name__,
            DbTableRecordGuerreiro2023SeqLogProb.__name__
        ), f"Unsupported table name: {table_name}"

        # fetching db records
        seq_records = self._fetch_prediction_results(config_name=config_name, eval_table_name=table_name)
        if len(seq_records) == 0:
            return {}
        else:
            assert len(seq_records) > 0, f"No records found for {config_name} in {table_name}."
            # obtaining the ground truth labels.
            dict_sentid2label = {str(_d['sentence_id']): _d['flagging_label'] for _d in seq_records}

            dict_evaluations = {eval_key_name: dict_sentid2label}
            return dict_evaluations
    
    def _conduct_evaluation(self,
                            seq_ground_truth_settings: ty.List[str],
                            dict_evaluations: ty.Dict[str, ty.Dict[str, int]],
                            mode_processing_none_prediction: str = "zero_replacement",
                            source_lang: ty.Optional[str] = None,
                            target_lang: ty.Optional[str] = None,
                            dict_sent_id2lang_tuple: ty.Optional[ty.Dict[str, ty.Tuple[str, str]]] = None,
                            ) -> ty.List[EvaluationRecord]:
        """
        Args:
            mode_processing_none_prediction: An option of processing None prediction labels.
                zero_replacement: replace None with 0. This mode is fair for the evaluation.
                drop_none: drop None labels.
        """
        assert mode_processing_none_prediction in ("zero_replacement", "drop_none"), f"Unsupported mode: {mode_processing_none_prediction}"

        # Evaluation values under various conditions, e.g. hallucination+MT-Error, hallucination only, full-detached only etc..
        seq_evaluation_records_container: ty.List[EvaluationRecord] = []
        for __setting_ground_truth in seq_ground_truth_settings:
            dict_ground_truth = get_ground_truth_label._get_ground_truth_labels(seq_dataset_record=self.seq_dataset_record,
                                                                                label_setting=__setting_ground_truth,
                                                                                dataset_name=self.dataset_type)
            for __eval_method_name, __dict_sentid2label in dict_evaluations.items():
                # alignment of sentence-id between ground-truth and prediction.
                __sentence_id_common: ty.List[str] = sorted(list(set(__dict_sentid2label.keys()) & set(dict_ground_truth.keys())))
                assert len(__sentence_id_common) > 0, "No common sentence-ids found."

                if dict_sent_id2lang_tuple is not None:
                    assert source_lang is not None and target_lang is not None

                    # filtering the sentence-id by source,  target languages.
                    _seq_filtered_sentence_id_common = []
                    for _sent_id in __sentence_id_common:
                        _t_language_pair: ty.Tuple[str, str] = dict_sent_id2lang_tuple[_sent_id]
                        if _t_language_pair[0] == source_lang and _t_language_pair[1] == target_lang:
                            _seq_filtered_sentence_id_common.append(_sent_id)
                        # end if
                    # end for
                    if len(_seq_filtered_sentence_id_common) == 0:
                        logger.debug(f'No sentence-id in common at the language pair {source_lang} and {target_lang}')
                        continue
                    # end if

                    __sentence_id_common = _seq_filtered_sentence_id_common
                # end if

                __seq_prediction_labels = [__dict_sentid2label[_k] for _k in __sentence_id_common]
                __seq_ground_truth_labels = [dict_ground_truth[_k] for _k in __sentence_id_common]
                assert len(__seq_prediction_labels) == len(__seq_ground_truth_labels), "Invalid length of the labels."

                if mode_processing_none_prediction == "zero_replacement":
                    # replacing None with 0.
                    __seq_prediction_labels = [0 if _e is None else _e for _e in __seq_prediction_labels]
                elif mode_processing_none_prediction == "drop_none":
                    # removing None element. None can happen because \tau parameter are mal-seeting and FairSeq could not generate translations.
                    _ind_none_prediction = np.where(np.array(__seq_prediction_labels) == None)[0]
                    __seq_prediction_labels = [__e for __i, __e in enumerate(__seq_prediction_labels) if __i not in _ind_none_prediction]
                    __seq_ground_truth_labels = [e for __i, e in enumerate(__seq_ground_truth_labels) if __i not in _ind_none_prediction]
                    __sentence_id_common = [e for __i, e in enumerate(__sentence_id_common) if __i not in _ind_none_prediction]
                # end if
                assert len(__seq_prediction_labels) == len(__seq_ground_truth_labels) == len(__sentence_id_common), "Invalid length of the labels."                
                
                # evaluation values.
                __array_prediction = np.array(__seq_prediction_labels, dtype=np.int8)
                __array_ground_truth = np.array(__seq_ground_truth_labels, dtype=np.int8)

                if np.sum(__array_ground_truth) == 0:
                    # logger.warning(f"No target labels found for {__setting_ground_truth}. Something Wrong. I skip this evaluation.")
                    continue
                # end if

                # calculating evaluation values. 
                # try:
                #     _eval_result = self._func_get_evaluation(array_label_prediction=__array_prediction,
                #                                             array_label_ground_truth=__array_ground_truth)
                # except Exception as e:
                #     breakpoint()
                _eval_result = self._func_get_evaluation(array_label_prediction=__array_prediction, array_label_ground_truth=__array_ground_truth)

                # creating an evaluation record.
                _eval_record = EvaluationRecord(approach_name=__eval_method_name,
                                                ground_truth_setting=__setting_ground_truth,
                                                evaluation_result=_eval_result)
                seq_evaluation_records_container.append(_eval_record)
            # end for
        # end for
        # assert len(seq_evaluation_records_container) > 0, "No records found."
        return seq_evaluation_records_container
    
    def _conduct_and_write_out_evaluation_language_pairs(self,
                                                         path_xlsx_file: Path,
                                                         dict_evaluations: ty.Dict[str, ty.Dict[str, int]],
                                                         dict_sent_id2lang_tuple: ty.Dict[str, ty.Tuple[str, str]],):
        """
        
        Args:
            dict_evaluations: {configuration-name: {sentence-id: label}}
            dict_sent_id2lang_tuple: {sentence-id: (source-language, target-language)}
        """

        _seq_lang_pair = sorted(list(set(dict_sent_id2lang_tuple.values())))
        # count the pair
        dict_langpair2_count = dict(collections.Counter(dict_sent_id2lang_tuple.values()))

        # One sheet: approach-full-name to sheet-name
        _d_approach_name2sheet_number = {}
        _stack_corresponding_approach2sheet_name = []

        for _index_approach_name, _approach_name in enumerate(dict_evaluations.keys()):
            _stack_corresponding_approach2sheet_name.append(dict(approach_name=_approach_name, sheet_name=f"sheet_name-{_index_approach_name}"))
            _d_approach_name2sheet_number[_approach_name] = _index_approach_name
        # end for

        # -------------------------------------------------------------------------------------------
        excel_writer = pd.ExcelWriter(path_xlsx_file)

        pd.DataFrame(_stack_corresponding_approach2sheet_name).to_excel(excel_writer, sheet_name="Table_Name_Hash")

        # exporting the counts of language pair
        _t_lang_pair: ty.Tuple[str, str]
        _seq_stack_latex_table_lang_pair = []
        for _t_lang_pair in _seq_lang_pair:
            _value_lang_pair = f'{_t_lang_pair[0].replace("_", "-")} to {_t_lang_pair[1].replace("_", "-")}'
            _seq_stack_latex_table_lang_pair.append(dict(
                    language_pair=_value_lang_pair,
                    error_type='hallucination',
                    value=dict_langpair2_count.get(_t_lang_pair, 0)
            ))
        # end for
        _df_eval_lang_pair = pd.DataFrame(_seq_stack_latex_table_lang_pair)
        _df_eval_lang_pair.to_excel(excel_writer, sheet_name="Counts")


        # exporting the evaluation results
        for _approach_name, _d_eva_results in dict_evaluations.items():
            _dict_evaluations = {_approach_name: _d_eva_results}
            _seq_stack_latex_table_lang_pair = []

            _t_lang_pair: ty.Tuple[str, str]
            for _t_lang_pair in _seq_lang_pair:
                _value_lang_pair = f'{_t_lang_pair[0].replace("_", "-")} to {_t_lang_pair[1].replace("_", "-")}'
                _seq_eval_record_error_type = self._conduct_evaluation(
                    seq_ground_truth_settings=['hallucination'],  # fixing the error-type into the hallucination.
                    dict_evaluations=_dict_evaluations,
                    source_lang=_t_lang_pair[0],
                    target_lang=_t_lang_pair[1],
                    dict_sent_id2lang_tuple=dict_sent_id2lang_tuple)
                
                for _eval_obj in _seq_eval_record_error_type:
                    _value_eval = f'{_eval_obj.evaluation_result.recall:.2f} ({_eval_obj.evaluation_result.precision:.2f})' # recall (precision)
                    _seq_stack_latex_table_lang_pair.append(dict(
                        language_pair=_value_lang_pair,
                        error_type='hallucination',
                        value=_value_eval
                    ))
                # end for
            # end for
            
            # writing to Excel
            _df_eval_lang_pair = pd.DataFrame(_seq_stack_latex_table_lang_pair)
            _sheet_name = f"sheet_name-{_d_approach_name2sheet_number[_approach_name]}"
            _df_eval_lang_pair.to_excel(excel_writer, sheet_name=_sheet_name)
        # end for
        excel_writer.close()


    def _write_out_evaluation_submission_data(self,
                                              path_submission_data_dir: Path,
                                              seq_evaluation_records_container: ty.List[EvaluationRecord],
                                              dict_evaluations: ty.Dict[str, ty.Dict[str, int]],
                                              dict_sent_id2lang_tuple: ty.Optional[ty.Dict[str, ty.Tuple[str, str]]],
                                              target_error_types: ty.Optional[ty.List[str]] = None):
        """A method of generating data for the paper submission.

        Args:
           dict_sent_id2lang_tuple: {sentence-id: (src-lang, tgt-lang)} .
        """
        def halomi_dataset():
            d_method_shown_definition = {
                "Raunak2021": "TNG",
                "Guerreiro2023McDSIM": "MC-DSIM",
                "Guerreiro2023SeqLogProb": "SeqLogProb",
                "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]_decoder.embed_tokens": "MMD-Hal-Flagger",
                "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.4, 0.8, 1.0, 1.2]_decoder.embed_tokens": "MMD-Hal-Flagger",
            }

            # comment: the current code is customised to Halomi.
            if self.dataset_type == 'lfan_hall':
                return None
            # end if

            # if target_error_types is None:
            #     target_error_types = ['error_full', 'error_strong', 'error_repetitions', 'error_omission', 'error_named_entity']
            # # end if

            seq_ground_truth_settings = ["2_Small_hallucination", "3_Partial_hallucination", "4_Full_hallucination"]

            # ----------------------------------------------------------------------

            # creating a eval table per error-type and language.
            path_subdir_lang_and_error_type = path_submission_data_dir / 'lang_and_error'
            path_subdir_lang_and_error_type.mkdir(parents=True, exist_ok=True)

            stack_lang_pair_eval: ty.List[ty.Dict] = []
            # redo the evaluation per error-type and lang-pair.
            _seq_lang_pair = sorted(list(set(dict_sent_id2lang_tuple.values())))
            _t_lang_pair: ty.Tuple[str, str]
            for _t_lang_pair in _seq_lang_pair:
                _seq_eval_record = self._conduct_evaluation(
                    seq_ground_truth_settings=seq_ground_truth_settings,
                    dict_evaluations=dict_evaluations,
                    source_lang=_t_lang_pair[0],
                    target_lang=_t_lang_pair[1],
                    dict_sent_id2lang_tuple=dict_sent_id2lang_tuple)
                # filter the approach using `d_method_shown_definition`
                _seq_eval_record = [
                    _record for _record in _seq_eval_record 
                    if _record.approach_name in d_method_shown_definition.keys()]
                # stack_lang_pair_eval.append(tuple([_t_lang_pair[0], _t_lang_pair[1], _seq_eval_record]))
                for _eval_obj in _seq_eval_record:
                    stack_lang_pair_eval.append(dict(
                        source_language=_t_lang_pair[0],
                        target_language=_t_lang_pair[1],
                        approach_name=_eval_obj.approach_name,
                        approach_name_shown=d_method_shown_definition.get(_eval_obj.approach_name, None),
                        error_label=_eval_obj.ground_truth_setting,
                        **_eval_obj.evaluation_result._asdict()
                    ))
                # end for
            # end for
            df_lang_and_error_eval_table = pd.DataFrame(stack_lang_pair_eval)

            # format a pivot table. 1st-Row: language pair (string type), 2nd-Row approach-name. Column: Error-type

            # TODO this block later.
            path_lang_and_error_eval_tsv = path_subdir_lang_and_error_type / "lang_and_error_eval.tsv"
            df_lang_and_error_eval_table.to_csv(path_lang_and_error_eval_tsv, sep='\t', index=False)

            # ----------------------------------------------------------------------
            # Evaluation per Error Type (Approach-name * Error-Type)
            path_subdir_error_type = path_submission_data_dir / 'error_type'
            path_subdir_error_type.mkdir(parents=True, exist_ok=True)

            # get counts of error
            d_error_label2count = {}
            for _error_label in seq_ground_truth_settings:
                _d_sent_id = get_ground_truth_label._get_ground_truth_labels(
                    seq_dataset_record=self.seq_dataset_record,
                    dataset_name=self.dataset_type,
                    label_setting=_error_label)
                d_error_label2count[_error_label] = sum(_d_sent_id.values())
            # end for

            _seq_eval_record_error_type = self._conduct_evaluation(
                seq_ground_truth_settings=seq_ground_truth_settings,
                dict_evaluations=dict_evaluations,
                source_lang=_t_lang_pair[0],
                target_lang=_t_lang_pair[1],
                dict_sent_id2lang_tuple=dict_sent_id2lang_tuple)
            # filter the approach using `d_method_shown_definition`
            _seq_eval_record_error_type = [
                _record for _record in _seq_eval_record_error_type 
                if _record.approach_name in d_method_shown_definition.keys()]

            _seq_stack_latex_table_error_type = []
            # format latex table.
            for _eval_record in _seq_eval_record_error_type:
                _value_string = f'{_eval_record.evaluation_result.precision:.2f} / {_eval_record.evaluation_result.recall:.2f}'  # precision / recall
                _error_label = _eval_record.ground_truth_setting.replace("_", " ")
                _seq_stack_latex_table_error_type.append(dict(
                    approach_name=_eval_record.approach_name,
                    approach_name_shown=d_method_shown_definition.get(_eval_record.approach_name, None),
                    error_type=_error_label,
                    value=_value_string))
            # end for
            # adding the count record
            for _error_label, _count in d_error_label2count.items():
                _error_label = _error_label.replace("_", " ")
                _seq_stack_latex_table_error_type.append(dict(
                    approach_name='count',
                    approach_name_shown='count',
                    error_type=_error_label,
                    value=str(_count)))
            # end for
        
            _df_error_source = pd.DataFrame(_seq_stack_latex_table_error_type)
            _df_error_export = _df_error_source.pivot(index='approach_name_shown', columns='error_type', values='value')
            _df_error_export.fillna(' ', inplace=True)

            # to latex.
            path_table_error_latex = path_subdir_error_type / 'evaluation_table.tex'
            with path_table_error_latex.open('w') as f:
                f.write(_df_error_export.to_latex(index=True, escape=False))
            # end with

            # export to tsv.
            path_table_error_tsv = path_subdir_error_type / 'evaluation_table.tsv'
            _df_error_export.to_csv(path_table_error_tsv, sep='\t')

            # plot barplot.
            # TODO: later.

            # ----------------------------------------------------------------------
            # Evaluation per Lang Type (Approach-name * Lang-Pair)
            path_subdir_lang_pair = path_submission_data_dir / 'lang_pair'
            path_subdir_lang_pair.mkdir(parents=True, exist_ok=True)

            _seq_stack_latex_table_lang_pair = []

            _seq_lang_pair = sorted(list(set(dict_sent_id2lang_tuple.values())))
            # count the pair
            dict_langpair2_count = dict(collections.Counter(dict_sent_id2lang_tuple.values()))

            _t_lang_pair: ty.Tuple[str, str]
            for _t_lang_pair in _seq_lang_pair:
                _value_lang_pair = f'{_t_lang_pair[0].replace("_", "-")} to {_t_lang_pair[1].replace("_", "-")}'
                _seq_eval_record_error_type = self._conduct_evaluation(
                    seq_ground_truth_settings=['hallucination'],  # fixing the error-type into the hallucination.
                    dict_evaluations=dict_evaluations,
                    source_lang=_t_lang_pair[0],
                    target_lang=_t_lang_pair[1],
                    dict_sent_id2lang_tuple=dict_sent_id2lang_tuple)
                # filter the approach using `d_method_shown_definition`
                _seq_eval_record_error_type = [
                    _record for _record in _seq_eval_record_error_type 
                    if _record.approach_name in d_method_shown_definition.keys()]
                
                for _eval_obj in _seq_eval_record_error_type:
                    _value_eval = f'{_eval_obj.evaluation_result.precision:.2f} / {_eval_obj.evaluation_result.recall:.2f}' # precision / recall
                    _seq_stack_latex_table_lang_pair.append(dict(
                        language_pair=_value_lang_pair,
                        approach_name=_eval_obj.approach_name,
                        approach_name_shown=d_method_shown_definition.get(_eval_obj.approach_name, None),
                        error_type='hallucination',
                        value=_value_eval
                    ))
                # end for
                _seq_stack_latex_table_lang_pair.append(dict(
                        language_pair=_value_lang_pair,
                        approach_name='count',
                        approach_name_shown='count',
                        error_type='hallucination',
                        value=dict_langpair2_count.get(_t_lang_pair, 0)
                ))
            # end for

            _df_lang_pair_source = pd.DataFrame(_seq_stack_latex_table_lang_pair)
            df_lang_pair_latex = _df_lang_pair_source.pivot(index='language_pair', columns='approach_name_shown', values='value')
            # bringing `count` into the first element.
            _seq_current_column_header = list(df_lang_pair_latex.columns)
            _seq_current_column_header.remove('count')
            _seq_current_column_header.insert(0, 'count')
            df_lang_pair_latex = df_lang_pair_latex[_seq_current_column_header]
            
            df_lang_pair_latex.fillna(' ', inplace=True)  # replacing NA

            # to latex
            path_table_lang_latex = path_subdir_lang_pair / 'evaluation_table.tex'
            with path_table_lang_latex.open('w') as f:
                f.write(df_lang_pair_latex.to_latex(index=True, escape=False))
            # end with

            # export to tsv
            path_tbale_lang_tsv = path_subdir_lang_pair / 'evaluation_table.tsv'
            df_lang_pair_latex.to_csv(path_tbale_lang_tsv, sep='\t')

            # plot table.
            # TODO: later.


        if self.dataset_type == "halomi":
            halomi_dataset()

    # def _write_out_detection_example_submission_data_halomi_dataset(self,
    #                                                  path_output_dir: Path,
    #                                                  dict_evaluations: ty.Dict[str, ty.Dict[str, int]],
    #                                                  dict_aggkey2db_primary_keys: ty.Dict[str, ty.List[str]],
    #                                                  language_pair_preference: ty.Optional[ty.Tuple[str, str]] = None
    #                                                  ):
    #     """

    #     Args:
    #         - dict_evaluations: {approach-name: {sentence-id: 0 or 1}}.
    #     """
    #     # ------------------------------------------------
    #     # making the keys to be shown
    #     seq_targets_shown_conditions = [
    #         generate_eval_key_mmd_trajectory_flagger._KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3(
    #             tau_sequence='[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]',
    #             n_sampling=25,
    #             mode_vector_preprocess='avg',
    #             mode_target_embedding_layer='decoder.word_embedding',
    #             mode_max_token_length_vector_concat='fixed',
    #             mode_max_token_length=10,
    #             args_kernel_options_json='',  # wild card
    #             args_trajectory_options_json=''  # wild card
    #         ),
    #         generate_eval_key_mmd_trajectory_flagger._KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3(
    #             tau_sequence='[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]',
    #             n_sampling=25,
    #             mode_vector_preprocess='concat',
    #             mode_target_embedding_layer='decoder.word_embedding',
    #             mode_max_token_length_vector_concat='fixed',
    #             mode_max_token_length=10,
    #             args_kernel_options_json='',  # wild card
    #             args_trajectory_options_json=''  # wild card
    #         )            
    #     ]
    #     seq_targets_shown_conditions_refex = [_t.generate_code_name() for _t in seq_targets_shown_conditions]
    #     # ------------------------------------------------

    #     # d_method_shown_definition = {
    #     #     # "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]_decoder.embed_tokens": "MMD-Hal-Flagger",
    #     #     "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.4, 0.8, 1.0, 1.2]_decoder.embed_tokens": "MMD-Hal-Flagger",
    #     # }

    #     d_method_comparison = {
    #         "Raunak2021": "TNG",
    #         "Guerreiro2023McDSIM": "MC-DSIM",
    #         "Guerreiro2023SeqLogProb": "SeqLogProb",
    #     }  # these approaches are refered to get "mmd-hal-flagger" can get, but others not.


    #     # I create a dict of the reference data (reserved to halomi)
    #     d_sent_id2record_obj: ty.Dict[str, HalomiDatasetRecord] = {_r_obj.key_unique: _r_obj for _r_obj in self.seq_dataset_record}  # type: ignore

    #     dict_hash_name2approach_name_key = {}  # {hash-name: original-code-name}

    #     _approach_name_key: str
    #     for _approach_name_key in tqdm.tqdm(dict_evaluations.keys(), desc="Plot Example MMD Trajectory"):
    #         # ------------------------------------------------
    #         # sub procedure of matching code name
    #         _is_condition_applied = False
    #         for _regex in seq_targets_shown_conditions_refex:
    #             # tmp command: deleting '--' at the suffix. 2025-05
    #             _regex = _regex.replace('--', '')

    #             # _is_matched = re.match(rf'{__regex}', _approach_name_key_regex)
    #             _is_condition_applied = _regex in _approach_name_key
    #             if _is_condition_applied:
    #                 break
    #             # end if
    #         # end for
    #         # ------------------------------------------------

    #         if _is_condition_applied is False:
    #             continue
    #         # end if

    #         file_name_hash_suffix = hashlib.md5(_approach_name_key.encode()).hexdigest()
    #         dict_hash_name2approach_name_key[file_name_hash_suffix] = _approach_name_key
    #         _path_subdir_approach_name = path_output_dir / file_name_hash_suffix
    #         _path_subdir_approach_name.mkdir(parents=True, exist_ok=True)

    #         # I classify the data into 4 cases, TP, FP, FN, TN. 
    #         # I down-sample into the specified numbers.

    #         # _path_subdir_approach_name = path_output_dir / _approach_name_key
    #         # _path_subdir_approach_name.mkdir(parents=True, exist_ok=True)

    #         # # I classify the data into 4 cases, TP, FP, FN, TN. 
    #         # # I down-sample into the specified numbers.

    #         # # exporting the True-Positive cases.
    #         # _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'true-positive'
    #         # _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
            
    #         _dict_args_common = dict(
    #             approach_code_name=_approach_name_key,
    #             dataset_name=self.dataset_type,
    #             d_sent_id2record_obj=d_sent_id2record_obj,
    #             dict_evaluations=dict_evaluations,
    #             d_method_comparison=d_method_comparison,
    #             db_handler=self.db_handler,
    #             language_pair_preference=language_pair_preference,
    #             dict_aggkey2db_primary_keys=dict_aggkey2db_primary_keys
    #         )
            
    #         plot_mmd_trajectory.main_procedure(
    #             **_dict_args_common,  # type: ignore
    #             selection_mode='true-positive',
    #             path_subdir_selection_mode=_path_subdir_approach_name_selection_mode
    #         )
    #         plot_mmd_trajectory.main_procedure(
    #             **_dict_args_common,  # type: ignore
    #             selection_mode='true-positive',
    #             path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)

    #         # exporting the False-Positive cases.
    #         _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'false-positive'
    #         _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
    #         plot_mmd_trajectory.main_procedure(
    #             **_dict_args_common,  # type: ignore
    #             selection_mode='false-positive',
    #             path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)

    #         # exporting the False-Negative cases.
    #         _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'false-negative'
    #         _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
    #         plot_mmd_trajectory.main_procedure(
    #             **_dict_args_common,  # type: ignore
    #             selection_mode='false-negative',
    #             path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)

    #         # exporting the True-Negative cases (correct translation).
    #         _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'true-negative'
    #         _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
    #         plot_mmd_trajectory.main_procedure(
    #             **_dict_args_common,  # type: ignore
    #             selection_mode='true-negative',
    #             path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)
        
    #         # ---- Procedure for the cases that "MMD-Hal-Falgger" only can detect. -----
    #         # I set the target approach.
    #         # I collect true-positive sentence-ids from the other methods. Refer `d_method_comparison`.
    #         # The target is The sentence-id that MMD-Hal-Falgger. Hint: Use set operation.
    #         _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'prefer-mmd-hal-flagger-tp'
    #         _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
    #         plot_mmd_trajectory.main_procedure(
    #             **_dict_args_common,  # type: ignore
    #             selection_mode='prefer-mmd-hal-flagger-tp',
    #             path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)
    #     # end for


    def _write_out_detection_example_submission_data(self,
                                                     path_output_dir: Path,
                                                     dict_evaluations: ty.Dict[str, ty.Dict[str, int]],
                                                     dict_code_name2agg_key_tuple: ty.Dict[str, generate_eval_key_mmd_trajectory_flagger._KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3],
                                                     dict_aggkey2db_primary_keys: ty.Dict[str, ty.List[str]],
                                                     path_dir_cache_translation: Path,
                                                     language_pair_preference: ty.Optional[ty.Tuple[str, str]] = None
                                                     ):
        """

        Args:
            - dict_evaluations: {approach-name: {sentence-id: 0 or 1}}.
        """

        # ------------------------------------------------
        # making the keys to be shown
        seq_targets_shown_conditions = [
            generate_eval_key_mmd_trajectory_flagger._KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3(
                tau_sequence='[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]',
                n_sampling=25,
                mode_vector_preprocess='avg',
                mode_target_embedding_layer='decoder.word_embedding',
                mode_max_token_length_vector_concat='',  # wild card
                mode_max_token_length='',  # wild card
                args_kernel_options_json='',  # wild card
                args_trajectory_options_json=''  # wild card
            ),
            generate_eval_key_mmd_trajectory_flagger._KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3(
                tau_sequence='[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]',
                n_sampling=25,
                mode_vector_preprocess='concat',
                mode_target_embedding_layer='decoder.word_embedding',
                mode_max_token_length_vector_concat='',  # wild card
                mode_max_token_length='',  # word card
                args_kernel_options_json='',  # wild card
                args_trajectory_options_json=''  # wild card
            )            
        ]

        conditions_extraction_kernel = {
            "kernel_type": "gaussian",
            "kernel_gaussian_length_scale": 25
        }

        seq_targets_shown_conditions_refex = [_t.generate_code_name() for _t in seq_targets_shown_conditions]
        # ------------------------------------------------

        # d_method_shown_definition = {
        #     # "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]_decoder.embed_tokens": "MMD-Hal-Flagger",
        #     "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.4, 0.8, 1.0, 1.2]_decoder.embed_tokens": "MMD-Hal-Flagger",
        # }

        d_method_comparison = {
            "Raunak2021": "TNG",
            "Guerreiro2023McDSIM": "MC-DSIM",
            "Guerreiro2023SeqLogProb": "SeqLogProb",
        }  # these approaches are refered to get "mmd-hal-flagger" can get, but others not.


        # I create a dict of the reference data (reserved to halomi)
        if self.dataset_type == 'halomi':
            d_sent_id2record_obj: ty.Dict[str, HalomiDatasetRecord] = {_r_obj.key_unique: _r_obj for _r_obj in self.seq_dataset_record}  # type: ignore
        elif self.dataset_type == 'lfan_hall':
            d_sent_id2record_obj: ty.Dict[str, WMTDatasetRecord] = {_r_obj.sentence_id: _r_obj for _r_obj in self.seq_dataset_record}  # type: ignore
        else:
            raise NotImplementedError()
        # end if

        dict_hash_name2approach_name_key = {}  # {hash-name: original-code-name}

        _approach_name_key: str
        for _approach_name_key in tqdm.tqdm(dict_evaluations.keys(), desc="Plot Example MMD Trajectory"):
            # ------------------------------------------------
            # sub procedure of matching code name
            _is_condition_applied = False
            for _regex in seq_targets_shown_conditions_refex:
                # tmp command: deleting '--' at the suffix. 2025-05
                # _regex = _regex.replace('--', '')
                _regex = re.sub(r'-{2,}', '', string=_regex)

                # _is_matched = re.match(rf'{__regex}', _approach_name_key_regex)
                _is_condition_applied = _regex in _approach_name_key
                if _is_condition_applied:
                    break
                # end if
            # end for

            # check the condition of the kernel argument.
            _agg_object = dict_code_name2agg_key_tuple[_approach_name_key]
            _args_kernel_obj = json.loads(_agg_object.args_kernel_options_json)
            for _key_kernel_condition, _value_kernel_condition in conditions_extraction_kernel.items():
                if _args_kernel_obj[_key_kernel_condition] != _value_kernel_condition:
                    _is_condition_applied = False
                # end if
            # end for
            # ------------------------------------------------

            if _is_condition_applied is False:
                continue
            # end if

            file_name_hash_suffix = hashlib.md5(_approach_name_key.encode()).hexdigest()
            dict_hash_name2approach_name_key[file_name_hash_suffix] = _approach_name_key
            _path_subdir_approach_name = path_output_dir / file_name_hash_suffix
            _path_subdir_approach_name.mkdir(parents=True, exist_ok=True)

            # I classify the data into 4 cases, TP, FP, FN, TN. 
            # I down-sample into the specified numbers.

            
            _dict_args_common = dict(
                approach_code_name=_approach_name_key,
                dataset_name=self.dataset_type,
                d_sent_id2record_obj=d_sent_id2record_obj,
                dict_evaluations=dict_evaluations,
                d_method_comparison=d_method_comparison,
                db_handler=self.db_handler,
                language_pair_preference=language_pair_preference,
                dict_aggkey2db_primary_keys=dict_aggkey2db_primary_keys,
                path_dir_cache_translation=path_dir_cache_translation
            )

            # exporting the True-Positive cases.
            _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'true-positive'
            _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)

            plot_mmd_trajectory.main_procedure(
                **_dict_args_common,  # type: ignore
                selection_mode='true-positive',
                path_subdir_selection_mode=_path_subdir_approach_name_selection_mode
            )

            # exporting the False-Positive cases.
            _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'false-positive'
            _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
            plot_mmd_trajectory.main_procedure(
                **_dict_args_common,  # type: ignore
                selection_mode='false-positive',
                path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)

            # exporting the False-Negative cases.
            _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'false-negative'
            _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
            plot_mmd_trajectory.main_procedure(
                **_dict_args_common,  # type: ignore
                selection_mode='false-negative',
                path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)

            # exporting the True-Negative cases (correct translation).
            _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'true-negative'
            _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
            plot_mmd_trajectory.main_procedure(
                **_dict_args_common,  # type: ignore
                selection_mode='true-negative',
                path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)
        
            # ---- Procedure for the cases that "MMD-Hal-Falgger" only can detect. -----
            # I set the target approach.
            # I collect true-positive sentence-ids from the other methods. Refer `d_method_comparison`.
            # The target is The sentence-id that MMD-Hal-Falgger. Hint: Use set operation.
            _path_subdir_approach_name_selection_mode = _path_subdir_approach_name / 'prefer-mmd-hal-flagger-tp'
            _path_subdir_approach_name_selection_mode.mkdir(parents=True, exist_ok=True)
            plot_mmd_trajectory.main_procedure(
                **_dict_args_common,  # type: ignore
                selection_mode='prefer-mmd-hal-flagger-tp',
                path_subdir_selection_mode=_path_subdir_approach_name_selection_mode)
        # end for

        _path_hash_id2approach_name = path_output_dir / 'hash_name2approach_name_key.json'
        with _path_hash_id2approach_name.open('w') as f:
            f.write(json.dumps(dict_hash_name2approach_name_key, ensure_ascii=False, indent=4))
        # end with

    # def _make_heatmap_mmd_flagger_ver1(self,
    #                                    path_output_dir: Path,
    #                                    seq_evaluation_records_container: ty.List[EvaluationRecord],
    #                                    _dict_parameters: ty.Dict[str, ty.Tuple[float, float]]
    #                                    ) -> None:
    #     """A deserved method for MMD Flagger Ver1."""
    #     # filtering the dict keys by the approach name.
    #     _dict_parameters = {_k: _v for _k, _v in _dict_parameters.items() if "MmdFlaggerVer1" in _k}
    #     if len(_dict_parameters) == 0:
    #         return None
    #     # end if

    #     # generating the heatmap
    #     path_dir_output_heatmap = path_output_dir / "mmd_flagger_heatmap"
    #     path_dir_output_heatmap.mkdir(parents=True, exist_ok=True)
    #     self._write_out_mmd_flagger_heatmap(
    #         seq_evaluation_records=seq_evaluation_records_container,
    #         dict_approach2parameters=_dict_parameters,
    #         path_dir_output_heatmap=path_dir_output_heatmap,
    #         target_score="recall")
    #     self._write_out_mmd_flagger_heatmap(
    #         seq_evaluation_records=seq_evaluation_records_container,
    #         dict_approach2parameters=_dict_parameters,
    #         path_dir_output_heatmap=path_dir_output_heatmap,
    #         target_score="precision")
    #     self._write_out_mmd_flagger_heatmap(
    #         seq_evaluation_records=seq_evaluation_records_container,
    #         dict_approach2parameters=_dict_parameters,
    #         path_dir_output_heatmap=path_dir_output_heatmap,
    #         target_score="f1")

    # def _make_precision_recall_curve_mmd_flagger_ver1(self,
    #                                                   path_output_dir: Path,
    #                                                   seq_evaluation_records_container: ty.List[EvaluationRecord],
    #                                                   _dict_parameters: ty.Dict[str, ty.Tuple[float, float]]
    #                                                   ) -> None:
    #     """Generating the precision-recall curve for MMD Flagger Ver1.
        
    #     This method plots the precision-recall curve. 
    #     """
    #     path_dir_output_heatmap = path_output_dir / "mmd_flagger_precision_recall_curve"
    #     path_dir_output_heatmap.mkdir(parents=True, exist_ok=True)

    #     # generating the precision-recall curve.
    #     seq_keys_mmd_flaggers = list(_dict_parameters.keys())
    #     seq_evaluation_mmd_flaggers = [rec for rec in seq_evaluation_records_container if rec.approach_name in seq_keys_mmd_flaggers]
    #     assert len(seq_evaluation_mmd_flaggers) > 0, "No records found."

    #     seq_precision = []
    #     seq_recall = []
    #     for _eval_record in seq_evaluation_mmd_flaggers:
    #         _precision = _eval_record.evaluation_result.precision
    #         _recall = _eval_record.evaluation_result.recall
    #         seq_precision.append(_precision)
    #         seq_recall.append(_recall)
    #     # end for

    #     # Note: I can not compute the Area-Under-Curve because the precision is not
    #     # # computing the Area-Under-Curve.
    #     # precision = np.array(seq_precision)
    #     # recall = np.array(seq_recall)
    #     # # score_auc = auc(precision, recall)
    #     # score_auc = 0.0

    #     # I want to visualise X: recall, Y: precision.
    #     f, ax = plot.subplots()
    #     sns.lineplot(x=seq_recall, y=seq_precision, ax=ax)
    #     ax.set_title(f"Precision-Recall Curve.")
    #     ax.set_xlabel("Recall")
    #     ax.set_ylabel("Precision")
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    #     ax.legend(loc='lower left')
    #     # __path_save_heatmap = path_visualisation / "heatmap_recall_score.png"
    #     f.savefig((path_dir_output_heatmap / "precision_recall_curve.png").as_posix())

    def _make_threshold_plots_for_baselines(self,
                                            path_output_dir: Path,
                                            config_name: str,
                                            dataset_type: str,
                                            setting_ground_truth: str = "hallucination",
                                            num_threshold: int = 100,
                                            ) -> None:
        """I want to plot 2D line-plot that shows the relationship between the threshold and the metrics.
        I design this method for the baselines.
        
        I have to fetch all evaluation records of the baselines since I need a score of the record.
        """
        def _extract_score_sequence(seq_db_records: ty.List[ty.Dict], score_column_name: str) -> npt.NDArray[np.float32]:
            """Extracting the score sequence from the database records."""
            seq_scores = []
            for _record in seq_db_records:
                assert score_column_name in dict(_record), f"Invalid record. {score_column_name} not found: {_record}"
                _score = float(_record[score_column_name])
                seq_scores.append(_score)
            # end for
            return np.array(seq_scores, dtype=np.float32)
        # end def

        def _extract_score_threshold(seq_db_records: ty.List[ty.Dict], threshold_column_name: str) -> ty.List[float]:
            __ = []
            for _record in seq_db_records:
                assert threshold_column_name in dict(_record), f"Invalid record. {threshold_column_name} not found: {_record}"
                _threshold = float(_record[threshold_column_name])
                __.append(_threshold)
            # end for

            return list(sorted(list(set(__))))
            # assert len(set(__)) == 1, f"Invalid threshold values: {__}"
            # return __[0]

        def _make_flag(record: ty.Dict, threshold: float, score_column_name: str, is_smaller_criteria: bool) -> int:
            """Putting a flag for the record."""
            assert score_column_name in dict(record), f"Invalid record. {score_column_name} not found: {record}"
            _score = float(record[score_column_name])
            if is_smaller_criteria:
                if _score <= threshold:
                    return 1
                else:
                    return 0
            else:
                if _score >= threshold:
                    return 1
                else:
                    return 0
        # end def

        def _get_ground_truth(record: ty.Dict, dict_ground_truth: ty.Dict[str, int]) -> int:
            """Getting the ground truth label."""
            assert "sentence_id" in dict(record), f"Invalid record. 'sentence_id' not found: {record}"
            _sentence_id = str(record["sentence_id"])
            assert _sentence_id in dict_ground_truth, f"Invalid sentence-id: {_sentence_id}"

            return dict_ground_truth[_sentence_id]
        # end def

        def _func_key_sort_record(d: ty.Dict) -> str:
            """Sorting key function for the database record."""
            assert "sentence_id" in dict(d), f"Invalid record. 'sentence_id' not found: {d}"
            return str(d["sentence_id"])
        # end def


        path_dir_output_plot = path_output_dir / "baselines_threshold_plots"
        path_dir_output_plot.mkdir(parents=True, exist_ok=True)

        TARGET_TABLES = [
            DbTableRecordGuerreiro2023McDSIM.__name__,
            DbTableRecordGuerreiro2023SeqLogProb.__name__,
        ]

        dict_ground_truth = get_ground_truth_label._get_ground_truth_labels(
            seq_dataset_record=self.seq_dataset_record,
            dataset_name=self.dataset_type,
            label_setting=setting_ground_truth)

        for _table_name in TARGET_TABLES:
            # I use this stack to store the evaluation records (with dict, and I convert it into pandas DF later).
            _stack_eval_record: ty.List[ty.Dict] = []

            _score_field_name = "score" if _table_name == DbTableRecordGuerreiro2023McDSIM.__name__ else "log_probability"
            _threshold_file_name = "score_threshold" if _table_name == DbTableRecordGuerreiro2023McDSIM.__name__ else "log_probability_threshold"
            _is_smaller_criteria = True if _table_name in (DbTableRecordGuerreiro2023McDSIM.__name__, DbTableRecordGuerreiro2023SeqLogProb.__name__) else True

            _path_subdir_table_nane = path_dir_output_plot / _table_name
            _path_subdir_table_nane.mkdir(parents=True, exist_ok=True)

            _seq_records = self._fetch_prediction_results(config_name=config_name, eval_table_name=_table_name)
            seq_records = sorted(_seq_records, key=_func_key_sort_record)
            seq_ground_truth_label = [_get_ground_truth(_r, dict_ground_truth) for _r in seq_records]

            if len(seq_records) == 0:
                logger.warning(f"No records found for {config_name} in {_table_name}.")
                continue
            # end if

            # I want to use the threshold for the visualisation.
            _score_threshold_possible = _extract_score_threshold(seq_db_records=seq_records, threshold_column_name=_threshold_file_name)

            if dataset_type == "lfan_hall":
                assert len(_score_threshold_possible) == 1, f"Invalid threshold values: {_score_threshold_possible}"
            else:
                # making a tag of language pair
                raise NotImplementedError()
            # end if

            for __score_threshold in _score_threshold_possible:
                # TODO I have to select records having the threshold.
                # TODO I have to confirm the len(records) > 0

                # I obtain min and max of the threshold values.
                _array_scores = _extract_score_sequence(seq_db_records=seq_records, score_column_name=_score_field_name)
                _min_score = np.min(_array_scores)
                _max_score = np.max(_array_scores)
                
                # I create a sequence of threshold values.
                _seq_threshold = np.linspace(_min_score, _max_score, num=num_threshold)

                # for-loop over the threshold values.
                for _threshold in _seq_threshold:
                    # I compute Precision, Recall, F at the threshold.
                    __seq_prediction_truth = [_make_flag(_r, _threshold, score_column_name=_score_field_name, is_smaller_criteria=_is_smaller_criteria) for _r in seq_records]

                    # evaluation values.
                    __array_prediction = np.array(__seq_prediction_truth, dtype=np.int8)
                    __array_ground_truth = np.array(seq_ground_truth_label, dtype=np.int8)
                    _eval_result = self._func_get_evaluation(array_label_prediction=__array_prediction,
                                                            array_label_ground_truth=__array_ground_truth)

                    # creating an evaluation record.
                    _eval_record = dict(threshold=_threshold,
                                        n_total=_eval_result.n_total,
                                        precision=_eval_result.precision,
                                        recall=_eval_result.recall,
                                        f1=_eval_result.f1)
                    _stack_eval_record.append(_eval_record)
                # end for
                
                # I want to visualise X: threshold, Y: metrics.
                _df_eval = pd.DataFrame(_stack_eval_record)
                for _metric_name in ["precision", "recall", "f1"]:
                    _path_output_graph = _path_subdir_table_nane / f"{_metric_name}_threshold_plot.png"
                    fig, ax = plot.subplots()
                    sns.lineplot(x="threshold", y=_metric_name, data=_df_eval, ax=ax)

                    ax.axvline(x=__score_threshold, color='red', linestyle='--', label='Threshold')
                    
                    ax.set_title(f"{_metric_name.capitalize()} vs Threshold")
                    ax.set_xlabel("Threshold")
                    ax.set_ylabel(_metric_name.capitalize())

                    fig.savefig(_path_output_graph.as_posix(), bbox_inches='tight', dpi=300)
                    logger.info(f"Saved the threshold plot to {_path_output_graph}")
                # end for

                # --------------------------------------------------------------------------------------------
                # plotting the relationship between the threshold and the metrics.
                _path_out_score_distribution = _path_subdir_table_nane / f"plot_score_distribution.png"
                fig, ax = plot.subplots()
                sns.histplot(_array_scores, bins=50, ax=ax)
                ax.set_title(f"Score Distribution")
                ax.axvline(x=__score_threshold, color='red', linestyle='--', label='Threshold')
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                fig.savefig(_path_out_score_distribution.as_posix(), bbox_inches='tight', dpi=300)

                # --------------------------------------------------------------------------------------------
                # evaluation values.
                _array_prediction_score = np.array(_array_scores, dtype=np.float32)
                _array_ground_truth = np.array(seq_ground_truth_label, dtype=np.int8)

                # computing AUC-PR.
                _roc_auc_score = roc_auc_score(
                    y_true=_array_ground_truth,
                    y_score=_array_prediction_score,
                )

                # ploting precall-recall curve.
                _path_out_p_r_curve = _path_subdir_table_nane / f"precision-recall-curve.png"
                fig, ax = plot.subplots()
                PrecisionRecallDisplay.from_predictions(
                    y_true=_array_ground_truth,
                    y_pred=_array_prediction_score,
                    ax=ax,
                    name="Precision-Recall Curve",)
                ax.set_title(f"Precision-Recall Curve (AUC = {round(_roc_auc_score, 4)})")
                fig.savefig(_path_out_p_r_curve.as_posix(), bbox_inches='tight', dpi=300)
        # end for


    def make_metric_graphs(self, 
                           path_output_dir: Path,
                           seq_evaluation_records_container: ty.List[EvaluationRecord]) -> None:
        """Visualising the metrics."""

        # visualising metrics graphs ex. F1, Precision, Recall
        seq_metrics = ["precision", "recall", "f1"]
        target_error_setting = ["hallucination"]
        
        
        for __key_metric in seq_metrics:
            for __error_setting in target_error_setting:
                path_sub_dir_output_dir = path_output_dir / __error_setting
                path_sub_dir_output_dir.mkdir(parents=True, exist_ok=True)

                seq_df_record_obj = []
                for _eval_record in seq_evaluation_records_container:
                    if _eval_record.ground_truth_setting != __error_setting:
                        _approach_name = _eval_record.approach_name
                        _eval_result = _eval_record.evaluation_result

                        _value_metric = getattr(_eval_result, seq_metrics[0])
                        __df_record_obj = dict(approach_name=_approach_name, value_metric=_value_metric)
                        seq_df_record_obj.append(__df_record_obj)
                # end for
                assert len(seq_df_record_obj) > 0, "No records found."
                _df_metric = pd.DataFrame(seq_df_record_obj)

                # visualising the metrics.
                # plotting the metrics.
                _path_output_graph = path_sub_dir_output_dir / f"metrics_{__key_metric}.png"
                fig, ax = plot.subplots()
                sns.barplot(x="approach_name", y="value_metric", data=_df_metric, ax=ax)
                ax.set_title(f"{__key_metric.capitalize()}")
                # I rotate the x-axis labels.
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

                fig.savefig(_path_output_graph.as_posix(), bbox_inches='tight', dpi=300)
                logger.info(f"Saved the metrics graph to {_path_output_graph}")

    def main(self,
             path_output_dir: Path,
             config_name: str,
             seq_eval_table_name: ty.List[str],
             path_dir_cache_translation: Path,
             seq_ground_truth_settings: ty.Optional[ty.List[str]] = None,
             is_do_submission_data_generation: bool = True,
             is_do_write_analysis_record_excel: bool = True
             ) -> None:
        """Main function to run the evaluation."""
        if seq_ground_truth_settings is None:
            if self.dataset_type == "halomi":
                seq_ground_truth_settings = list(get_ground_truth_label.DEFAULT_GROUND_TRUTH_SETTINGS)
            elif self.dataset_type == "lfan_hall":
                seq_ground_truth_settings = list(set(get_ground_truth_label.DEFAULT_GROUND_TRUTH_SETTINGS) - {"2_Small_hallucination", "3_Partial_hallucination", "4_Full_hallucination"})
            else:
                raise ValueError(f'Not defined.')
        # end if

        # pre-processing the prediction results.
        dict_evaluations = {}
        for _eval_table_name in seq_eval_table_name:
            if _eval_table_name == DbTableRecordRaunak2021.__name__:
                logger.warning(f"Skipping. I will do it later.")
                _dict_predictions = self._prep_baselines(config_name=config_name, table_name=DbTableRecordRaunak2021.__name__, eval_key_name="Raunak2021")
            elif _eval_table_name == DbTableRecordGuerreiro2023McDSIM.__name__:
                logger.warning(f"Skipping. I will do it later.")
                _dict_predictions = {}
                # _dict_predictions = self._prep_baselines(config_name=config_name, table_name=DbTableRecordGuerreiro2023McDSIM.__name__, eval_key_name="Guerreiro2023McDSIM")
            elif _eval_table_name == DbTableRecordGuerreiro2023SeqLogProb.__name__:
                _dict_predictions = self._prep_baselines(config_name=config_name, table_name=DbTableRecordGuerreiro2023SeqLogProb.__name__, eval_key_name="Guerreiro2023SeqLogProb")
            elif _eval_table_name == DbTableRecordProposalMmdFlaggerTrajectoryVer3.__name__:
                _dict_predictions, _dict_code_name2agg_key_tuple, _dict_aggkey2db_primary_keys = self._prep_mmd_trajectory_flagger_ver3(config_name=config_name)
            else:
                raise ValueError(f"Unsupported table name: {_eval_table_name}")
            # end if
            dict_evaluations.update(_dict_predictions)
        # end for

        # making the dict of {sent-id: (src-lang, tgt-lang)}
        if self.dataset_type == "halomi":
            dict_sent_id2lang_tuple = self._fetch_language_pair_labels()
        elif self.dataset_type == "lfan_hall":
            dict_sent_id2lang_tuple = {_sent_id: ('deu-Latn', 'eng-Latn') for _sent_id in _dict_predictions.keys()}
        else:
            raise ValueError()
        # end 

        seq_evaluation_records_container = self._conduct_evaluation(
            seq_ground_truth_settings=seq_ground_truth_settings,
            dict_evaluations=dict_evaluations)


        # writing out the evaluation results.
        path_output_book = path_output_dir / "evaluation_results.xlsx"
        self._write_out_evaluation_excel_book(
            path_excel=path_output_book,
            seq_evaluation_records=seq_evaluation_records_container)

        # detailed analysis data, mainly designed for the draft.
        if is_do_submission_data_generation:
            path_submission_table_data = path_output_dir / "submission_table_data"
            path_submission_table_data.mkdir(parents=True, exist_ok=True)
            self._write_out_evaluation_submission_data(
                path_submission_data_dir=path_submission_table_data,
                seq_evaluation_records_container=seq_evaluation_records_container,
                dict_evaluations=dict_evaluations,
                dict_sent_id2lang_tuple=dict_sent_id2lang_tuple)
            
            if self.dataset_type == "halomi":
                # 
                path_xlsx_file = path_output_dir / "evaluation_results_language_pairs.xlsx"
                self._conduct_and_write_out_evaluation_language_pairs(
                    path_xlsx_file=path_xlsx_file,
                    dict_evaluations=dict_evaluations,
                    dict_sent_id2lang_tuple=dict_sent_id2lang_tuple
                )

                # selecting translation examples and plotting the MMD-trajectories.
                path_submission_detection_examples = path_output_dir / "submission_detection_examples"
                path_submission_detection_examples.mkdir(parents=True, exist_ok=True)
                self._write_out_detection_example_submission_data(
                    path_output_dir=path_submission_detection_examples,
                    dict_evaluations=dict_evaluations,
                    dict_code_name2agg_key_tuple=_dict_code_name2agg_key_tuple,
                    dict_aggkey2db_primary_keys=_dict_aggkey2db_primary_keys,
                    path_dir_cache_translation=path_dir_cache_translation
                )
                self._write_out_detection_example_submission_data(
                    path_output_dir=path_submission_detection_examples,
                    dict_evaluations=dict_evaluations,
                    dict_code_name2agg_key_tuple=_dict_code_name2agg_key_tuple,
                    dict_aggkey2db_primary_keys=_dict_aggkey2db_primary_keys,
                    language_pair_preference=('deu_Latn', 'eng_Latn'),
                    path_dir_cache_translation=path_dir_cache_translation
                )
                self._write_out_detection_example_submission_data(
                    path_output_dir=path_submission_detection_examples,
                    dict_evaluations=dict_evaluations,
                    dict_code_name2agg_key_tuple=_dict_code_name2agg_key_tuple,
                    dict_aggkey2db_primary_keys=_dict_aggkey2db_primary_keys,
                    language_pair_preference=('eng_Latn', 'deu_Latn'),
                    path_dir_cache_translation=path_dir_cache_translation
                )
            elif self.dataset_type == 'lfan_hall':
                logger.info('Generating Submission Data for lfan_hall...')
                path_submission_detection_examples = path_output_dir / "submission_detection_examples"
                path_submission_detection_examples.mkdir(parents=True, exist_ok=True)
                self._write_out_detection_example_submission_data(
                    path_output_dir=path_submission_detection_examples,
                    dict_evaluations=dict_evaluations,
                    dict_code_name2agg_key_tuple=_dict_code_name2agg_key_tuple,
                    dict_aggkey2db_primary_keys=_dict_aggkey2db_primary_keys,
                    path_dir_cache_translation=path_dir_cache_translation,)
                logger.info('Done Generating Submission Data for lfan_hall...')            
            else:
                logger.debug("SKIP")
                
        if is_do_write_analysis_record_excel:
            # making an analysis excel book.
            path_output_analysis_book = path_output_dir.parent / "analysis_results.xlsx"
            self._write_out_analysis_excel_book(
                path_excel=path_output_analysis_book,
                dict_evaluations=dict_evaluations)
            # end

