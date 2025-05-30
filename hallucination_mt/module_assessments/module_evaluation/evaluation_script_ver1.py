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

import torch
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay

from ...guerreiro_2023_wmt.data_models.data_models import WMTDatasetRecord
from ...guerreiro_2023_wmt.data_models.utils import load_dataset

from ..module_management_db.module_sqlite3_handler import DBHandlerExp
from ..module_management_db.module_db_record import (
    DbTableRecordRaunak2021,
    DbTableRecordProposalMmdFlaggerVer1,
    DbTableRecordProposalMmdFlaggerTrajectoryVer1,
    DbTableRecordGuerreiro2023McDSIM,
    DbTableRecordGuerreiro2023SeqLogProb
)
from ... import visualisation_header  # just loading the header file. Then, configured.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_SETTINGS = (
    "hallucination", 
    "hallucination+mt-error", 
    "mt-error", 
    "error_named_entity",
    "error_omission",
    "error_full",
    "error_strong",
    "error_repetitions")


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


class EvaluationVer1:
    def __init__(self,
                 seq_dataset_record: ty.List[WMTDatasetRecord],
                 path_prediction_database: Path) -> None:
        assert path_prediction_database.exists(), f"Database file not found: {path_prediction_database}"
        self.seq_dataset_record = seq_dataset_record
        # opening the database file.
        db_con = sqlite3.connect(str(path_prediction_database))
        self.db_handler = DBHandlerExp(path_prediction_database)
        self.db_handler.conn = db_con

    @staticmethod
    def _func_get_evaluation(
            array_label_prediction: npt.NDArray[np.int8],
            array_label_ground_truth: npt.NDArray[np.int8]
            ) -> EvaluationResultContainer:
        """An abstract function to get the evaluation metric."""
        # confusion matrix
        # computing elements of a confusion matrix.
        matrix_confusion = confusion_matrix(array_label_ground_truth, array_label_prediction)
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
            logger.debug(f"Saved the heatmap to {_path_output_heatmap}")

    def _write_out_evaluation_submission_data(self,
                                              path_output_dir: Path,
                                              dict_evaluations: ty.Dict[str, ty.Dict[int, int]],
                                              target_error_types: ty.Optional[ty.List[str]] = None,
                                              mode_processing_none_prediction: str = "zero_replacement",
                                              ) -> None:
        """I generate data for the submission.
        
        Output Files:
            Table file
                Rows of method names.
                Columns of error-types.
                Values: A format of "Precision / Recall".
            Latex table format.
                Latex table format of the table data above.
            Plot.
                X: error-type
                Y: Precision, Recall,
                Hue: method name

        Args:
            is_show_best_only: If True, I show only the best method among various options.

        """
        if target_error_types is None:
            target_error_types = ['error_full', 'error_strong', 'error_repetitions', 'error_omission', 'error_named_entity']
        # end if

        # Method names to be written out.
        # methods_shown = [
        #     "Raunak2021",
        #     "Guerreiro2023McDSIM",
        #     "Guerreiro2023SeqLogProb",
        #     "MmdTrajectoryFlaggerVer1_v1_no_filter_None_25_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]"
        # ]

        d_method_shown_definition = {
            "Raunak2021": "TNG",
            "Guerreiro2023McDSIM": "MC-DSIM",
            "Guerreiro2023SeqLogProb": "SeqLogProb",
            "MmdTrajectoryFlaggerVer1_v1_no_filter_None_25_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]": "MMD-Hal-Flagger"
        }

        assert mode_processing_none_prediction in ("zero_replacement", "drop_none"), f"Unsupported mode: {mode_processing_none_prediction}"
        # TODO The following procedure is the same as the one in the function `_conduct_evaluation`. Should be unified.
        # ------------------------------------------------------------------
        # Main Pair of calculating Evaluation Scores.
        # Evaluation values under various conditions, e.g. hallucination+MT-Error, hallucination only, full-detached only etc..
        seq_evaluation_records_container: ty.List[EvaluationRecord] = []
        for __setting_ground_truth in target_error_types:
            dict_ground_truth = self._get_ground_truth_labels(label_setting=__setting_ground_truth)
            for __eval_method_name, __dict_sentid2label in dict_evaluations.items():
                # alignment of sentence-id.                
                __sentence_id_common = sorted(list(set(__dict_sentid2label.keys()) & set(dict_ground_truth.keys())))
                assert len(__sentence_id_common) > 0, "No common sentence-ids found."
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
                try:
                    _eval_result = self._func_get_evaluation(array_label_prediction=__array_prediction,
                                                            array_label_ground_truth=__array_ground_truth)
                except Exception as e:
                    breakpoint()
                
                # creating an evaluation record.
                _eval_record = EvaluationRecord(approach_name=__eval_method_name,
                                                ground_truth_setting=__setting_ground_truth,
                                                evaluation_result=_eval_result)
                
                if __eval_method_name in d_method_shown_definition.keys():
                    seq_evaluation_records_container.append(_eval_record)
                # end if
            # end for
        # end for
        # ------------------------------------------------------------------
        assert len(seq_evaluation_records_container) > 0, "No records found."

        # making the table data object and constructing Pandas DataFrame object.
        seq_table_data = []  # {approach_name: {ground_truth_setting: table-value}}

        # adding a record of the ground-truth counts.
        for __ground_truth_label in target_error_types:
            dict_ground_truth = self._get_ground_truth_labels(label_setting=__ground_truth_label)
            _sum_count = int(sum(dict_ground_truth.values()))
            seq_table_data.append(dict(
                value=f'{_sum_count}',
                method='count',
                label=__ground_truth_label))
        # end for

        for _selected_obj in seq_evaluation_records_container:
            approach_name = _selected_obj.approach_name
            approach_name_shown = d_method_shown_definition[approach_name]

            ground_truth_setting = _selected_obj.ground_truth_setting
            precision = _selected_obj.evaluation_result.precision
            recall = _selected_obj.evaluation_result.recall

            # Format the cell as "precision / recall"
            formatted_value = f"{precision:.2f} / {recall:.2f}"
            seq_table_data.append(dict(
                value=formatted_value,
                method=approach_name_shown,
                label=ground_truth_setting))
        # end for
        _df_table_tmp = pd.DataFrame(seq_table_data)
        _df_table_tmp = _df_table_tmp.fillna("")  # Fill missing values with an empty string
        df_table = _df_table_tmp.pivot(columns='label', index='method', values='value')
        df_table = df_table[target_error_types] # re-sort the column order.

        # writing out the table.
        path_output_table_tsv = path_output_dir / "evaluation_table.tsv"
        path_output_table_tsv.write_text(df_table.to_csv(sep="\t", index=False))
        logger.debug(f"Table file is written out: {path_output_table_tsv}")

        # formatting the latex table.
        path_output_table_latex = path_output_dir / "evaluation_table.tex"
        with open(path_output_table_latex, "w") as f:
            f.write(df_table.to_latex(index=True, escape=False))
        logger.debug(f"Latex table file is written out: {path_output_table_latex}")

        # plot the bar visualisation.
        # I make two plots. One is Precision, the other is Recall.
        df_plot_source = pd.DataFrame([
            {'approach_name': _t.approach_name, 
             'approach_name_shown': d_method_shown_definition[_t.approach_name],
             'ground_truth_setting': _t.ground_truth_setting, 
             **_t.evaluation_result._asdict()} 
            for _t in seq_evaluation_records_container])
                
        # Precision Visualisation. I add legend label.
        path_output_precision = path_output_dir / "evaluation_precision.png"
        f, ax = plot.subplots(figsize=(10, 4))
        sns.barplot(data=df_plot_source, x='ground_truth_setting', y='precision', hue='approach_name_shown', ax=ax)
        ax.set_ylabel('Precision')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        # moving the legend to the upper center
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2, frameon=False)        
        f.savefig(path_output_precision.as_posix(), bbox_inches='tight', dpi=300)

        path_output_recall = path_output_dir / "evaluation_recall.png"
        f, ax = plot.subplots(figsize=(10, 4))
        sns.barplot(data=df_plot_source, x='ground_truth_setting', y='recall', hue='approach_name_shown', ax=ax)
        ax.set_ylabel('Recall')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend_.remove() # deleting the legend
        f.savefig(path_output_recall.as_posix(), bbox_inches='tight', dpi=300)

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
        logger.debug(f"Excel file is written out: {path_excel}")

    def _write_out_analysis_excel_book(self,
                                       path_excel: Path,
                                       dict_evaluations: ty.Dict[str, ty.Dict[int, int]]):
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

        for _eval_record in sorted(self.seq_dataset_record, key=lambda x: int(x.sentence_id)):
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
                logger.warning(f"No prediction labels found for sentence-id: {_sentence_id}")
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
        ] + list(dict_evaluations.keys())

        assert len(seq_excel_record_obj) > 0, "No records found."

        df_excel = pd.DataFrame(seq_excel_record_obj)
        df_excel = df_excel[EXCEL_HEADER]
        df_excel.to_excel(path_excel, index=False)
        logger.debug(f"Excel file is written out: {path_excel}")


    # -------------------------------------------------------
    
    def _get_ground_truth_labels(self, label_setting: str) -> ty.Dict[int, int]:
        """Obtaining ground truth labels."""
        assert label_setting in DEFAULT_GROUND_TRUTH_SETTINGS, f"Unsupported label setting: {label_setting}"
        dict_config_name2label = {}
        for dataset_record in self.seq_dataset_record:
            _sentence_id = int(dataset_record.sentence_id)
            if label_setting == "hallucination":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.error_type == "hallucination" else 0
            elif label_setting == "hallucination+mt-error":
                dict_config_name2label[_sentence_id] = 1 if (dataset_record.error_type == "hallucination") or (dataset_record.error_type == "mt-error") else 0
            elif label_setting == "mt-error":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.error_type == "mt-error" else 0
            elif label_setting == "error_full":
                dict_config_name2label[_sentence_id] = dataset_record.error_full
            elif label_setting == "error_strong":
                dict_config_name2label[_sentence_id] = dataset_record.error_strong
            elif label_setting == "error_repetitions":
                dict_config_name2label[_sentence_id] = dataset_record.error_repetitions
            elif label_setting == "error_omission":
                dict_config_name2label[_sentence_id] = dataset_record.error_omission
            elif label_setting == "error_named_entity":
                dict_config_name2label[_sentence_id] = dataset_record.error_named_entities
            else:
                raise ValueError(f"Unsupported label setting: {label_setting}")
            # end if
        # end for
        assert len(dict_config_name2label) > 0, "No records found."
        return dict_config_name2label


    def _fetch_prediction_results(self,
                                  config_name: str,
                                  eval_table_name: str) -> ty.List[ty.Dict]:
        """Fetching prediction results from the database."""
        # selecting the records having the specified config_name and eval_table_name.
        assert self.db_handler.conn is not None, "Database connection is not established."
        db_connection = self.db_handler.conn
        db_connection.row_factory = sqlite3.Row
        db_cursor = db_connection.cursor()
        sql_query = f"SELECT * FROM {eval_table_name} WHERE config_name = ?"

        db_cursor.execute(sql_query, (config_name,))
        seq_records = db_cursor.fetchall()

        return seq_records

    # custom-tailored pre-processing functions for each evaluation method.
    def _prep_mmd_flagger_ver1(self,
                               config_name: str,
                               table_name: str = DbTableRecordProposalMmdFlaggerVer1.__name__
                               ) -> ty.Tuple[ty.Dict[str, ty.Dict[int, int]], ty.Dict[str, ty.Tuple[float, float]]]:
        """I do aggregation. The aggregation key is (temperature-low, temperature-high)."""
        # fetching db records
        seq_records = self._fetch_prediction_results(config_name=config_name, eval_table_name=table_name)
        assert len(seq_records) > 0, f"No records found for {config_name} in {table_name}."
        
        # sort and aggregate the records.
        def _func_key_agg(record: ty.Dict) -> ty.Tuple[str, str]:
            return record['temperature_low'], record['temperature_high']
        # end def

        seq_predictions = []
        iter_group = itertools.groupby(sorted(seq_records, key=_func_key_agg), key=_func_key_agg)
        for __t_key, __g_obj in iter_group:
            _t_key = __t_key
            _seq_records = list(__g_obj)
            # obtaining the ground truth labels.
            _dict_sentid2label = {int(_d['sentence_id']): _d['flagging_label'] for _d in _seq_records}
            _obj = AggregatedMmdFlaggerVer1(temperature_low=_t_key[0], temperature_high=_t_key[1], dict_sentid2label=_dict_sentid2label)
            seq_predictions.append(_obj)
        # end for

        dict_evaluations = {}
        for __aggregated_container in seq_predictions:
            __key_name = f'MmdFlaggerVer1_{__aggregated_container.temperature_low}_{__aggregated_container.temperature_high}'
            dict_evaluations[__key_name] = __aggregated_container.dict_sentid2label
        # end

        dict_parameters = {}
        for __aggregated_container in seq_predictions:
            __key_name = f'MmdFlaggerVer1_{__aggregated_container.temperature_low}_{__aggregated_container.temperature_high}'
            dict_parameters[__key_name] = (__aggregated_container.temperature_low, __aggregated_container.temperature_high)
        # end

        return dict_evaluations, dict_parameters
    
    # custom-tailored pre-processing functions for each evaluation method.
    def _prep_mmd_trajectory_flagger_ver1(self,
                                          config_name: str,
                                          table_name: str = DbTableRecordProposalMmdFlaggerTrajectoryVer1.__name__
                                          ) -> ty.Dict[str, ty.Dict[int, int]]:
        """I do aggregation.
        
        Returns:
            - dict_evaluations: {tau-sequence: {sentence-id: label}}
                The `tau-sequence` is a json string of a list. The list is ty.List[float] and they are the temperature values.
        """
        # fetching db records
        seq_records = self._fetch_prediction_results(config_name=config_name, eval_table_name=table_name)
        if len(seq_records) == 0:
            logger.warning(f"No records found for {config_name} in {table_name}.")
            return {}
        # end if

        assert len(seq_records) > 0, f"No records found for {config_name} in {table_name}."
        
        # sort and aggregate the records by the field `tau_sequence` & `n_sampling` & filter options.
        def _func_key_agg(record: ty.Dict) -> ty.Tuple[str, int, str, str, str]:
            return record['tau_sequence'], record['n_sampling'], record['trajectory_rule'], record['trajectory_rule_smoothing'], record['trajectory_rule_smoothing_window']
        # end def

        seq_predictions = []
        iter_group = itertools.groupby(sorted(seq_records, key=_func_key_agg), key=_func_key_agg)
        for __t_key, __g_obj in iter_group:
            _tau_sequence = __t_key[0]
            _n_sampling = __t_key[1]
            _filter_name = f"{__t_key[2]}_{__t_key[3]}_{__t_key[4]}"

            _seq_records = list(__g_obj)
            # obtaining the ground truth labels.
            _dict_sentid2label = {int(_d['sentence_id']): _d['flagging_label'] for _d in _seq_records}

            _obj = dict(tau_sequence=_tau_sequence, 
                        filter_name=_filter_name,
                        n_sampling=_n_sampling,
                        dict_sentid2label=_dict_sentid2label)
            seq_predictions.append(_obj)
        # end for

        dict_evaluations = {}
        for __aggregated_container in seq_predictions:
            assert "tau_sequence" in __aggregated_container, "Invalid key."
            assert "dict_sentid2label" in __aggregated_container, "Invalid key."
            assert "filter_name" in __aggregated_container, "Invalid key."
            assert "n_sampling" in __aggregated_container, "Invalid key."

            __tau_parameter = __aggregated_container["tau_sequence"]
            __dict_sentid2label = __aggregated_container["dict_sentid2label"]
            __filter_name = __aggregated_container["filter_name"]
            __n_sampling = __aggregated_container["n_sampling"]

            __key_name = f'MmdTrajectoryFlaggerVer1_{__filter_name}_{__n_sampling}_{__tau_parameter}'
            dict_evaluations[__key_name] = __dict_sentid2label
        # end

        return dict_evaluations

    def _prep_baselines(self, 
                          config_name: str,
                          table_name: str,
                          eval_key_name: str
                          ) -> ty.Dict[str, ty.Dict[int, int]]:
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
            dict_sentid2label = {int(_d['sentence_id']): _d['flagging_label'] for _d in seq_records}

            dict_evaluations = {eval_key_name: dict_sentid2label}
            return dict_evaluations
    
    def _conduct_evaluation(self,
                            seq_ground_truth_settings: ty.List[str],
                            dict_evaluations: ty.Dict[str, ty.Dict[int, int]],
                            mode_processing_none_prediction: str = "zero_replacement") -> ty.List[EvaluationRecord]:
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
            dict_ground_truth = self._get_ground_truth_labels(label_setting=__setting_ground_truth)
            for __eval_method_name, __dict_sentid2label in dict_evaluations.items():
                # alignment of sentence-id.                
                __sentence_id_common = sorted(list(set(__dict_sentid2label.keys()) & set(dict_ground_truth.keys())))
                assert len(__sentence_id_common) > 0, "No common sentence-ids found."
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
                    logger.warning(f"No target labels found for {__setting_ground_truth}. Something Wrong. I skip this evaluation.")
                    continue
                # end if

                # calculating evaluation values. 
                try:
                    _eval_result = self._func_get_evaluation(array_label_prediction=__array_prediction,
                                                            array_label_ground_truth=__array_ground_truth)
                except Exception as e:
                    breakpoint()
                
                # creating an evaluation record.
                _eval_record = EvaluationRecord(approach_name=__eval_method_name,
                                                ground_truth_setting=__setting_ground_truth,
                                                evaluation_result=_eval_result)
                seq_evaluation_records_container.append(_eval_record)
            # end for
        # end for
        assert len(seq_evaluation_records_container) > 0, "No records found."
        return seq_evaluation_records_container
    
    def _make_heatmap_mmd_flagger_ver1(self,
                                       path_output_dir: Path,
                                       seq_evaluation_records_container: ty.List[EvaluationRecord],
                                       _dict_parameters: ty.Dict[str, ty.Tuple[float, float]]
                                       ) -> None:
        """A deserved method for MMD Flagger Ver1."""
        # generating the heatmap
        path_dir_output_heatmap = path_output_dir / "mmd_flagger_heatmap"
        path_dir_output_heatmap.mkdir(parents=True, exist_ok=True)
        self._write_out_mmd_flagger_heatmap(
            seq_evaluation_records=seq_evaluation_records_container,
            dict_approach2parameters=_dict_parameters,
            path_dir_output_heatmap=path_dir_output_heatmap,
            target_score="recall")
        self._write_out_mmd_flagger_heatmap(
            seq_evaluation_records=seq_evaluation_records_container,
            dict_approach2parameters=_dict_parameters,
            path_dir_output_heatmap=path_dir_output_heatmap,
            target_score="precision")
        self._write_out_mmd_flagger_heatmap(
            seq_evaluation_records=seq_evaluation_records_container,
            dict_approach2parameters=_dict_parameters,
            path_dir_output_heatmap=path_dir_output_heatmap,
            target_score="f1")

    def _make_precision_recall_curve_mmd_flagger_ver1(self,
                                                      path_output_dir: Path,
                                                      seq_evaluation_records_container: ty.List[EvaluationRecord],
                                                      _dict_parameters: ty.Dict[str, ty.Tuple[float, float]]
                                                      ) -> None:
        """Generating the precision-recall curve for MMD Flagger Ver1.
        
        This method plots the precision-recall curve. 
        """
        path_dir_output_heatmap = path_output_dir / "mmd_flagger_precision_recall_curve"
        path_dir_output_heatmap.mkdir(parents=True, exist_ok=True)

        # generating the precision-recall curve.
        seq_keys_mmd_flaggers = list(_dict_parameters.keys())
        seq_evaluation_mmd_flaggers = [rec for rec in seq_evaluation_records_container if rec.approach_name in seq_keys_mmd_flaggers]
        assert len(seq_evaluation_mmd_flaggers) > 0, "No records found."

        seq_precision = []
        seq_recall = []
        for _eval_record in seq_evaluation_mmd_flaggers:
            _precision = _eval_record.evaluation_result.precision
            _recall = _eval_record.evaluation_result.recall
            seq_precision.append(_precision)
            seq_recall.append(_recall)
        # end for

        # Note: I can not compute the Area-Under-Curve because the precision is not
        # # computing the Area-Under-Curve.
        # precision = np.array(seq_precision)
        # recall = np.array(seq_recall)
        # # score_auc = auc(precision, recall)
        # score_auc = 0.0

        # I want to visualise X: recall, Y: precision.
        f, ax = plot.subplots()
        sns.lineplot(x=seq_recall, y=seq_precision, ax=ax)
        ax.set_title(f"Precision-Recall Curve.")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower left')
        # __path_save_heatmap = path_visualisation / "heatmap_recall_score.png"
        f.savefig((path_dir_output_heatmap / "precision_recall_curve.png").as_posix())


    def _make_threshold_plots_for_baselines(self,
                                            path_output_dir: Path,
                                            config_name: str,
                                            setting_ground_truth: str = "hallucination",
                                            num_threshold: int = 5000
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

        def _extract_score_threshold(seq_db_records: ty.List[ty.Dict], threshold_column_name: str) -> float:
            __ = []
            for _record in seq_db_records:
                assert threshold_column_name in dict(_record), f"Invalid record. {threshold_column_name} not found: {_record}"
                _threshold = float(_record[threshold_column_name])
                __.append(_threshold)
            # end for
            assert len(set(__)) == 1, f"Invalid threshold values: {__}"
            return __[0]

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

        def _get_ground_truth(record: ty.Dict, dict_ground_truth: ty.Dict[int, int]) -> int:
            """Getting the ground truth label."""
            assert "sentence_id" in dict(record), f"Invalid record. 'sentence_id' not found: {record}"
            _sentence_id = int(record["sentence_id"])
            assert _sentence_id in dict_ground_truth, f"Invalid sentence-id: {_sentence_id}"

            return dict_ground_truth[_sentence_id]
        # end def

        def _func_key_sort_record(d: ty.Dict) -> str:
            """Sorting key function for the database record."""
            assert "sentence_id" in dict(d), f"Invalid record. 'sentence_id' not found: {d}"
            return str(int(d["sentence_id"]))
        # end def


        path_dir_output_plot = path_output_dir / "baselines_threshold_plots"
        path_dir_output_plot.mkdir(parents=True, exist_ok=True)

        TARGET_TABLES = [
            DbTableRecordGuerreiro2023McDSIM.__name__,
            DbTableRecordGuerreiro2023SeqLogProb.__name__,
        ]

        dict_ground_truth = self._get_ground_truth_labels(label_setting=setting_ground_truth)

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
                # logger.warning(f"No records found for {config_name} in {_table_name}.")
                continue
            # end if

            # I want to use the threshold for the visualisation.
            _score_threshold = _extract_score_threshold(seq_db_records=seq_records, threshold_column_name=_threshold_file_name)

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

                ax.axvline(x=_score_threshold, color='red', linestyle='--', label='Threshold')
                
                ax.set_title(f"{_metric_name.capitalize()} vs Threshold")
                ax.set_xlabel("Threshold")
                ax.set_ylabel(_metric_name.capitalize())

                fig.savefig(_path_output_graph.as_posix(), bbox_inches='tight', dpi=300)
                logger.debug(f"Saved the threshold plot to {_path_output_graph}")
            # end for

            # --------------------------------------------------------------------------------------------
            # plotting the relationship between the threshold and the metrics.
            _path_out_score_distribution = _path_subdir_table_nane / f"plot_score_distribution.png"
            fig, ax = plot.subplots()
            sns.histplot(_array_scores, bins=50, ax=ax)
            ax.set_title(f"Score Distribution")
            ax.axvline(x=_score_threshold, color='red', linestyle='--', label='Threshold')
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            fig.savefig(_path_out_score_distribution.as_posix(), bbox_inches='tight', dpi=300)

            # --------------------------------------------------------------------------------------------
            # evaluation values.
            _array_prediction_score = np.array(_array_scores, dtype=np.float32)
            _array_ground_truth = np.array(seq_ground_truth_label, dtype=np.int8)

            # computing the thresold steps
            _step = (max(_array_prediction_score) - min(_array_prediction_score)) / num_threshold
            seq_threshold_steps = np.arange(min(_array_prediction_score), max(_array_prediction_score), step=_step)
            
            _seq_value_plot = []
            for _threshold in seq_threshold_steps:
                _prediction_label = [1 if _score < _threshold else 0 for _score in _array_prediction_score]
                
                # computing the scores
                _matrix_confusion = confusion_matrix(_array_ground_truth, _prediction_label)
                tn, fp, fn, tp = _matrix_confusion.ravel()

                _tp_rate = (tp) / (tp + fn)
                _fp_rate = (fp) / (fp + tn)

                _seq_value_plot.append(dict(tp_rate=_tp_rate, fp_rate=_fp_rate))
            # end for

            # ploting precall-recall curve.
            _path_out_p_r_curve = _path_subdir_table_nane / f"roc-curve.png"
            fig, ax = plot.subplots()

            _df_rate_plot = pd.DataFrame(_seq_value_plot)

            sns.lineplot(data=_df_rate_plot, x="fp_rate", y="tp_rate", linewidth=3, ax=ax, legend=False)
            
            # adding the chance rate line.
            y = x = np.arange(0.0, 1.05, 0.05)
            sns.lineplot(x=x, y=y, dashes=True, linewidth=3, ax=ax, linestyle='--', legend=False)

            seq_tpr = [0.0] + _df_rate_plot.tp_rate
            seq_fpr = [0.0] + _df_rate_plot.fp_rate
            # Compute AUC using trapezoidal rule
            _roc_auc_score = np.trapz(seq_tpr, seq_fpr)            

            ax.set_title(f"ROC Curve (AUC = {round(_roc_auc_score, 4)})")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            
            fig.savefig(_path_out_p_r_curve.as_posix(), bbox_inches='tight', dpi=300)
            
            # saving the tsv file
            _path_out_p_r_curve_tsv = _path_subdir_table_nane / f"roc-curve.tsv"
            _df_rate_plot.to_csv(_path_out_p_r_curve_tsv, sep="\t", index=False)
        # # end for


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
                logger.debug(f"Saved the metrics graph to {_path_output_graph}")

    def main(self,
             path_output_dir: Path,
             config_name: str,
             seq_eval_table_name: ty.List[str],
             seq_ground_truth_settings: ty.Optional[ty.List[str]] = None
             ) -> None:
        """Main function to run the evaluation."""
        if seq_ground_truth_settings is None:
            seq_ground_truth_settings = list(DEFAULT_GROUND_TRUTH_SETTINGS)
        # end if

        # pre-processing the prediction results.
        dict_evaluations = {}
        for _eval_table_name in seq_eval_table_name:
            if _eval_table_name == DbTableRecordRaunak2021.__name__:
                _dict_predictions = self._prep_baselines(config_name=config_name, table_name=DbTableRecordRaunak2021.__name__, eval_key_name="Raunak2021")
            elif _eval_table_name == DbTableRecordGuerreiro2023McDSIM.__name__:
                _dict_predictions = self._prep_baselines(config_name=config_name, table_name=DbTableRecordGuerreiro2023McDSIM.__name__, eval_key_name="Guerreiro2023McDSIM")
            elif _eval_table_name == DbTableRecordGuerreiro2023SeqLogProb.__name__:
                _dict_predictions = self._prep_baselines(config_name=config_name, table_name=DbTableRecordGuerreiro2023SeqLogProb.__name__, eval_key_name="Guerreiro2023SeqLogProb")
            elif _eval_table_name == DbTableRecordProposalMmdFlaggerVer1.__name__:
                _dict_predictions, _dict_parameters = self._prep_mmd_flagger_ver1(config_name=config_name)
            elif _eval_table_name == DbTableRecordProposalMmdFlaggerTrajectoryVer1.__name__:
                _dict_predictions = self._prep_mmd_trajectory_flagger_ver1(config_name=config_name)
            else:
                raise ValueError(f"Unsupported table name: {_eval_table_name}")
            # end if
            dict_evaluations.update(_dict_predictions)
        # end for

        seq_evaluation_records_container = self._conduct_evaluation(
            seq_ground_truth_settings=seq_ground_truth_settings,
            dict_evaluations=dict_evaluations)

        # -------------------------------------------------------------
        # plotting relationships of metrics and threshold (for DbTableRecordGuerreiro2023McDSIM and DbTableRecordGuerreiro2023SeqLogProb)
        # TODO make a plot of the relationship between the metric distribution and the threshold.
        if _eval_table_name in (DbTableRecordGuerreiro2023McDSIM.__name__, DbTableRecordGuerreiro2023SeqLogProb.__name__):
            self._make_threshold_plots_for_baselines(
                path_output_dir=path_output_dir,
                config_name=config_name,
                setting_ground_truth="hallucination",
                num_threshold=1000
            )
        # end if
    
        # -------------------------------------------------------------        
        # I comment out the eval. related files about MMD Flagger Ver1.
        # # making the heatmap for MMD Flagger Ver1.
        # self._make_heatmap_mmd_flagger_ver1(
        #     path_output_dir=path_output_dir,
        #     seq_evaluation_records_container=seq_evaluation_records_container,
        #     _dict_parameters=_dict_parameters)
        # # precision-recall curve for MMD Flagger Ver1.
        # self._make_precision_recall_curve_mmd_flagger_ver1(
        #     path_output_dir=path_output_dir,
        #     seq_evaluation_records_container=seq_evaluation_records_container,
        #     _dict_parameters=_dict_parameters)
        # -------------------------------------------------------------        
        # Eval. files about Overview Analysis

        # writing out the evaluation results.
        path_output_book = path_output_dir / "evaluation_results.xlsx"
        self._write_out_evaluation_excel_book(
            path_excel=path_output_book,
            seq_evaluation_records=seq_evaluation_records_container)
        

        # path_submission_table_data = path_output_dir / "submission_table_data"
        # path_submission_table_data.mkdir(parents=True, exist_ok=True)
        # self._write_out_evaluation_submission_data(
        #     path_output_dir=path_submission_table_data,
        #     dict_evaluations=dict_evaluations)
        
        # I comment out the following method. Waste of time.
        # # visualising metrics graphs ex. F1, Precision, Recall
        # path_graph_output_dir = path_output_dir / "graphs_metrics"
        # path_graph_output_dir.mkdir(parents=True, exist_ok=True)        
        # self.make_metric_graphs(
        #     path_output_dir=path_graph_output_dir,
        #     seq_evaluation_records_container=seq_evaluation_records_container)

        # making an analysis excel book.
        path_output_analysis_book = path_output_dir.parent / "analysis_results.xlsx"
        self._write_out_analysis_excel_book(
            path_excel=path_output_analysis_book,
            dict_evaluations=dict_evaluations)
        # end

