import typing as ty
import sqlite3
import itertools
import logging
import json
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix

from ...guerreiro_2023_wmt.data_models.data_models import WMTDatasetRecord
from ...guerreiro_2023_wmt.data_models.utils import load_dataset

from ..module_management_db.module_sqlite3_handler import DBHandlerExp
from ..module_management_db.module_db_record import (
    DbTableRecordRaunak2021,
    DbTableRecordProposalMmdFlaggerVer1,
    DbTableRecordProposalMmdFlaggerTrajectoryVer1,
    DbTableRecordGuerreiro2023SeqLogProb,
    DbTableRecordGuerreiro2023McDSIM
)
from .evaluation_script_ver1 import EvaluationVer1

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




class AggregatedMmdFlaggerVer1MmdDistance(ty.NamedTuple):
    temperature_low: str
    temperature_high: str
    array_sentence_id: npt.NDArray[np.int8]
    array_mmd_a: npt.NDArray[np.float32]
    array_mmd_b: npt.NDArray[np.float32]
    array_diff_mmd_ab: npt.NDArray[np.float32]

    def __post_init__(self):
        assert len(self.array_sentence_id) == len(self.array_mmd_a), "Invalid array."
        assert len(self.array_sentence_id) == len(self.array_mmd_b), "Invalid array."
        assert len(self.array_mmd_a) == len(self.array_mmd_b), "Invalid array."


class ResultAnalysisRunnerVer1(EvaluationVer1):
    def __init__(self,
                 seq_dataset_record: ty.List[WMTDatasetRecord],
                 path_prediction_database: Path) -> None:
        assert path_prediction_database.exists(), f"Database file not found: {path_prediction_database}"
        self.seq_dataset_record = seq_dataset_record
        # opening the database file.
        db_con = sqlite3.connect(str(path_prediction_database))
        self.db_handler = DBHandlerExp(path_prediction_database)
        self.db_handler.conn = db_con

    # custom-tailored pre-processing functions for each evaluation method.
    def _prep_mmd_flagger_ver1(self,
                               config_name: str,
                               table_name: str = DbTableRecordProposalMmdFlaggerVer1.__name__
                               ) -> ty.Dict[str, AggregatedMmdFlaggerVer1MmdDistance]:
        """I do aggregation. The aggregation key is (temperature-low, temperature-high).
        
        I want to extract mmd-a, mmd-b from the database records."""
        # fetching db records
        seq_records = self._fetch_prediction_results(config_name=config_name, eval_table_name=table_name)
        assert len(seq_records) > 0, f"No records found for {config_name} in {table_name}."
        
        # sort and aggregate the records.
        def _func_key_agg(record: ty.Dict) -> ty.Tuple[str, str]:
            return record['temperature_low'], record['temperature_high']
        # end def

        # seq_predictions = []
        dict_evaluations = {}
        iter_group = itertools.groupby(sorted(seq_records, key=_func_key_agg), key=_func_key_agg)
        for __t_key, __g_obj in iter_group:
            _t_key = __t_key
            _seq_records = list(__g_obj)

            _seq_sentence_id = []
            _seq_mmd_a = []
            _seq_mmd_b = []
            for _record_obj in _seq_records:
                assert 'flagging_argument_json' in dict(_record_obj), "Invalid record object."
                _obj_json_string: str = _record_obj['flagging_argument_json']
                assert len(_obj_json_string) > 0, "Invalid json string."
                _obj_json = json.loads(_obj_json_string)
                assert isinstance(_obj_json, dict), "Invalid json object."

                assert "mmd_a" in _obj_json, "Invalid json object."
                assert "mmd_b" in _obj_json, "Invalid json object."

                _seq_sentence_id.append(int(_record_obj['sentence_id']))
                _seq_mmd_a.append(float(_obj_json['mmd_a']))
                _seq_mmd_b.append(float(_obj_json['mmd_b']))
            # end for

            _array_sent_id = np.array(_seq_sentence_id, dtype=np.int8)
            _array_mmd_a = np.array(_seq_mmd_a, dtype=np.float32)
            _array_mmd_b = np.array(_seq_mmd_b, dtype=np.float32)

            _temp_a = _t_key[0]
            _temp_b = _t_key[1]
            _obj = AggregatedMmdFlaggerVer1MmdDistance(
                temperature_low=_temp_a, 
                temperature_high=_temp_b, 
                array_sentence_id=_array_sent_id,
                array_mmd_a=_array_mmd_a,
                array_mmd_b=_array_mmd_b,
                array_diff_mmd_ab=np.abs(_array_mmd_a - _array_mmd_b))
            # seq_predictions.append(_obj)
            dict_evaluations[f'MmdFlaggerVer1_{_temp_a}_{_temp_b}'] = _obj
        # end for
        
        return dict_evaluations
    
    def _make_mmd_distance_histogram(self,
                                     path_output_dir: Path,
                                     dict_evaluations: ty.Dict[str, AggregatedMmdFlaggerVer1MmdDistance],
                                     seq_label_setting: ty.List[str] = list(DEFAULT_GROUND_TRUTH_SETTINGS),
                                     n_bins: int = 50
                                     ) -> None:
        """I want to make a histogram plot where,
        the x-axis: MMD distance, the y-axis: frequency.

        I create two histogram plots: plot for label=1, and plot for label=0.

        I create the pair of histogram plots for each approach and error-type.
        """
        assert path_output_dir.exists(), f"Output directory not found: {path_output_dir}"

        seq_approach_names = dict_evaluations.keys()
        for _label_setting in seq_label_setting:
            # obtaining the ground truth labels.
            _dict_ground_truth = self._get_ground_truth_labels(_label_setting)
            _seq_sent_id_truth = np.array(
                [_sent_id for _sent_id, _label in _dict_ground_truth.items() if _label == 1],
                dtype=np.int8)
            for _approach_name in seq_approach_names:
                _agg_mmd_distance_obj = dict_evaluations[_approach_name]
                # separating the prediction into two groups: label=1, and label=0.
                _array_filter_one_truth = np.isin(_agg_mmd_distance_obj.array_sentence_id, _seq_sent_id_truth).ravel()
                _array_filter_zero_truth = ~_array_filter_one_truth
                assert (_array_filter_one_truth == True).sum() + (_array_filter_zero_truth == True).sum() == len(_agg_mmd_distance_obj.array_sentence_id), "Invalid filter."
                
                array_mmd_a = _agg_mmd_distance_obj.array_mmd_a
                array_mmd_b = _agg_mmd_distance_obj.array_mmd_b
                array_diff_mmd_ab = _agg_mmd_distance_obj.array_diff_mmd_ab

                _max_mmd_value = max(array_mmd_a.max(), array_mmd_b.max())

                # making the histogram plot.
                _fig, _ax = plt.subplots(2, 1, figsize=(12, 10))
                # the histogram has two colours in a plot: mmd-a, mmd-b.
                array_mmd_a_label_one = array_mmd_a[_array_filter_one_truth]
                array_mmd_b_label_one = array_mmd_b[_array_filter_one_truth]
                _ax[0].hist([array_mmd_a_label_one, array_mmd_b_label_one], 
                            bins=n_bins, 
                            color=['red', 'blue'], 
                            label=['mmd-a', 'mmd-b'])
                _ax[0].set_title(f"Label=1. Red: mmd-a, Blue: mmd-b")
                _ax[0].set_xlim(0, _max_mmd_value)
                # the histogram has two colours in a plot: mmd-a, mmd-b.
                array_mmd_a_label_zero = array_mmd_a[_array_filter_zero_truth]
                array_mmd_b_label_zero = array_mmd_b[_array_filter_zero_truth]
                _ax[1].hist([array_mmd_a_label_zero, array_mmd_b_label_zero], 
                            bins=n_bins, 
                            color=['red', 'blue'], 
                            label=['mmd-a', 'mmd-b'])
                _ax[1].set_title(f"Label=0. Red: mmd-a, Blue: mmd-b")
                _ax[1].set_xlim(0, _max_mmd_value)

                # exporting the plot.
                _fig.suptitle(f"{_approach_name} - {_label_setting}")
                _path_histogram = path_output_dir / f"{_approach_name}_{_label_setting}.png"
                _fig.savefig(_path_histogram.as_posix(), dpi=300, bbox_inches='tight')
                logger.info(f"Saved: {_path_histogram}")

                # ----------------------------------------------
                # making the histogram plot.
                _fig, _ax = plt.subplots(2, 1, figsize=(12, 10))
                # the histogram has two colours in a plot: mmd-a, mmd-b.
                array_diff_ab_label_one = array_diff_mmd_ab[_array_filter_one_truth]
                _ax[0].hist(array_diff_ab_label_one, bins=n_bins)
                _ax[0].set_title(f"Label=1.")
                _ax[0].set_xlim(0, array_diff_mmd_ab.max())
                # the histogram has two colours in a plot: mmd-a, mmd-b.
                array_diff_ab_label_zero = array_diff_mmd_ab[_array_filter_zero_truth]
                _ax[1].hist(array_diff_ab_label_zero, bins=n_bins)
                _ax[1].set_title(f"Label=0.")
                _ax[1].set_xlim(0, array_diff_mmd_ab.max())

                # exporting the plot.
                _fig.suptitle(f"{_approach_name} - {_label_setting}")
                _path_histogram = path_output_dir / f"MMD-DIFF-{_approach_name}_{_label_setting}.png"
                _fig.savefig(_path_histogram.as_posix(), dpi=300, bbox_inches='tight')
                logger.info(f"Saved: {_path_histogram}")
            # end for
        # end for
    
    def main(self,
             path_output_dir: Path,
             config_name: str,
             seq_eval_table_name: ty.List[str],
             seq_ground_truth_settings: ty.Optional[ty.List[str]] = None
             ) -> None:
        """Main function to run the evaluation."""
        path_output_dir.mkdir(parents=True, exist_ok=True)

        if seq_ground_truth_settings is None:
            seq_ground_truth_settings = list(DEFAULT_GROUND_TRUTH_SETTINGS)
        # end if

        # pre-processing the prediction results.
        dict_evaluations = {}
        for _eval_table_name in seq_eval_table_name:
            if _eval_table_name == DbTableRecordRaunak2021.__name__:
                logger.warning("Raunak 2021 is not implemented yet. No Analysis using Raunak 2021.")
                #     _dict_predictions = self._prep_raunak_2021(config_name=config_name)
            elif _eval_table_name == DbTableRecordGuerreiro2023McDSIM.__name__:
                logger.warning("Not Implemented yet.")
            elif _eval_table_name == DbTableRecordGuerreiro2023SeqLogProb.__name__:
                logger.warning("Not Implemented yet.")
            elif _eval_table_name == DbTableRecordProposalMmdFlaggerTrajectoryVer1.__name__:
                logger.warning("Not Implemented yet.")
            elif _eval_table_name == DbTableRecordProposalMmdFlaggerVer1.__name__:
                _dict_predictions = self._prep_mmd_flagger_ver1(config_name=config_name)
                dict_evaluations.update(_dict_predictions)
            else:
                raise ValueError(f"Unsupported table name: {_eval_table_name}")
            # end if
            # dict_evaluations.update(_dict_predictions)
        # end for

        path_mmd_distance_histogram = path_output_dir / "mmd_distance_histogram"
        path_mmd_distance_histogram.mkdir(parents=True, exist_ok=True)
        self._make_mmd_distance_histogram(path_mmd_distance_histogram,
                                          dict_evaluations=dict_evaluations)
