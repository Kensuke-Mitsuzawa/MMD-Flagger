import typing as ty
from pathlib import Path
import json
import sqlite3
import random
import logging
import pickle
import zlib
import torch
import joblib
import numpy as np


import seaborn as sns
import matplotlib
import matplotlib.pylab as plot 

from ....guerreiro_2023_wmt.data_models.data_models import WMTDatasetRecord
from ....dale_2023_halomi.load_dataset import HalomiDatasetRecord
from ....module_translation_handler.ver2.module_base import TranslationResultContainer
from ....module_assessments.module_management_db.module_sqlite3_handler import DBHandlerExp
from ....logger_module import formatter

from .... import visualisation_header  # importing header

from . import get_ground_truth_label

logger = logging.getLogger(__name__)
std_handler = logging.StreamHandler()
std_handler.setLevel(logging.DEBUG)
std_handler.setFormatter(formatter)

matplotlib.use('Agg')  # Use non-interactive backend

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# example of the variable definitions.
# "DB-code-name": 
# d_relation_method_name2db_approach_key = {
#     "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.4, 0.8, 1.0, 1.2]_decoder.embed_tokens": "MmdErrorFlaggerTrajectoryVer2/25/decoder.embed_tokens/v1/no_filter/None/[0.1, 0.4, 0.8, 1.0, 1.2]"
# }  # Note: I have to define the DB key too. `dict_evaluations` has another key system.

# DB-code-name: Label for Human.
# d_method_shown_definition = {
#     # "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]_decoder.embed_tokens": "MMD-Hal-Flagger",
#     "MmdTrajectoryFlaggerVer2_v1_no_filter_None_25_[0.1, 0.4, 0.8, 1.0, 1.2]_decoder.embed_tokens": "MMD-Hal-Flagger",
# }

# approach code name: Label for human
# d_method_comparison = {
#     "Raunak2021": "TNG",
#     "Guerreiro2023McDSIM": "MC-DSIM",
#     "Guerreiro2023SeqLogProb": "SeqLogProb",
# }  # these approaches are refered to get "mmd-hal-flagger" can get, but others not.


class MmdTrajectoryTuple(ty.NamedTuple):
    sentence_id: str
    tau_sequence: ty.List
    mmd_distances: ty.List
    translation_stable: str
    hypothesis_translation: ty.List[ty.List[str]]
    flagging_argument: ty.Dict


def unpack_file(path_file: Path) -> ty.List[ty.Dict]:
    if "pkl.zlib" in path_file.name:
        with path_file.open('rb') as f:
            obj_saved = pickle.loads(zlib.decompress(f.read()))
            return obj_saved
    elif path_file.exists():
        obj_saved = torch.load(path_file)
        return obj_saved
    else:
        raise Exception()
# end def


def load_file(_path: Path) -> ty.Tuple[float, ty.List[TranslationResultContainer]]:
    _sample_size = _path.parent.name
    if _path.parent.parent.name == "beam":
        return (-1, [])
    # end if

    _tau_parameter = float(_path.parent.parent.name)
    _seq_obj_dict = unpack_file(_path)
    obj_cache = [TranslationResultContainer(**o) for o in _seq_obj_dict]

    return (_tau_parameter, obj_cache)


def fetch_translations(seq_all_translation_cache_files: ty.List[Path],
                       dataset_type: str,
                       sentence_id: str) -> ty.List[ty.List[str]]:

    def fetch_lfan_hall_translations() -> ty.List[ty.Tuple[float, ty.List[TranslationResultContainer]]]:
        # path_dir_root = path_dir_cache_translation / 'lfan_hall' / "FairSeqTranslationModelHandlerVer2"
        # if path_dir_root.exists() is False:
        #     logger.error(f"No translation cache directory found at {path_dir_root}")
            
        #     return []
        # # end if
        # seq_files = list(path_dir_root.rglob(f'**/{sentence_id}.pkl.zlib'))

        seq_files = [_path for _path in seq_all_translation_cache_files if  (f'{sentence_id}.pkl.zlib' == _path.name or f'{sentence_id}.pt' == _path.name) and "lfan_hall" in _path.as_posix()]

        if len(seq_files) == 0:
            logger.error(f"No translation files are found.")            
            raise NotImplementedError()
            return []
        # end if

        # seq_obj = []
        # for _path in seq_files:
        #     _sample_size = _path.parent.name
        #     if _path.parent.parent.name == "beam":
        #         continue
        #     # end if

        #     _tau_parameter = float(_path.parent.parent.name)
        #     _seq_obj_dict = unpack_file(_path)
        #     obj_cache = [TranslationResultContainer(**o) for o in _seq_obj_dict]
        #     seq_obj.append([_tau_parameter, obj_cache])
        # # end for
        _n_jobs = len(seq_files)
        seq_obj: ty.List = joblib.Parallel(n_jobs=1)(joblib.delayed(load_file)(_path) for _path in seq_files)
        seq_obj = [_t for _t in seq_obj if _t[0] != -1]

        seq_obj = sorted(seq_obj, key=lambda t: t[0])
        return seq_obj
    # end def

    def extract_hypothesis_translations(seq_translation_container: ty.List[ty.Tuple[float, ty.List[TranslationResultContainer]]]) -> ty.List[ty.List[str]]:
        seq_extracted_list = []
        for _t_container in seq_translation_container:
            _seq_container: ty.List[TranslationResultContainer] = _t_container[1]
            _translation_tau = [_container.translation_text for _container in _seq_container]
            seq_extracted_list.append(_translation_tau)
        # end for
        return seq_extracted_list
    # end def

    if dataset_type == 'lfan_hall':
        seq_containers = fetch_lfan_hall_translations()
        return extract_hypothesis_translations(seq_containers)
    else:
        logger.warning("Not Implemented.")
        return []


def _fetch_exectution_result_db(db_handler: DBHandlerExp,
                                dict_aggkey2db_primary_keys: ty.Dict[str, ty.List[str]],  # {approach-code-name: [DB primary keys]}
                                approach_code_name: str,
                                seq_sentence_ids: ty.List[str],
                                path_dir_cache_translation: Path,
                                dataset_type: str,
                                eval_table_name: str = 'DbTableRecordProposalMmdFlaggerTrajectoryVer3',
                                n_select_example: int = 10,
                                is_allow_like_query: bool = False
                                ) -> ty.List[MmdTrajectoryTuple]:
    """I refer to the db and extract information about,
    - tau values
    - mmd values
    - set of translations.
    """
    # approach_name_db_key = d_relation_method_name2db_approach_key[approach_name]
    
    assert db_handler.conn is not None, "Database connection is not established."
    db_connection = db_handler.conn
    db_connection.row_factory = sqlite3.Row
    db_cursor = db_connection.cursor()

    assert approach_code_name in dict_aggkey2db_primary_keys
    seq_unique_keys_db = dict_aggkey2db_primary_keys[approach_code_name]

    import more_itertools

    if is_allow_like_query:
        seq_records = []
        for _list_chunked in more_itertools.chunked(seq_unique_keys_db, n=500):
            _stack_sqlite_query = []
            for _db_key_name in _list_chunked:
                sql_query = f"SELECT sentence_id, tau_sequence, flagging_argument_json, record_id FROM {eval_table_name} WHERE record_id LIKE '{_db_key_name}%'"
                _stack_sqlite_query.append(sql_query)
            # end for
            _sql_query_union = ' UNION '.join(_stack_sqlite_query)
            db_cursor.execute(_sql_query_union)
            _record = db_cursor.fetchall()
            assert _record is not None
            seq_records += _record
    else:
        placeholders = ', '.join(['?'] * len(seq_unique_keys_db))
        sql_query = f"SELECT sentence_id, tau_sequence, flagging_argument_json, record_id FROM {eval_table_name} WHERE record_id IN ({placeholders})"

        db_cursor.execute(sql_query, seq_unique_keys_db)
        seq_records = db_cursor.fetchall()
        # assert len(seq_records) > 0
    # end if

    stack_extract_obj = []

    # listing up all available cache files
    seq_all_translation_cache_files_zlib = list(path_dir_cache_translation.rglob(f'**/*zlib'))
    seq_all_translation_cache_files_pt = list(path_dir_cache_translation.rglob(f'**/*pt'))
    seq_all_translation_cache_files = seq_all_translation_cache_files_zlib + seq_all_translation_cache_files_pt

    for _t_record in seq_records:
        assert len(_t_record) == 4
        _sent_id: str = _t_record[0]
        _json_tau_sequence: str = _t_record[1]
        _json_flagging_argument_json: str = _t_record[2]

        _tau_sequence = json.loads(_json_tau_sequence)
        _flagging_argument = json.loads(_json_flagging_argument_json)

        if _sent_id not in seq_sentence_ids:
            continue
        # end if


        # extracting the trasnaltions
        if len(seq_all_translation_cache_files) > 0:
            _seq_hypothesis_translations = fetch_translations(dataset_type=dataset_type,
                            seq_all_translation_cache_files=seq_all_translation_cache_files,
                            sentence_id=_sent_id)
        else:
            logger.error(f"No translation cache files are found. No hypothesis translations in the result.")
            _seq_hypothesis_translations = []

        stack_extract_obj.append(MmdTrajectoryTuple(
            sentence_id=_sent_id,
            tau_sequence=_tau_sequence,
            mmd_distances=_flagging_argument['mmd_distances'],
            translation_stable=_flagging_argument['translation_stable'],
            hypothesis_translation=_seq_hypothesis_translations,
            flagging_argument=_flagging_argument
        ))
    # end for
    assert len(stack_extract_obj) > 0
    return stack_extract_obj


def _filter_language_pair(d_sent_id2record_obj: ty.Dict[str, ty.Union[WMTDatasetRecord, HalomiDatasetRecord]],
                          seq_target_sentence_id: ty.List[ty.Tuple[str, str]],
                          language_pair_preference: ty.Tuple[str, str]) -> ty.List[ty.Tuple[str, str]]:
    assert language_pair_preference is not None
    stack_record = []
    for _t_sent_id in seq_target_sentence_id:
        _record_obj = d_sent_id2record_obj[_t_sent_id[0]]
        if isinstance(_record_obj, HalomiDatasetRecord):
            if _record_obj.src_lang == language_pair_preference[0] and _record_obj.tgt_lang == language_pair_preference[1]:
                stack_record.append(_t_sent_id)
        elif isinstance(_record_obj, WMTDatasetRecord):
            stack_record.append(_t_sent_id)
        # end if
    # end for
    # assert len(stack_record) > 0, f'0 records found for condition src={language_pair_preference[0]} tgt={language_pair_preference[1]}'
    return stack_record
# end def


def _select_target_sentence_ids(approach_code_name: str,
                                dataset_name: str,
                                d_sent_id2record_obj: ty.Dict[str, ty.Union[WMTDatasetRecord, HalomiDatasetRecord]],
                                dict_evaluations: ty.Dict[str, ty.Dict[str, int]],
                                selection_mode: str,
                                language_pair_preference: ty.Optional[ty.Tuple[str, str]] = None
                                ) -> ty.List[str]:
    """selecting the sentence-ids that I export.
    
    Return: A list of sentence-id that meets the `selection_mode` condition.
    """
    assert approach_code_name in dict_evaluations

    d_predictions = dict_evaluations[approach_code_name]

    seq_dataset_record = list(d_sent_id2record_obj.values())
    dict_ground_truth_hallucination = get_ground_truth_label._get_ground_truth_labels(
        seq_dataset_record=seq_dataset_record,
        dataset_name=dataset_name,
        label_setting='hallucination')
    sentence_id_common: ty.List[str] = sorted(list(set(dict_ground_truth_hallucination.keys()) & set(d_predictions.keys())))
    assert len(sentence_id_common) > 0, "No common sentence-ids found."

    seq_eval_label = []
    for _sent_id in sentence_id_common:
        _prediction_label: int = d_predictions[_sent_id]
        _ground_truth_label: int = dict_ground_truth_hallucination[_sent_id]

        _eval_label: str
        if _prediction_label == 1 and _ground_truth_label == 1:
            _eval_label = 'tp'
        elif _prediction_label == 1 and _ground_truth_label == 0:
            _eval_label = 'fp'
        elif _prediction_label == 0 and _ground_truth_label == 1:
            _eval_label = 'fn'
        elif _prediction_label == 0 and _ground_truth_label == 0:
            _eval_label = 'tn'
        else:
            raise ValueError()
        # end if
        seq_eval_label.append([_sent_id, _eval_label])
    # end if

    seq_target_sentence_id = []
    if selection_mode == 'true-positive' or selection_mode == 'prefer-mmd-hal-flagger-tp':
        seq_target_sentence_id = [_t[0] for _t in seq_eval_label if _t[1] == 'tp']
    elif selection_mode == 'false-positive':
        seq_target_sentence_id = [_t[0] for _t in seq_eval_label if _t[1] == 'fp']
    elif selection_mode == 'false-negative':
        seq_target_sentence_id = [_t[0] for _t in seq_eval_label if _t[1] == 'fn']
    elif selection_mode == 'true-negative':
        seq_target_sentence_id = [_t[0] for _t in seq_eval_label if _t[1] == 'tn']
    else:
        raise ValueError()
    # end if

    if language_pair_preference is not None:
        # filtering records by language pair.
        seq_eval_label = _filter_language_pair(d_sent_id2record_obj=d_sent_id2record_obj, 
                                               seq_target_sentence_id=seq_target_sentence_id, 
                                               language_pair_preference=language_pair_preference)
        if len(seq_eval_label) == 0:
            logger.error(f'No record found for selection-mode={selection_mode}, language-pair={language_pair_preference}')
            return []
    # end if

    # assert len(seq_target_sentence_id) > 0
    return seq_target_sentence_id
# end def


def _generate_example_object(execution_obj: ty.Dict) -> ty.Dict:
    """
    I collect the following,
    - source_text
    - reference_text
    - label
    - sets of sampled translation (at all tau values)
    - tau values.
    - mmd-values.
    
    And I pack the information above into an object.            
    """
    # obj_mmd_flag_res = MmdErrorFlagResult(**execution_obj)
    assert 'tau_sequence' in execution_obj
    assert 'flagging_argument' in execution_obj
    assert 'mmd_distances' in execution_obj['flagging_argument']
    return execution_obj
# end def


def _filter_prefer_mmd_hal_flagger_tp(seq_selection_id: ty.List[str],
                                      dict_evaluations: ty.Dict[str, ty.Dict],
                                      d_method_comparison: ty.Dict) -> ty.List[str]:
    """Filter function. I collect sentence-id that other methods did not detect."""
    stack_sent_id = []
    for _sent_id in seq_selection_id:
        _seq_labels = [
            dict_evaluations[_approach_name][_sent_id] for _approach_name in d_method_comparison.keys()
            if _approach_name in dict_evaluations]
        if sum(_seq_labels) == 0:
            stack_sent_id.append(_sent_id)
        # end if
        # for _approach_name in d_method_comparison.keys():
        #     _label_prediction: int = dict_evaluations[_approach_name][_sent_id]
    # end for
    return stack_sent_id


def get_minimum_mmd_args(seq_tau: ty.List[float], seq_mmd: ty.List[float]):
    # Circle mark at the minimum value
    min_idx = np.argmin(seq_mmd)
    min_tau = seq_tau[min_idx]
    min_mmd = seq_mmd[min_idx]

    return min_tau, min_mmd



def main_procedure(approach_code_name: str,
                   dataset_name: str,
                   d_sent_id2record_obj: ty.Dict[str, ty.Union[WMTDatasetRecord, HalomiDatasetRecord]],
                   dict_evaluations: ty.Dict[str, ty.Dict[str, int]],
                   dict_aggkey2db_primary_keys: ty.Dict[str, ty.List[str]],
                   d_method_comparison: ty.Dict[str, str],
                   db_handler: DBHandlerExp,
                   selection_mode: str,
                   path_subdir_selection_mode: Path,
                   path_dir_cache_translation: Path,  # a directory where the translation cache files are saved.
                   random_seed: int = 42,
                   language_pair_preference: ty.Optional[ty.Tuple[str, str]] = None,
                   n_sample_example: int = 10,
                   y_lim_mmd: ty.Tuple[float, float] = (0.0, 2.0),
                   range_threshold: float = 0.01,
                   range_threshold_pad_visualisation: float = 0.02
                   ):
    """The main procedure of exporting the example."""
    random_gen = random.Random(random_seed)

    _seq_selection_id = _select_target_sentence_ids(
        approach_code_name=approach_code_name, 
        dataset_name=dataset_name,
        d_sent_id2record_obj=d_sent_id2record_obj,
        dict_evaluations=dict_evaluations,
        selection_mode=selection_mode,
        language_pair_preference=language_pair_preference)
    
    if len(_seq_selection_id) == 0:
        return None

    if selection_mode == "prefer-mmd-hal-flagger-tp":
        # find sentence-id that other approaches did not detec (label value 0).
        _seq_selection_id = _filter_prefer_mmd_hal_flagger_tp(
            seq_selection_id=_seq_selection_id,
            dict_evaluations=dict_evaluations,
            d_method_comparison=d_method_comparison)
        if len(_seq_selection_id) == 0:
            return None
        # end if
    # end if

    _n_sample_example = min(n_sample_example, len(_seq_selection_id))
    # select N examples.
    _seq_sentence_id_selected_example = random_gen.sample(_seq_selection_id, k=_n_sample_example)

    _seq_execution_data = _fetch_exectution_result_db(
        db_handler=db_handler,
        dict_aggkey2db_primary_keys=dict_aggkey2db_primary_keys,
        approach_code_name=approach_code_name, 
        seq_sentence_ids=_seq_sentence_id_selected_example,
        dataset_type=dataset_name,
        path_dir_cache_translation=path_dir_cache_translation)
    
    for _d_obj in _seq_execution_data:
        _sent_id: str = _d_obj.sentence_id
        _path_subdir_sent_id = path_subdir_selection_mode / _sent_id
        _path_subdir_sent_id.mkdir(parents=True, exist_ok=True)

        # _d_obj_export = _generate_example_object(_d_obj)
        # save into json
        _path_subdir_sent_id_json = _path_subdir_sent_id / 'record.json'
        with _path_subdir_sent_id_json.open('w') as f:
            f.write(json.dumps(_d_obj._asdict(), ensure_ascii=False, indent=4))
        # end with

        # plot the tau-mmd.
        _path_subdir_sent_id_plot = _path_subdir_sent_id / 'trajectory.png'
        # _seq_tau = _d_obj_export['tau_sequence']
        # _seq_mmd = _d_obj_export['flagging_argument']['mmd_distances']
        _seq_tau = _d_obj.tau_sequence
        _seq_mmd = _d_obj.mmd_distances

        if len(_seq_mmd) != len(_seq_tau):
            logger.error("Encountering Exception. len(_seq_mmd) != len(_seq_tau).")
        else:
            f, ax = plot.subplots()
            sns.lineplot(x=_seq_tau, y=_seq_mmd, ax=ax, linewidth=3)
            ax.set_xlabel('Temperature $\\tau$')
            ax.set_ylabel('$\\widehat{MMD}_n (H_{\\text{beam}}, H_{\\text{sto}^{\\tau}})$')            
            
            tau_range = max(_seq_tau) - min(_seq_tau)
            left_threshold = min(_seq_tau) + (range_threshold + range_threshold_pad_visualisation) * tau_range
            # right_threshold = max(values_tau) - (range_threshold + range_threshold_pad_visualisation) * tau_range

            min_tau, min_mmd = get_minimum_mmd_args(_seq_tau, _seq_mmd)
        
            ax.plot(min_tau, min_mmd, 'o', color='red', markersize=10)            

            # setting the min and max of plotting.
            tau_minimum = min(_seq_tau) - 0.05
            tau_maximum = max(_seq_tau) + 0.01

            ax.set_xlim((tau_minimum, tau_maximum))
            # ax.set_ylim((-0.05, max_mmd_common + 0.05))
            ax.set_ylim(y_lim_mmd)

            ax.axvline(left_threshold, color='white', linestyle='--')
            ax.axvline(min_tau, color='black', linestyle='--')
            # ax.axvline(right_threshold, color='gray', linestyle='--')
            ax.axvspan(left_threshold, tau_maximum, color='tomato', alpha=0.5)
            ax.axvspan(tau_minimum, left_threshold, color='green', alpha=0.5)

            f.savefig(_path_subdir_sent_id_plot.as_posix(), bbox_inches='tight', dpi=300)

            plot.close()
    # end for
# end def