from pathlib import Path
import toml
import torch
import typing as ty
import dacite
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import itertools
import random


import matplotlib
import matplotlib.pyplot as plot

import dataclasses

from sklearn.metrics import confusion_matrix

from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset
from hallucination_mt.module_flagging.mmd_error_flagger_trajectory_ver3 import MmdErrorFlagResultVer3

import logzero
root_logger = logzero.logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


plot.rc('pdf', fonttype=42)
plot.rc('ps', fonttype=42)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20

BIG_TITLE_SIZE = 20

font = {'size'   : 22}

matplotlib.rc('font', **font)
matplotlib.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



@dataclasses.dataclass
class ConfigRoot:
    path_dataset_tsv: Path
    path_fairseq_model_checkpoint: Path
    path_fairseq_model_dir: Path
    path_sentencepiece_model: Path
    
    path_output_dir: Path
    path_translation_cache_root: Path

    dir_name_log: str = "logs"
    dir_name_eval: str = "evaluations"



path_toml = Path("<REPLACE HERE>/config_2025_05_16.toml")
config_obj = toml.load(path_toml)
config_obj["path_dataset_tsv"] = Path(config_obj["path_dataset_tsv"])
config_obj["path_fairseq_model_checkpoint"] = Path(config_obj["path_fairseq_model_checkpoint"])
config_obj["path_fairseq_model_dir"] = Path(config_obj["path_fairseq_model_dir"])
config_obj["path_sentencepiece_model"] = Path(config_obj["path_sentencepiece_model"])
config_obj["path_output_dir"] = Path(config_obj["path_output_dir"])
config_obj["path_translation_cache_root"] = Path(config_obj["path_translation_cache_root"])    

config_obj = dacite.from_dict(ConfigRoot, config_obj)



path_output_dir = config_obj.path_output_dir


# ------------------------------------------------------------------------
def parse_result_files(path_output_dir: Path) -> ty.Dict[int, ty.List[ty.List[MmdErrorFlagResultVer3]]]:

    # collecting result files
    seq_iterations = list(path_output_dir.rglob("**/iteration*/**"))

    dict_sample_size2iterations = {}  # {n-sample-size: [MmdErrorFlagResultVer3]}. One list is the collection of one iteration.

    root_logger.info(f"collecting .pt files...")
    for path_iteration_dir in seq_iterations:
        _name_iteration = path_iteration_dir.name

        # skip iteration_0 because it contains bug.
        if _name_iteration == "iteration_0" or _name_iteration == "iteration_5":
            continue

        # extracting sample size (in directory name)
        _n_sample = int(path_iteration_dir.parent.name)
        try:
            int(_n_sample)
        except ValueError:
            raise ValueError(f"Invalid directory name -> {path_iteration_dir.parent}")
        # end try
        _n_sample = int(_n_sample)
        dict_sample_size2iterations.setdefault(_n_sample, [])


        # collecting pt files
        _seq_pt_files = path_iteration_dir.rglob("**/*pt")

        _seq_obj_detection = []
        for _path_pt in _seq_pt_files:
            _obj_detection = MmdErrorFlagResultVer3(**torch.load(_path_pt))
            _seq_obj_detection.append(_obj_detection)
        # end for
        dict_sample_size2iterations[_n_sample].append(_seq_obj_detection)
        root_logger.info(f"Collected {len(_seq_obj_detection)} files from {_name_iteration} of N-sample={_n_sample}")
    # end for
    root_logger.info("Done parsing files")
    return dict_sample_size2iterations
# end def




# ------------------------------------------------------------------------
# evaluation of MMD-Flagger per iteration.


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


# ------------------------------------------------------------------------


def get_gold_label(prediction_record: MmdErrorFlagResultVer3) -> int:
    record_obj_gold = dict_sent_id2record[prediction_record.evaluation_pair['sentence_id']]
    if record_obj_gold.error_type == "hallucination":
        return 1
    else:
        return 0


def eval_detection_result(dict_sample_size2iterations: ty.Dict[int, ty.List[ty.List[MmdErrorFlagResultVer3]]]) -> ty.Dict[int, ty.List[_EvaluationResultContainer]]:
    dict_sample_size2evaluation_container = {}  # {n-sample-size: [_EvaluationResultContainer]}

    _set_iterations: ty.List[ty.List[MmdErrorFlagResultVer3]]
    for _n_sample, _set_iterations in dict_sample_size2iterations.items():
        dict_sample_size2evaluation_container.setdefault(_n_sample, [])
        for _set_one_iteration in _set_iterations:
            # making the array of prediction label
            _set_one_iteration: ty.List[MmdErrorFlagResultVer3] = list(sorted(_set_one_iteration, key=lambda t: t.evaluation_pair['sentence_id']))
            _seq_array_precition = [int(_o.is_hallucination) for _o in _set_one_iteration]
            _seq_array_gold = [get_gold_label(_r) for _r in _set_one_iteration]

            _eval_container = _func_get_evaluation(np.array(_seq_array_precition), np.array(_seq_array_gold))
            
            dict_sample_size2evaluation_container[_n_sample].append(_eval_container)
        # end for
        root_logger.info(f"Evaluated ")
    # end for
    return dict_sample_size2evaluation_container



# -------------------------------------------------------------------



def export_eval_score_to_table_file(path_excel_book: Path, dict_sample_size2evaluation_container: ty.Dict):
    # writing out to Excel book
    # Columns: sample-size, iteration-number, recall, n_total, true_positive, false_negative
    stack_record = []

    for _sample_size, _seq_list_iterations in dict_sample_size2evaluation_container.items():
        _eval_container: _EvaluationResultContainer
        for _iter_number, _eval_container in enumerate(_seq_list_iterations):
            _d_record = dict(
                n_sample=_sample_size,
                iteration_number=_iter_number,
                recall=_eval_container.recall,
                n_total=_eval_container.n_total,
                true_positive=_eval_container.true_positive,
                false_negative=_eval_container.false_negative)
            stack_record.append(_d_record)
        # end for
    # end for

    with pd.ExcelWriter(path_excel_book) as writer:
        pd.DataFrame(stack_record).to_excel(writer, sheet_name='evaluation_record')
    # end with

    root_logger.info(f"excel book is written at {path_excel_book}")
# end def



def make_box_plot_recall_score(path_plot: Path, dict_sample_size2evaluation_container: ty.Dict):
    stack_record = []

    for _sample_size, _seq_list_iterations in dict_sample_size2evaluation_container.items():
        _eval_container: _EvaluationResultContainer
        for _iter_number, _eval_container in enumerate(_seq_list_iterations):
            _d_record = dict(
                n_sample=_sample_size,
                iteration_number=_iter_number,
                recall=_eval_container.recall,
                n_total=_eval_container.n_total,
                true_positive=_eval_container.true_positive,
                false_negative=_eval_container.false_negative)
            stack_record.append(_d_record)
        # end for
    # end for

    df_plot = pd.DataFrame(stack_record)

    f, ax = plot.subplots(figsize=(10, 9))
    sns.boxplot(data=df_plot, x="n_sample", y="recall", hue="n_sample", ax=ax)
    # deleting the legend box
    ax.get_legend().remove()

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Recall")    


    f.savefig(path_plot.as_posix(), bbox_inches="tight", dpi=300)
    root_logger.info(f"box plot png is at {path_plot}")

# end def


def make_bar_plot_recall_score(path_plot: Path, dict_sample_size2evaluation_container: ty.Dict):
    stack_record = []

    for _sample_size, _seq_list_iterations in dict_sample_size2evaluation_container.items():
        _eval_container: _EvaluationResultContainer
        for _iter_number, _eval_container in enumerate(_seq_list_iterations):
            _d_record = dict(
                n_sample=_sample_size,
                iteration_number=_iter_number,
                recall=_eval_container.recall,
                n_total=_eval_container.n_total,
                true_positive=_eval_container.true_positive,
                false_negative=_eval_container.false_negative)
            stack_record.append(_d_record)
        # end for
    # end for

    df_plot = pd.DataFrame(stack_record)

    f, ax = plot.subplots(figsize=(10, 9))
    sns.barplot(data=df_plot, x="n_sample", y="recall", hue="n_sample", ax=ax, dodge=False)
    # deleting the legend box
    ax.get_legend().remove()

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Recall")    


    f.savefig(path_plot.as_posix(), bbox_inches="tight", dpi=300)
    root_logger.info(f"bar plot png is at {path_plot}")

# end def


def extract_mmd_distance(dict_sample_size2iterations: ty.Dict[int, ty.List[ty.List[MmdErrorFlagResultVer3]]],
                         index_mmd_extraction: int = 0) -> ty.Dict[int, ty.List[ty.List[float]]]:
    # extracting the MMD distance at the 1st tau value.
    dict_sample_size2iterations_mmd_distance = {}

    for _n_sample_size, _set_iterations in dict_sample_size2iterations.items():
        dict_sample_size2iterations_mmd_distance.setdefault(_n_sample_size, [])

        for _seq_one_iteration in _set_iterations:
            _stack_mmd_stance_at_index_iteration = []            
            _mmd_result: MmdErrorFlagResultVer3
            for _mmd_result in _seq_one_iteration:
                _mmd_distance_at_index = _mmd_result.mmd_distances[index_mmd_extraction]
                
                if np.isnan(_mmd_distance_at_index):
                    continue
                else:
                    _stack_mmd_stance_at_index_iteration.append(_mmd_distance_at_index)
                # end if
            # end for
            dict_sample_size2iterations_mmd_distance[_n_sample_size].append(_stack_mmd_stance_at_index_iteration)
        # end for
    # end for

    return dict_sample_size2iterations_mmd_distance


def plot_mmd_distance_distributions(path_dir_plot: Path,
                                    dict_sample_size2iterations_mmd_distance: ty.Dict[int, ty.List[ty.List[float]]],
                                    y_frequency_max: int = 30,
                                    x_mmd_max: float = 2.0,
                                    n_bins: int = 100
                                    ):
    """I make a set of png files. A png files represents a histogram of MMD distances at tau=0.1"""
    path_dir_plot.mkdir(parents=True, exist_ok=True)
    for _n_sample, _set_iterations in dict_sample_size2iterations_mmd_distance.items():
        for _iteration_no, _seq_one_iteration in enumerate(_set_iterations):
            _path_plot = path_dir_plot / f"sample-size-{_n_sample}_iteration-number-{_iteration_no}.png"

            _f, _ax = plot.subplots()
            sns.histplot(_seq_one_iteration, ax=_ax, bins=n_bins)

            _ax.set_xlim(0, x_mmd_max)
            _ax.set_ylim(0, y_frequency_max)

            _f.savefig(_path_plot.as_posix(), bbox_inches="tight", dpi=300)
        # end fir
    # end for
    root_logger.info(f"MMD distance distributions plot at {path_dir_plot}")
# end def
    

def make_mmd_distance_summary_stats_table(path_excel_book: Path, dict_sample_size2iterations_mmd_distance: ty.Dict[int, ty.List[ty.List[float]]]):
    # I make the summary statistics of the MMD distance distributions per sample-size and iteration and write them out to the Excel file.
    stack_record = []
    for _n_sample, _set_iterations in dict_sample_size2iterations_mmd_distance.items():
        for _iteration_no, _seq_one_iteration in enumerate(_set_iterations):
            _avg = np.average(_seq_one_iteration)
            _var = np.var(_seq_one_iteration)
            _median = np.median(_seq_one_iteration)
            _min = np.min(_seq_one_iteration)
            _max = np.max(_seq_one_iteration)

            _one_record_dict = dict(
                sample_size=_n_sample,
                iteration_number=_iteration_no,
                avg=_avg,
                var=_var,
                median=_median,
                min=_min,
                max=_max
            )

            for _percentile in [25, 75]:
                _v_at_percentile = np.percentile(_seq_one_iteration, q=_percentile)
                _one_record_dict[f'percentile_{_percentile}'] = _v_at_percentile
            # end for
            stack_record.append(_one_record_dict)
        # end for
    # end for

    with pd.ExcelWriter(path_excel_book) as writer:
        pd.DataFrame(stack_record).to_excel(writer, sheet_name="summary_statistics")
    # end with



# ------------------------------------------------------------------------

# parsing and collecting results.
dict_sample_size2iterations = parse_result_files(path_output_dir)

# evaluation per iteration.
seq_dataset = load_dataset(config_obj.path_dataset_tsv, delimiter='\t')
dict_sent_id2record = {str(_o.sentence_id): _o for _o in seq_dataset}

dict_sample_size2evaluation_container = eval_detection_result(dict_sample_size2iterations)
# import pprint
# pprint.pprint(dict_sample_size2evaluation_container)

# ------------------------------------------------------------------------
# MMD statistics forcusing on the sentence-ids

class _TupleAggregationTauParameterPerSentence(ty.NamedTuple):
    sentence_id: str
    tau_param: float
    seq_n_sample_and_iteration_no: ty.List[ty.Tuple[int, int]]
    seq_mmd: ty.List[float]


# ====================================================


def get_sentence_id_wise_aggtegation_data_structure(n_select_as_example_sentence = 10,
                                                    random_seed = 42) -> ty.Dict[str, ty.List[_TupleAggregationTauParameterPerSentence]]:
    _stack_stats_info = []

    for _n_sample, _set_iteration in dict_sample_size2iterations.items():
        _mmd_flag_res: MmdErrorFlagResultVer3
        for _i_iteration, _one_iteration in enumerate(_set_iteration):
            for _mmd_flag_res in _one_iteration:
                for _tau, _mmd in zip(_mmd_flag_res.tau_parameter, _mmd_flag_res.mmd_distances):    
                    _stack_stats_info.append([_mmd_flag_res.evaluation_pair["sentence_id"], _mmd_flag_res.n_sample, _i_iteration, _tau, _mmd])
                # end for
            # end for
        # end for
    # end for


    def _func_key_agg(t: ty.Tuple) -> str:
        return t[0]
    # end def

    # I want to make the data structure of {sentence-id: [_TupleAggregationTauParameterPerSentence]}. `_TupleAggregationTauParameterPerSentence` is sorted by tau-param and contains a set of mmd distances under various conditions.
    dict_sent_id2mmd_stats = {}
    for _sent_id, _g_obj in itertools.groupby(sorted(_stack_stats_info, key=_func_key_agg), key=_func_key_agg):
        _seq_mmd_stats_info = list(_g_obj)

        _seq_agg_obj_per_key_tau = []
        # aggregation by tau parameter.
        _g_obj_key_tau_param = itertools.groupby(sorted(_seq_mmd_stats_info, key=lambda t: t[3]), key=lambda t: t[3])
        for _tau_param, _g_packed_by_tau in _g_obj_key_tau_param:
            _g_packed_by_tau = list(_g_packed_by_tau)
            _seq_n_sample_and_iteration_no = [(_t[1], _t[2]) for _t in _g_packed_by_tau]
            _seq_mmd = [_t[-1] for _t in _g_packed_by_tau]

            _agg_obj = _TupleAggregationTauParameterPerSentence(
                sentence_id=_sent_id,
                tau_param=_tau_param,
                seq_n_sample_and_iteration_no=_seq_n_sample_and_iteration_no,
                seq_mmd=_seq_mmd)
            _seq_agg_obj_per_key_tau.append(_agg_obj)
        # end for
        _seq_agg_obj_per_key_tau = sorted(_seq_agg_obj_per_key_tau, key=lambda t: t.tau_param)
        dict_sent_id2mmd_stats[_sent_id] = _seq_agg_obj_per_key_tau
    # end for

    # selecting sentence-id to show as the example
    rand_gen = random.Random(random_seed)
    sentence_id_example = rand_gen.sample(list(dict_sent_id2mmd_stats.keys()), k=n_select_as_example_sentence)
    # selected sentences
    dict_sent_id2mmd_stats_selected = {_sent_id: dict_sent_id2mmd_stats[_sent_id] for _sent_id in sentence_id_example}

    return dict_sent_id2mmd_stats_selected
# end def


def plot_sentence_id_wise_mmd_stability(path_dir_plot: Path,
                                        dict_sent_id2mmd_stats: ty.Dict[str, ty.List[_TupleAggregationTauParameterPerSentence]]):
    for _sent_id, _seq_stats_tuple in dict_sent_id2mmd_stats.items():

        _seq_stack_df = []
        for _t in _seq_stats_tuple:
            for _mmd_value, _t_n_sample_and_iteration in zip(_t.seq_mmd, _t.seq_n_sample_and_iteration_no):
                _n_sample = _t_n_sample_and_iteration[0]
                _seq_stack_df.append(dict(tau_param=_t.tau_param, mmd=_mmd_value, n_sample=_n_sample))
            # end for
        # end for

        _df_plot = pd.DataFrame(_seq_stack_df)

        seq_n_samples = _df_plot.n_sample.unique()
        for _n_sample in seq_n_samples:
            _f, _ax = plot.subplots()

            _path_path_plot = path_dir_plot / f"plot-{_sent_id}-{_n_sample}.png"


            # _palette = sns.color_palette("colorblind", 4)
            _df_sub_data = _df_plot[_df_plot.n_sample == _n_sample]
            # sns.lineplot(data=_df_sub_data, x="tau_param", y="mmd", ax=_ax)

            # Compute mean and std at each x
            agg_df = _df_sub_data.groupby("tau_param").agg(y_mean=("mmd", "mean"), y_std=("mmd", "std")).reset_index()

            # Plot with error area
            sns.lineplot(data=agg_df, x="tau_param", y="y_mean", errorbar=None, label="Average")
            _ax.fill_between(agg_df["tau_param"],
                            agg_df["y_mean"] - agg_df["y_std"],
                            agg_df["y_mean"] + agg_df["y_std"],
                            alpha=0.3, color='blue')
            _ax.get_legend().remove()

            _ax.set_xlabel("Temperature $\\tau$")
            _ax.set_ylabel("$\\widehat{MMD}_n(H_{\\rm beam}, H_{\\rm sto}^{\\tau})$")

            _ax.set_title(f"sentence-id = {_sent_id}")

            # _ax.legend(loc='upper center', bbox_to_anchor=(1.2, 1.0), ncol=1, fancybox=False, shadow=False, frameon=False)  # bbox_to_anchor: (horizontal, vertical) of a relative position.

            _f.savefig(_path_path_plot.as_posix(), bbox_inches='tight', dpi=300)
            plot.close()
        # end for
    # end for
    root_logger.info(f"MMD stability per sentence-ids at {path_dir_plot}")
# end def


dict_sent_id2mmd_stats = get_sentence_id_wise_aggtegation_data_structure()

path_dir_plot_sentence_id_wise_stability = config_obj.path_output_dir / config_obj.dir_name_eval / "sentence_id_wise_mmd_stability"
path_dir_plot_sentence_id_wise_stability.mkdir(parents=True, exist_ok=True)
plot_sentence_id_wise_mmd_stability(path_dir_plot_sentence_id_wise_stability, dict_sent_id2mmd_stats)


# ------------------------------------------------------------------------
# plotintg MMD-Flagger detection performance.

path_eval_result = config_obj.path_output_dir / config_obj.dir_name_eval
path_eval_result.mkdir(parents=True, exist_ok=True)

path_excel_book_eval = path_eval_result / "recall_eval_result.xlsx"
export_eval_score_to_table_file(path_excel_book_eval, dict_sample_size2evaluation_container)

# Making a box plot; x: sample-size, y: recall.
path_recall_box_plot = path_eval_result / "recall_box_plot.png"
make_box_plot_recall_score(path_recall_box_plot, dict_sample_size2evaluation_container)


# Making a bar plot; x: sample-size, y: recall.
path_recall_bar_plot = path_eval_result / "recall_bar_plot.png"
make_bar_plot_recall_score(path_recall_bar_plot, dict_sample_size2evaluation_container)



# ------------------------------------------------------------------------
# about MMD distance (focusing on the tau=0.1)

dict_sample_size2iterations_mmd_distance = extract_mmd_distance(dict_sample_size2iterations)


path_dir_plot_mmd_distributions = config_obj.path_output_dir / config_obj.dir_name_eval / "mmd_distance_distributions_per_iterations"
plot_mmd_distance_distributions(path_dir_plot_mmd_distributions, dict_sample_size2iterations_mmd_distance)

path_dir_plot_mmd_distributions = config_obj.path_output_dir / config_obj.dir_name_eval / "mmd_distances_summary_statistics.xlsx"
make_mmd_distance_summary_stats_table(path_dir_plot_mmd_distributions, dict_sample_size2iterations_mmd_distance)
