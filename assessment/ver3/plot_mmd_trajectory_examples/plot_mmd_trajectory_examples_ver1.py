"""A script of generating png files for Figure 1.
"""


# -------------------------------------------------------------------
# SCRIPT PARAMETERS

PATH_PLOT_DIR = "<DIRECTORY WHERE THE DETECTION RESULTS ARE SAVED> /evaluation_output/submission_detection_examples/25797ac7b01b7457adc0f9ee04905896"
PATH_SAVE_PLOT = "<DIRECTORY TO SAVE>"

PREFERECE_SENTENCE_ID_TP = "2515"
PREFERECE_SENTENCE_ID_TN = "2066"
PREFERECE_SENTENCE_ID_FP = None
PREFERECE_SENTENCE_ID_FN = "1140"
# -------------------------------------------------------------------


from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plot
import random
import matplotlib
import json
import numpy as np
import pandas as pd
import typing as ty
import logging

from hallucination_mt.module_assessments.module_evaluation.module_evaluation_script_ver3.plot_mmd_trajectory import MmdTrajectoryTuple
from hallucination_mt.module_flagging.module_classify_trajectory.module_classify_rule_base import apply_filter
from hallucination_mt.logger_module import formatter

random_seed = 20
range_threshold = 0.01
gen_random = random.Random(random_seed)



PATH_PLOT_DIR = Path(PATH_PLOT_DIR)
assert PATH_PLOT_DIR.exists()

PATH_SAVE_PLOT = Path(PATH_SAVE_PLOT)
PATH_SAVE_PLOT.mkdir(parents=True, exist_ok=True)


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


path_log_dir = PATH_SAVE_PLOT / 'figure1-selection.log'
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(path_log_dir)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_trajectory_object(file_json_selected: Path) -> MmdTrajectoryTuple:
    with file_json_selected.open() as f:
        obj_loaded = json.loads(f.read())
        obj_loaded = MmdTrajectoryTuple(**obj_loaded)
    # end with

    return obj_loaded


def plot_mmd_trajectory(file_json_selected: Path, 
                        max_mmd_common: float,
                        is_apply_window_filter: bool = False):
    """
    Args:
        is_apply_window_filter: if True, plot the filtered trajectory too in the same plot.
    """
    def _procedure_plot_trajectory(ax, seq_tau: ty.List[float], seq_mmd: ty.List[float], is_second_plot_line: bool = False):
        if is_second_plot_line:
            palette = "green"
        else:
            palette = "blue"
        
        sns.lineplot(x=seq_tau, y=seq_mmd, ax=ax, color=palette)

        # Circle mark at the minimum value
        min_idx = np.argmin(seq_mmd)
        min_tau = seq_tau[min_idx]
        min_mmd = seq_mmd[min_idx]

        ax.plot(min_tau, min_mmd, 'o', color='red', label='Minimum')

        if is_second_plot_line is False:
            # Vertical band on the left 
            ax.axvline(left_threshold, color='gray', linestyle='--')
            ax.axvspan(tau_minimum, left_threshold, color='gray', alpha=0.3, label='Left shaded')

            # Vertical band on the right
            ax.axvline(left_threshold, color='gray', linestyle='--')
            ax.axvspan(right_threshold, tau_maximum, color='gray', alpha=0.3, label='Left shaded')
        # end if

        return ax


    obj_loaded = load_trajectory_object(file_json_selected)

    values_tau = obj_loaded.tau_sequence
    values_mmd = obj_loaded.mmd_distances

    # setting value-range that the algorithm refers to.
    tau_range = max(values_tau) - min(values_tau)
    left_threshold = min(values_tau) + range_threshold * tau_range
    right_threshold = max(values_mmd) - range_threshold * tau_range

    # setting the min and max of plotting.
    tau_minimum = min(values_tau) - 0.05
    tau_maximum = max(values_tau) + 0.01

    f, ax = plot.subplots()

    _procedure_plot_trajectory(ax, seq_tau=values_tau, seq_mmd=values_mmd)

    if is_apply_window_filter:
        values_tau_filter, values_mmd_filter = apply_filter(np.array(values_tau), np.array(values_mmd), type_filter='rolling_mean', window_length=2)
        _procedure_plot_trajectory(ax, seq_tau=values_tau_filter.tolist(), seq_mmd=values_mmd_filter.tolist(), is_second_plot_line=True)
    # end if


    ax.set_xlim([tau_minimum, tau_maximum])
    ax.set_ylim([0, max_mmd_common + 0.05])

    ax.set_xlabel('temperature')
    ax.set_ylabel('MMD')

    plot.show()
    plot.close()

    return f, ax, obj_loaded


def save_plot(f, loaded_obj: MmdTrajectoryTuple, is_filter: bool = False):
    if is_filter:
            path_save = PATH_SAVE_PLOT / f'filter-{loaded_obj.sentence_id}.png'
    else:
        path_save = PATH_SAVE_PLOT / f'{loaded_obj.sentence_id}.png'
    f.savefig(path_save.as_posix(), bbox_inches='tight', dpi=300)
# end def


def get_file_path_or_random_select(path_dir_examples: Path, sentence_id_name: ty.Optional[str]):
    if sentence_id_name is None:
        assert path_dir_examples.exists()
        seq_list_trajectory_json = list(path_dir_examples.rglob("**/*json"))
        assert len(seq_list_trajectory_json) > 0

        file_json_selected = gen_random.sample(seq_list_trajectory_json, 1)[0]
    else:
        file_json_selected = path_dir_examples / f'{sentence_id_name}/record.json'
        assert file_json_selected.exists(), file_json_selected
    # end if
    return file_json_selected
# end if


def get_mmd_max_common(
        path_json_tp: Path,
        path_json_fp: Path,
        path_json_tn: Path,
        path_json_fn: Path) -> float:
    """I get the max of MMD values over tp, fp, tn, fn"""
    t_tp = load_trajectory_object(path_json_tp)
    t_fp = load_trajectory_object(path_json_fp)
    t_tn = load_trajectory_object(path_json_tn)
    t_fn = load_trajectory_object(path_json_fn)

    max_mmd = max(t_tp.mmd_distances + t_fp.mmd_distances + t_tn.mmd_distances + t_fn.mmd_distances)
    return max_mmd
# end def

path_examples_true_positive = PATH_PLOT_DIR / "prefer-mmd-hal-flagger-tp"
file_json_selected_tp = get_file_path_or_random_select(path_examples_true_positive, PREFERECE_SENTENCE_ID_TP)

path_examples_tn = PATH_PLOT_DIR / "true-negative"
file_json_selected_tn = get_file_path_or_random_select(path_examples_tn, PREFERECE_SENTENCE_ID_TN)

path_examples_fn = PATH_PLOT_DIR / "false-negative"
file_json_selected_fn = get_file_path_or_random_select(path_examples_fn, PREFERECE_SENTENCE_ID_FN)

path_examples_fp = PATH_PLOT_DIR / "false-positive"
file_json_selected_fp = get_file_path_or_random_select(path_examples_fp, PREFERECE_SENTENCE_ID_FP)


max_mmd_common = get_mmd_max_common(
    file_json_selected_tp,
    file_json_selected_fp,
    file_json_selected_tn,
    file_json_selected_fn
)



f, ax, obj_loaded = plot_mmd_trajectory(file_json_selected_tp, max_mmd_common, True)
save_plot(f, obj_loaded)
logger.info('prefer-mmd-hal-flagger-tp')
logger.info(f"sentence-id {obj_loaded.sentence_id}")
logger.info(f"text {obj_loaded.flagging_argument['evaluation_pair']}")




f, ax, obj_loaded = plot_mmd_trajectory(file_json_selected_tn, max_mmd_common)
save_plot(f, obj_loaded)
logger.info('true-negative')
logger.info(f"sentence-id {obj_loaded.sentence_id}")
logger.info(f"text {obj_loaded.flagging_argument['evaluation_pair']}")





f, ax, obj_loaded = plot_mmd_trajectory(file_json_selected_fn, max_mmd_common)
save_plot(f, obj_loaded)
logger.info('false-negative')
logger.info(f"sentence-id {obj_loaded.sentence_id}")
logger.info(f"text {obj_loaded.flagging_argument['evaluation_pair']}")



f, ax, obj_loaded = plot_mmd_trajectory(file_json_selected_fp, max_mmd_common)
save_plot(f, obj_loaded)
logger.info('false-positive')
logger.info(f"sentence-id {obj_loaded.sentence_id}")
logger.info(f"text {obj_loaded.flagging_argument['evaluation_pair']}")


