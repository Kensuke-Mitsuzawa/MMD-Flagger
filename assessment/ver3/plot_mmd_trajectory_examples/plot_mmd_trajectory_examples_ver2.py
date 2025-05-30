"""A script of generating png files for Figure 1.
"""


# -------------------------------------------------------------------
# SCRIPT PARAMETERS

PATH_PLOT_DIR = "<<DIRECTORY WHERE THE DETECTION RESULTS ARE SAVED>> /evaluation_output/submission_detection_examples/25797ac7b01b7457adc0f9ee04905896"
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
range_threshold_pad_visualisation = 0.02  # padding value adding to `range_threshold`; this is for the visualisation purpose. Not the actual algorithm. 
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


def get_minimum_mmd_args(seq_tau: ty.List[float], seq_mmd: ty.List[float]):
    # Circle mark at the minimum value
    min_idx = np.argmin(seq_mmd)
    min_tau = seq_tau[min_idx]
    min_mmd = seq_mmd[min_idx]

    return min_tau, min_mmd



# selecting the target files.
path_examples_true_positive = PATH_PLOT_DIR / "prefer-mmd-hal-flagger-tp"
file_json_selected_tp = get_file_path_or_random_select(path_examples_true_positive, PREFERECE_SENTENCE_ID_TP)

path_examples_tn = PATH_PLOT_DIR / "true-negative"
file_json_selected_tn = get_file_path_or_random_select(path_examples_tn, PREFERECE_SENTENCE_ID_TN)

path_examples_fn = PATH_PLOT_DIR / "false-negative"
file_json_selected_fn = get_file_path_or_random_select(path_examples_fn, PREFERECE_SENTENCE_ID_FN)

path_examples_fp = PATH_PLOT_DIR / "false-positive"
file_json_selected_fp = get_file_path_or_random_select(path_examples_fp, PREFERECE_SENTENCE_ID_FP)

# max_mmd_common = get_mmd_max_common(
#     file_json_selected_tp,
#     file_json_selected_fp,
#     file_json_selected_tn,
#     file_json_selected_fn
# )



def plot_paired_line(file_json_selected_tp: Path,
                     file_json_selected_tn: Path,
                     path_save_png: Path,
                     is_add_text_annotation: bool = False,
                     is_apply_window_filter: bool = False,
                     window_length: int = 3):
    """Function to plot two MMD-trajectory in one plot. One trajectory is hallucination-text, and the other is correct-text."""
    obj_loaded_tp_hallucination = load_trajectory_object(file_json_selected_tp)
    obj_loaded_tn_correct = load_trajectory_object(file_json_selected_tn)


    # ----------------------------------------------
    # log text
    msg_tp = f"[Hallucination] sentence-id -> {obj_loaded_tp_hallucination.sentence_id}. Text: {obj_loaded_tp_hallucination.flagging_argument['evaluation_pair']}"
    msg_tn = f"[Correct Translation] sentence-id -> {obj_loaded_tn_correct.sentence_id}. Text: {obj_loaded_tn_correct.flagging_argument['evaluation_pair']}"    
    logger.info(msg_tp)
    logger.info(msg_tn)


    max_mmd_common = max(list(obj_loaded_tp_hallucination.mmd_distances) + list(obj_loaded_tn_correct.mmd_distances)) + 0.1

    if is_apply_window_filter:
        values_tau_filter_tp, values_mmd_filter_tp = apply_filter(np.array(obj_loaded_tp_hallucination.tau_sequence), np.array(obj_loaded_tp_hallucination.mmd_distances), type_filter='rolling_mean', window_length=window_length)
        values_tau_filter_tn, values_mmd_filter_tn = apply_filter(np.array(obj_loaded_tn_correct.tau_sequence), np.array(obj_loaded_tn_correct.mmd_distances), type_filter='rolling_mean', window_length=window_length) 
        df_plot_data = pd.DataFrame({
            'tau_tp': values_tau_filter_tp,
            'mmd_tp': values_mmd_filter_tp,
            'tau_tn': values_tau_filter_tn,
            'mmd_tn': values_mmd_filter_tn
        })
    else:
        df_plot_data = pd.DataFrame({
            'tau_tp': obj_loaded_tp_hallucination.tau_sequence,
            'mmd_tp': obj_loaded_tp_hallucination.mmd_distances,
            'tau_tn': obj_loaded_tn_correct.tau_sequence,
            'mmd_tn': obj_loaded_tn_correct.mmd_distances
        })
    # end if

    f, ax = plot.subplots()

    is_place_dot_markers = True 
    sns.lineplot(data=df_plot_data, x='tau_tp', y='mmd_tp', color='red', ax=ax, legend=True, label="Hallucinated Text", markers=is_place_dot_markers)
    sns.lineplot(data=df_plot_data, x='tau_tp', y='mmd_tn', color='green', ax=ax, legend=True, label="Correct Text", markers=is_place_dot_markers)

    min_tau_tp, min_mmd_tp = get_minimum_mmd_args(df_plot_data.tau_tp.tolist(), df_plot_data.mmd_tp.tolist())
    min_tau_tn, min_mmd_tn = get_minimum_mmd_args(df_plot_data.tau_tn.tolist(), df_plot_data.mmd_tn.tolist())

    # setting minimum point marks
    ax.plot(min_tau_tp, min_mmd_tp, 'o', color='red', markersize=10)
    ax.plot(min_tau_tn, min_mmd_tn, 'o', color='green', markersize=10)

    # setting labels
    ax.set_xlabel('temperature ($\\tau$)')
    ax.set_ylabel('$\\widehat{MMD}_n(H_{\\rm beam}, H_{\\rm sto}^{\\tau})$')

    # moving the legend box to up.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=False, shadow=False, frameon=False)  # bbox_to_anchor: (horizontal, vertical) of a relative position.

    # setting value-range that the algorithm refers to.
    values_tau = df_plot_data.tau_tp
    tau_range = max(values_tau) - min(values_tau)
    left_threshold = min(values_tau) + (range_threshold + range_threshold_pad_visualisation) * tau_range
    # right_threshold = max(values_tau) - (range_threshold + range_threshold_pad_visualisation) * tau_range

    # setting the min and max of plotting.
    tau_minimum = min(values_tau) - 0.05
    tau_maximum = max(values_tau) + 0.01

    ax.set_xlim((tau_minimum, tau_maximum))
    ax.set_ylim((-0.05, max_mmd_common + 0.05))

    # # Vertical band on the left 
    # ax.axvline(left_threshold, color='gray', linestyle='--')
    # ax.axvspan(tau_minimum, left_threshold, color='gray', alpha=0.3, label='Left shaded')
    # # Vertical band on the right
    # ax.axvline(left_threshold, color='gray', linestyle='--')
    # ax.axvspan(right_threshold, tau_maximum, color='gray', alpha=0.3, label='Left shaded')
    # color-masking in the middle 
    ax.axvline(left_threshold, color='white', linestyle='--')
    ax.axvline(min_tau_tp, color='black', linestyle='--')
    ax.axvline(min_tau_tn, color='black', linestyle='--')    
    # ax.axvline(right_threshold, color='gray', linestyle='--')
    ax.axvspan(left_threshold, tau_maximum, color='tomato', alpha=0.5)
    ax.axvspan(tau_minimum, left_threshold, color='green', alpha=0.5)    
    

    def _procedure_text_annotation_correct_translation():
        # Two-line text for the callout box
        text_max_length_char_tn = 20

        text_h_beam_tn = obj_loaded_tn_correct.translation_stable
        # selecting a text, which is different from `text_h_beam_tn`
        text_candidate_tau_01_1st = [_t for _t in obj_loaded_tn_correct.hypothesis_translation[0]][0]
        text_candidate_tau_01_2nd = [_t for _t in obj_loaded_tn_correct.hypothesis_translation[0]][0]        
        
        text_h_beam_01 = '{' + f"'{text_candidate_tau_01_1st[:text_max_length_char_tn]}...',\n'{text_candidate_tau_01_2nd[:text_max_length_char_tn]}...'" + '}'

        text = f"$y_{{\\rm beam}}$: '{text_h_beam_tn[:text_max_length_char_tn]}...'\n$Y_{{\\rm sto}}^{{\\tau=0.1}}$: {text_h_beam_01}"

        record_tau_tn_smallest = df_plot_data[df_plot_data.tau_tn == df_plot_data.tau_tn.min()]
        tau_min_tn = record_tau_tn_smallest.tau_tn.item()
        mmd_min_tn = record_tau_tn_smallest.mmd_tn.item()

        # Annotate with a callout (fancy arrow and box)
        ax.annotate(
            text,
            xy=(tau_min_tn, mmd_min_tn),           # point to annotate
            xytext=(tau_min_tn + 0.1, mmd_min_tn + 0.5),  # location of text
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
                color='black'
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=1),
            fontsize=13
        )
    # end if

    def _procedure_text_annotation_hallucination():
        # Two-line text for the callout box
        text_max_length_char_tp = 30
        is_select_disimilar = False

        n_show_text = 2

        # refer to these variables `min_tau_tp`, `min_mmd_tp`
        # I need a index of the arg-min(mmd) to select the text example.
        index_argmin_mmd = obj_loaded_tp_hallucination.tau_sequence.index(min_tau_tp)

        text_h_beam_tp = obj_loaded_tp_hallucination.translation_stable
        seq_text_H_tau_t = obj_loaded_tp_hallucination.hypothesis_translation[index_argmin_mmd]

        # leave the code-chunk of selecting similar text.
        # # I want to select a text that is really different from the h_beam.
        # seq_score_levenstein = [[jellyfish.levenshtein_distance(text_h_beam_tp, _t_tau), _t_tau] for _t_tau in seq_text_H_tau_t]
        # seq_score_levenstein = sorted(seq_score_levenstein, key=lambda t: t[0], reverse=is_select_disimilar)
        # # I want to make a format {text-1, text-2, ..., }
        # _seq_selected_text = []
        # logger.info(f"Candiate texts: {seq_text_H_tau_t}")
        # for _t_text_score in seq_score_levenstein[:n_show_text]:
        #     _seq_selected_text.append(_t_text_score[1][:text_max_length_char_tp])
        #     logger.info(f'Levenstein-score -> {_t_text_score[0]} at text -> {_t_text_score[1]}')
        # # end for

        _seq_selected_text = []
        logger.info(f"Candiate texts: {seq_text_H_tau_t}")
        for _t_text_score in seq_text_H_tau_t[:n_show_text]:
            assert isinstance(_t_text_score, str)
            _seq_selected_text.append(f"'{_t_text_score[:text_max_length_char_tp]}...'")
            logger.info(f'text -> {_t_text_score}')
        # end for
        stack_text_H_tau = "{" + ',\n'.join(_seq_selected_text) + "}"
        text = f"$y_{{\\rm beam}}$: '{text_h_beam_tp[:text_max_length_char_tp]}...'\n$Y_{{\\rm sto}}^{{\\tau={min_tau_tp}}}$: {stack_text_H_tau}"


        # Annotate with a callout (fancy arrow and box)
        ax.annotate(
            text,
            xy=(min_tau_tp, min_mmd_tp),           # point to annotate
            xytext=(min_tau_tp - 0.2, min_mmd_tp + 0.2),  # location of text
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
                color='black'
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=1),
            fontsize=13
        )
    # end if

    if is_add_text_annotation:
        _procedure_text_annotation_correct_translation()
        _procedure_text_annotation_hallucination()


    f.savefig( path_save_png.as_posix(), bbox_inches='tight', dpi=300)
# end def


logger.info(f'------ SCRIPT START ----------')

# I plot two lines: one trajectory of true-positive (of hallucination). the other of true-negative (correct translation).
path_save_png = PATH_SAVE_PLOT / "hallucination-and-correct-no-filter.png"
plot_paired_line(file_json_selected_tp, file_json_selected_tn, path_save_png=path_save_png, is_add_text_annotation=True, is_apply_window_filter=False)

path_save_png = PATH_SAVE_PLOT / "hallucination-and-correct-windows-filter.png"
plot_paired_line(file_json_selected_tp, file_json_selected_tn, path_save_png=path_save_png, is_add_text_annotation=False, is_apply_window_filter=True, window_length=2)

logger.info(f'------ SCRIPT END ----------')
