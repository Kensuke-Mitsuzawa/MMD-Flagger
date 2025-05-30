# %%
import sqlite3
import toml
from pathlib import Path
import typing as ty
import json
import logzero
import pandas as pd

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plot

from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset as load_dataset_lfan_hall
from hallucination_mt.guerreiro_2023_wmt.data_models.utils import WMTDatasetRecord

# management db module
from hallucination_mt.module_assessments.module_management_db.interface_ver3.module_db_record import (
    DbTableRecordProposalMmdFlaggerTrajectoryVer3
)


from hallucination_mt.dale_2023_halomi import load_dataset as load_dataset_halomi
from hallucination_mt.dale_2023_halomi.load_dataset import HalomiDatasetRecord

from hallucination_mt.module_assessments.module_management_db.module_sqlite3_handler import DBHandlerExp

from hallucination_mt import visualisation_header  # just importing.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logzero.logger



# %%
# -----------------------------------------------------------------------------------

PATH_TSV_LFAN_HALL = "<DIRECTORY PATH TO THE LFAN-HALL DATASET> /guerreiro_2023/annotated_corpus_checkpoint_2025_03_03_14h.tsv"
PATH_TSV_HALOMI = "<DIRECTORY PATH TO THE HALOMI DATASET> /halomi/dataset/data/halomi_core.tsv"

PATH_OUTPUT_ROOT_LFAN_HALL = "<DIRECTORY WHERE THE RESULTS ARE SAVED> 2025-05-06-word-embedding"
PATH_OUTPUT_ROOT_HALOMI = "<DIRECTORY WHERE THE RESULTS ARE SAVED> 05-06-word-embedding"

PATH_ANALYSIS_OUPUT_DIR = "<OUTPUT> submission_data"

N_BINS_MMD_ONE = 50
# -----------------------------------------------------------------------------------
# target conditions to render

kernel_conditions = {
    "kernel_type": "gaussian",
    "kernel_gaussian_length_scale": 25,
    "kernel_gaussian_length_scale_computation": "single"
}


# -----------------------------------------------------------------------------------

# %%
PATH_ANALYSIS_OUPUT_DIR = Path(PATH_ANALYSIS_OUPUT_DIR)
PATH_ANALYSIS_OUPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
PATH_TSV_LFAN_HALL = Path(PATH_TSV_LFAN_HALL)
PATH_TSV_HALOMI = Path(PATH_TSV_HALOMI)
# loading datasets
seq_lfan_hall = load_dataset_lfan_hall(PATH_TSV_LFAN_HALL, delimiter='\t')
seq_halomi = load_dataset_halomi.load_dataset(PATH_TSV_HALOMI)

# %%
PATH_OUTPUT_ROOT_LFAN_HALL = Path(PATH_OUTPUT_ROOT_LFAN_HALL)
PATH_OUTPUT_ROOT_HALOMI = Path(PATH_OUTPUT_ROOT_HALOMI)

# %%
path_db_lfan_hall = PATH_OUTPUT_ROOT_LFAN_HALL / 'management_db.sqlite3'
path_db_halomi = PATH_OUTPUT_ROOT_HALOMI / 'management_db.sqlite3'

assert path_db_lfan_hall.exists()
assert path_db_halomi.exists()

# I load records from the DB.

db_handler_lfan_hall = DBHandlerExp(path_db_lfan_hall)
db_handler_halomi = DBHandlerExp(path_db_halomi)


def fetch_target_record(db_handler: DBHandlerExp) -> ty.List[DbTableRecordProposalMmdFlaggerTrajectoryVer3]:
    assert db_handler.conn is not None
    cur = db_handler.conn.cursor()
    cur.execute("select * from DbTableRecordProposalMmdFlaggerTrajectoryVer3 where n_sampling = 25;")
    seq_record = cur.fetchall()
    assert len(seq_record) > 0

    seq_stack = []
    for _d_record in seq_record:
        _record_obj = DbTableRecordProposalMmdFlaggerTrajectoryVer3(**_d_record)
        _record_obj.record_id = _d_record["record_id"]
        seq_stack.append(_record_obj)
    # end for

    def filter_records(record_obj: DbTableRecordProposalMmdFlaggerTrajectoryVer3) -> bool:
        """filter the fetched records by the conditions"""
        args_kernels = json.loads(record_obj.args_kernel_options_json)
        _key_available_keys = list(args_kernels.keys())

        stack_conditions = []
        for _key_name, _value in kernel_conditions.items():
            assert _key_name in args_kernels, f"condition key {_key_name} is not found. Possible keys -> {_key_available_keys}"
            stack_conditions.append(args_kernels[_key_name] == _value)
        # end if

        return all(stack_conditions)
    # end def

    seq_stack_filtered = [_r for _r in seq_stack if filter_records(_r)]

    return seq_stack_filtered
# end for



seq_exec_lfan_hall = fetch_target_record(db_handler_lfan_hall)
seq_exec_halomi = fetch_target_record(db_handler_halomi)
logger.debug(f'LFAN -> {len(seq_exec_lfan_hall)} records, Halomi -> {len(seq_exec_halomi)} records')

# %%
# I get ground-truth labels and sentence-id.

def merge_records_lfan_hall(seq_exec_lfan_hall: ty.List[DbTableRecordProposalMmdFlaggerTrajectoryVer3], 
                            seq_dataset: ty.List[WMTDatasetRecord]) -> ty.List[ty.Dict]:
    d_sent_id2obj = {str(_r.sentence_id): _r for _r in seq_dataset}
    stack_obj = []
    for _d_prediction in seq_exec_lfan_hall:
        _sent_id: str = str(_d_prediction.sentence_id)
        _record_obj: WMTDatasetRecord = d_sent_id2obj[_sent_id]
        
        flagging_argument = json.loads(_d_prediction.flagging_argument_json)
        assert 'tau_parameter' in flagging_argument
        assert 'mmd_distances' in flagging_argument

        _obj = dict(
            sentence_id=_sent_id,
            record=_record_obj,
            tau_parameter=flagging_argument['tau_parameter'],
            mmd_distances=flagging_argument['mmd_distances'],
            label=_record_obj.error_type
        )
        stack_obj.append(_obj)
    # end for
    return stack_obj
# end def


def merge_records_halomi(seq_exec_halomi: ty.List[DbTableRecordProposalMmdFlaggerTrajectoryVer3], 
                         seq_dataset: ty.List[HalomiDatasetRecord]) -> ty.List[ty.Dict]:
    d_sent_id2obj = {str(_r.key_unique): _r for _r in seq_dataset}
    stack_obj = []
    for _d_prediction in seq_exec_halomi:
        _sent_id: str = str(_d_prediction.sentence_id)
        _record_obj: HalomiDatasetRecord = d_sent_id2obj[_sent_id]
        
        flagging_argument = json.loads(_d_prediction.flagging_argument_json)
        assert 'tau_parameter' in flagging_argument
        assert 'mmd_distances' in flagging_argument

        _obj = dict(
            sentence_id=_sent_id,
            record=_record_obj,
            tau_parameter=flagging_argument['tau_parameter'],
            mmd_distances=flagging_argument['mmd_distances'],
            label=_record_obj.error_type
        )
        stack_obj.append(_obj)
    # end for
    return stack_obj
# end def

seq_analysis_lfan_hall = merge_records_lfan_hall(seq_exec_lfan_hall, seq_lfan_hall)
seq_analysis_halomi = merge_records_halomi(seq_exec_halomi, seq_halomi)

assert len(seq_analysis_lfan_hall) > 0 and len(seq_analysis_halomi)

# %% [markdown]
# # MMD at $\tau=0.1$ Distribution

# %%
# I make the relationships of (sentence-id, label, MMD-distance)
def format_analysis_dataset(seq_analysis_record: ty.List[ty.Dict]) -> pd.DataFrame:
    # I extract the 1st MMD value
    seq_stack = []
    for _record in seq_analysis_record:
        _mmd_distance = _record['mmd_distances']
        assert len(_mmd_distance) > 0
        _mmd_1st: float = _mmd_distance[0]

        _sent_id: str = str(_record['sentence_id'])
        _laebl: str = _record['label']

        seq_stack.append(dict(
            sentence_id=_sent_id,
            label=_laebl,
            mmd=_mmd_1st
        ))
    # end for
    df_analysis = pd.DataFrame(seq_stack)
    return df_analysis
# end def


df_lfan_hall = format_analysis_dataset(seq_analysis_lfan_hall)
df_halomi = format_analysis_dataset(seq_analysis_halomi)


# %%
path_subdir_mmd_tau_one = PATH_ANALYSIS_OUPUT_DIR / 'mmd_tau_one_distribution'
path_subdir_mmd_tau_one.mkdir(parents=True, exist_ok=True)

# %%
# -----------------------------------------------------------------------------
# Plotting distributions of MMD(H_beam, H_tau_0.1) for the LFAN-HALL dataset.

f, ax = plot.subplots(nrows=3, figsize=(10, 7))
df_lfan_hall_correct = df_lfan_hall[df_lfan_hall['label'] == 'correct']
sns.histplot(data=df_lfan_hall_correct, x='mmd', ax=ax[0], stat='percent', bins=N_BINS_MMD_ONE)

df_lfan_hall_hall = df_lfan_hall[df_lfan_hall['label'] == 'hallucination']
sns.histplot(data=df_lfan_hall_hall, x='mmd', ax=ax[1], stat='percent', bins=N_BINS_MMD_ONE)


df_lfan_hall_mt = df_lfan_hall[df_lfan_hall['label'] == 'mt_error']
sns.histplot(data=df_lfan_hall_mt, x='mmd', ax=ax[2], stat='percent', bins=N_BINS_MMD_ONE)

max_x_lim = df_lfan_hall.mmd.max()
max_y_lim = 50

for _i_ax, _label in enumerate(['Correct', 'Hallucination', 'MT Error']):
    ax[_i_ax].set_xlim(0, max_x_lim)
    ax[_i_ax].set_ylim(0, max_y_lim)    
    ax[_i_ax].set_title(_label)
    ax[_i_ax].set_xlabel('')
    ax[_i_ax].set_ylabel('')
# end for
ax[1].set_ylabel('Percentage (%)')

plot.subplots_adjust(hspace=0.7)

path_plot = path_subdir_mmd_tau_one / 'lfan_hall.png'
f.savefig(path_plot.as_posix(), bbox_inches='tight')
logger.debug(f'plot -> {path_plot}')

path_tsv_mmd_tau_one = path_subdir_mmd_tau_one / 'lfan_hall.tsv'
df_lfan_hall.to_csv(path_tsv_mmd_tau_one, sep="\t", index=False)

# -----------------------------------------------------------------------------
# Plotting distributions of MMD(H_beam, H_tau_0.1) for the Halomi dataset.
# %%
f, ax = plot.subplots(nrows=3, figsize=(10, 7))
df_halomi_correct = df_halomi[df_halomi['label'] == 'correct']
sns.histplot(data=df_halomi_correct, x='mmd', ax=ax[0], stat='percent', bins=N_BINS_MMD_ONE)

df_halomi_hall = df_halomi[df_halomi['label'] == 'hallucination']
sns.histplot(data=df_halomi_hall, x='mmd', ax=ax[1], stat='percent', bins=N_BINS_MMD_ONE)

df_halomi_mt = df_halomi[df_halomi['label'] == 'mt_error']
sns.histplot(data=df_halomi_mt, x='mmd', ax=ax[2], stat='percent', bins=N_BINS_MMD_ONE)

max_x_lim = df_lfan_hall.mmd.max()
max_y_lim = 50

for _i_ax, _label in enumerate(['Correct', 'Hallucination', 'MT Error']):
    ax[_i_ax].set_xlim(0, max_x_lim)
    ax[_i_ax].set_ylim(0, max_y_lim)    
    ax[_i_ax].set_title(_label)
    ax[_i_ax].set_xlabel('')
    ax[_i_ax].set_ylabel('')
# end for
ax[1].set_ylabel('Percentage (%)')

plot.subplots_adjust(hspace=0.7)


path_plot = path_subdir_mmd_tau_one / 'halomi.png'
f.savefig(path_plot.as_posix(), bbox_inches='tight')
logger.debug(f'plot -> {path_plot}')

path_tsv_mmd_tau_one = path_subdir_mmd_tau_one / 'halomi.tsv'
df_lfan_hall.to_csv(path_tsv_mmd_tau_one, sep="\t", index=False)


# %% [markdown]
# # Smallest $\tau$ 

# %%
# I make the relationships of (sentence-id, label, tau-smallest-MMD)
def format_analysis_dataset_smallest_tau(seq_analysis_record: ty.List[ty.Dict]) -> pd.DataFrame:
    # I extract the 1st MMD value
    seq_stack = []
    for _record in seq_analysis_record:
        _mmd_distance = _record['mmd_distances']
        assert len(_mmd_distance) > 0
        _ind_min = np.argmin(_mmd_distance)

        _tau_seq = _record['tau_parameter']
        _tau_val = _tau_seq[_ind_min]

        _sent_id: str = str(_record['sentence_id'])
        _laebl: str = _record['label']

        seq_stack.append(dict(
            sentence_id=_sent_id,
            label=_laebl,
            index_min=_ind_min,
            tau_argmin=_tau_val
        ))
    # end for
    df_analysis = pd.DataFrame(seq_stack)
    return df_analysis
# end def


df_lfan_hall_min_tau_analysis = format_analysis_dataset_smallest_tau(seq_analysis_lfan_hall)
df_halomi_min_tau_analysis = format_analysis_dataset_smallest_tau(seq_analysis_halomi)


# %%
path_subdir_argmin_tau = PATH_ANALYSIS_OUPUT_DIR / 'argmin_tau'
path_subdir_argmin_tau.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Plotting distributions of min(MMD-trajectory) for the LFAN-HALL dataset.

# %%

# Compute bin edges from centers
bin_centers = np.arange(0.1, 1.1, 0.1)
bin_width = 0.1
bin_edges = bin_centers - bin_width / 2
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)  # add last edge

f, ax = plot.subplots(nrows=3, figsize=(10, 7))
df_lfan_hall_correct = df_lfan_hall_min_tau_analysis[df_lfan_hall_min_tau_analysis['label'] == 'correct']
sns.histplot(data=df_lfan_hall_correct, x='tau_argmin', ax=ax[0], stat='percent', bins=bin_edges)

df_lfan_hall_hall = df_lfan_hall_min_tau_analysis[df_lfan_hall_min_tau_analysis['label'] == 'hallucination']
sns.histplot(data=df_lfan_hall_hall, x='tau_argmin', ax=ax[1], stat='percent', bins=bin_edges)

df_lfan_hall_mt = df_lfan_hall_min_tau_analysis[df_lfan_hall_min_tau_analysis['label'] == 'mt_error']
sns.histplot(data=df_lfan_hall_mt, x='tau_argmin', ax=ax[2], stat='percent', bins=bin_edges)

max_x_lim = df_lfan_hall_min_tau_analysis.tau_argmin.max()
max_y_lim = 50

for _i_ax, _label in enumerate(['Correct', 'Hallucination', 'MT Error']):
    ax[_i_ax].set_xlim(0, max_x_lim)
    ax[_i_ax].set_ylim(0, max_y_lim)    
    ax[_i_ax].set_title(_label)
    ax[_i_ax].set_xlabel('')
    ax[_i_ax].set_ylabel('')
# end for
ax[1].set_ylabel('Percentage (%)')


plot.subplots_adjust(hspace=0.7)


path_plot = path_subdir_argmin_tau / 'lfan_hall.png'
f.savefig(path_plot.as_posix(), bbox_inches='tight')
logger.debug(f'plot -> {path_plot}')


path_tsv = path_subdir_argmin_tau / 'lfan_hall.tsv'
df_lfan_hall_min_tau_analysis.to_csv(path_tsv, sep="\t", index=False)

# %%
f, ax = plot.subplots(nrows=3, figsize=(10, 7))

# Compute bin edges from centers
bin_centers = np.arange(0.1, 1.1, 0.1)
bin_width = 0.1
bin_edges = bin_centers - bin_width / 2
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)  # add last edge


df_halomi_correct = df_halomi_min_tau_analysis[df_halomi_min_tau_analysis['label'] == 'correct']
sns.histplot(data=df_halomi_correct, x='tau_argmin', ax=ax[0], stat='percent', bins=bin_edges)

df_halomi_hall = df_halomi_min_tau_analysis[df_halomi_min_tau_analysis['label'] == 'hallucination']
sns.histplot(data=df_halomi_hall, x='tau_argmin', ax=ax[1], stat='percent', bins=bin_edges)

df_halomi_mt = df_halomi_min_tau_analysis[df_halomi_min_tau_analysis['label'] == 'mt_error']
sns.histplot(data=df_halomi_mt, x='tau_argmin', ax=ax[2], stat='percent', bins=bin_edges)

max_x_lim = df_halomi_min_tau_analysis.tau_argmin.max()
max_y_lim = 30

for _i_ax, _label in enumerate(['Correct', 'Hallucination', 'MT Error']):
    ax[_i_ax].set_xlim(0, max_x_lim)
    ax[_i_ax].set_ylim(0, max_y_lim)    
    ax[_i_ax].set_title(_label)
    ax[_i_ax].set_xlabel('')
    ax[_i_ax].set_ylabel('')
# end for
ax[1].set_ylabel('Percentage (%)')

plot.subplots_adjust(hspace=0.7)


path_plot = path_subdir_argmin_tau / 'halomi.png'
f.savefig(path_plot.as_posix(), bbox_inches='tight')
logger.debug(f'plot -> {path_plot}')

path_tsv = path_subdir_argmin_tau / 'halomi.tsv'
df_halomi_min_tau_analysis.to_csv(path_tsv, sep="\t", index=False)



