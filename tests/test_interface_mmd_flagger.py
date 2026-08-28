import typing as ty
import tempfile
from pathlib import Path
import shutil

import torch
import numpy as np

import duckdb

from mmd_flagger.module_mmd_flagger import MMDFlaggerInterface, SingleSample, SampleSet, PossibleMetricsMMDFlaggerInterface
from mmd_flagger.module_mmd_flagger import QuadraticMmdEstimator
from mmd_flagger.module_mmd_flagger.module_kernels import DotProductKernel
from mmd_flagger.module_mmd_flagger.module_kernels import StringBasedGaussianKernel
from mmd_flagger.module_mmd_flagger.module_kernels.module_distance import MeteorDistanceModule

from mmd_flagger.utils import DecodingStrategyName, DecodingConfig
from mmd_flagger.utils import setup_resources


def test_interface_mmd_flagger_embedding():
    path_tmp_dir = Path(tempfile.mkdtemp())
    path_tmp_db = path_tmp_dir / 'tmp.duckdb'
    connect_duckdb = duckdb.connect(path_tmp_db)

    mmd_estimator = QuadraticMmdEstimator(DotProductKernel())

    seq_y_sto = []

    temp_settings = [0.1, 0.5, 1.0, 1.5]
    for _tmp in temp_settings:
        dec_config = DecodingConfig(
            method=DecodingStrategyName.Stochastic,
            temperature=_tmp
        )
        samples_y_sto = [np.random.normal(0, 1.0, size=(512,)) for _ in range(5)]
        _feat_h_sto = [SingleSample(sample_unique_id=f'{_i}', feature_vector=torch.from_numpy(_vec), text='') for _i, _vec in enumerate(samples_y_sto)]
        _sample_set_h_sto = SampleSet(label='Y_sto', decoding_config=dec_config, temperature_parameter=dec_config.temperature, samples=_feat_h_sto)

        seq_y_sto.append(_sample_set_h_sto)
    # end for
    
    samples_y_hyp = np.random.normal(0, 1.0, size=(512,))
    # Duplicating hyp sample to match user design (m=2)
    feature_vector_hyp = torch.stack([torch.from_numpy(samples_y_hyp), torch.from_numpy(samples_y_hyp)])
    _feat_h_hyp = SingleSample(sample_unique_id='0', feature_vector=feature_vector_hyp, text='')
    sample_set_h_sto = SampleSet(label='Y_hyp', decoding_config=None, temperature_parameter=None, samples=[_feat_h_hyp])

    mmd_flagger = MMDFlaggerInterface(mmd_estimator=mmd_estimator, backend_db=connect_duckdb)

    for _metric_set in ty.get_args(PossibleMetricsMMDFlaggerInterface):
        for _metric in ty.get_args(_metric_set):        
            score_obj = mmd_flagger.estimate(sample_set_h_sto, seq_y_sto, scoring_method=_metric)
            print(score_obj.score)

    connect_duckdb.close()

    shutil.rmtree(path_tmp_dir)


def test_interface_mmd_flagger_text():
    setup_resources.setup_string_kernel()

    path_tmp_dir = Path(tempfile.mkdtemp())
    path_tmp_db = path_tmp_dir / 'tmp.duckdb'
    connect_duckdb = duckdb.connect(path_tmp_db)

    mmd_estimator = QuadraticMmdEstimator(StringBasedGaussianKernel(MeteorDistanceModule()))

    seq_y_sto = []

    temp_settings = [0.1, 0.5, 1.0, 1.5]
    dict_tmp2text = {
        0.1: ['Tom is the well-seen first name in English spoken countries.', 'Tom is frequent name in English spoken countries.', 'Tom is common first name.'],
        0.5: ['Tom is the abbreviation of Tomas', 'Tom is a so-called nick-name of Tomas', 'Tom is common name.'],
        1.0: ['Tom is my dog.', 'Tom has been a president of the United States', 'Tom is who? That is the good question. To answer the question, Tom identity is necessary.' ],
        1.5: ['Tom is a devine', 'Tom is the superior existence above all', 'Tom knows everything.'],
    }
    for _tmp in temp_settings:
        dec_config = DecodingConfig(
            method=DecodingStrategyName.Stochastic,
            temperature=_tmp
        )
        _feat_h_sto = [SingleSample(sample_unique_id=f'{_i}', text=_t, feature_vector=None) for _i, _t in enumerate(dict_tmp2text[_tmp])]
        _sample_set_h_sto = SampleSet(label='Y_sto', decoding_config=dec_config, temperature_parameter=dec_config.temperature, samples=_feat_h_sto)

        seq_y_sto.append(_sample_set_h_sto)
    # end for
    
    _feat_h_hyp = SingleSample(sample_unique_id='0', feature_vector=None, text='Tom is the well-seen first name in English spoken countries.')
    sample_set_h_sto = SampleSet(label='Y_hyp', decoding_config=None, temperature_parameter=None, samples=[_feat_h_hyp])

    mmd_flagger = MMDFlaggerInterface(mmd_estimator=mmd_estimator, backend_db=connect_duckdb)

    for _metric_set in ty.get_args(PossibleMetricsMMDFlaggerInterface):
        for _metric in ty.get_args(_metric_set):        
            score_obj = mmd_flagger.estimate(sample_set_h_sto, seq_y_sto, scoring_method=_metric)
            print(score_obj.score)

    connect_duckdb.close()

    shutil.rmtree(path_tmp_dir)


if __name__ == '__main__':
    test_interface_mmd_flagger_text()
    test_interface_mmd_flagger_embedding()
    
