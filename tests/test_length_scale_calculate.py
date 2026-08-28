from llm_decoding_comparison.modules_stats.module_mmd_flagger import (
    TensorPreprocessorVer1,
    KernelLengthScaleCalculator
)

import torch
import numpy as np

def test_length_scale_calculate():
    n_samples = 100
    n_tokens = 100
    n_dim = 512
    tensor_hyp = [torch.from_numpy(np.random.normal(size=(n_tokens, n_dim))) for _i in range(n_samples)]

    tensor_prop = TensorPreprocessorVer1()
    length_scale_calc = KernelLengthScaleCalculator(tensor_prop)
    scale_single = length_scale_calc.get_length_scale(tensor_hyp, kernel_length_scale_median_option='single')
    scale_vec = length_scale_calc.get_length_scale(tensor_hyp, kernel_length_scale_median_option='dimensionwise')
    assert scale_vec.shape[0] == n_dim
    assert len(scale_single.shape) == 0



if __name__ == '__main__':
    test_length_scale_calculate()
