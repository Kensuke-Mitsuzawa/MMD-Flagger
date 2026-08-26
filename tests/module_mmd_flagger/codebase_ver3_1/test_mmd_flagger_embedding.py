import typing as ty

import torch
import numpy as np

from llm_decoding_comparison.modules_stats.module_mmd_flagger.module_mmd_flagger.codebase_ver3_1 import mmd_flagger_embedding

from llm_decoding_comparison.modules_stats.module_mmd_flagger import (
    QuadraticMmdEstimator,
    L2Distance,
)
from llm_decoding_comparison.modules_stats.module_mmd_flagger.module_kernels import *


def test_mmd_flagger_embedding():
    tensor_hyp = torch.normal(mean=torch.ones(2, 512))
    tau2embedding = {_t: torch.from_numpy(np.random.normal(loc=1+_t, size=(20, 512))) for _t in [0.1, 0.5, 1.0]}

    kernels = [
        'QuadraticKernelGaussianKernelCustom',
        'LaplaceKernel',
        'MaternKernel',
        'PolynomialKernel',
        'DotProductKernel'
    ]

    for _k_name in kernels:
        if _k_name == KernelType.Gaussian:
            _k_obj = QuadraticKernelGaussianKernelCustom(
                bandwidth_percentile=50,
                is_dimension_median_heuristic=False, 
                bandwidth=torch.Tensor([1.0,]),
                ard_weights=torch.ones(size=(512,)))
        elif _k_name == KernelType.DotProductKernel:
            _k_obj = DotProductKernel()
        elif _k_name == KernelType.Laplace:
            _k_obj = LaplaceKernel(sigma=1.0)
        elif _k_name == KernelType.Matern:
            _k_obj = MaternKernel(nu=0.5, length_scale=0.5)
        elif _k_name == KernelType.Polynominal:
            _k_obj = PolynomialKernel(degree=3, constant=1.0)
        else:
            raise NotImplementedError(f'Unknown Name {_k_name}')
        
        mmd_estimator = QuadraticMmdEstimator(kernel_obj=_k_obj)

        mmd_flagger = mmd_flagger_embedding.MmdErrorFlaggerTrajectoryVer3(mmd_estimator)
        res = mmd_flagger.flag_hallucination(
            processed_embedding_hypothesis=tensor_hyp,
            tau2processed_embedding_samples=tau2embedding,
        )
        assert isinstance(res, mmd_flagger_embedding.MmdErrorFlagResultVer3)
    

if __name__ == '__main__':
    test_mmd_flagger_embedding()

