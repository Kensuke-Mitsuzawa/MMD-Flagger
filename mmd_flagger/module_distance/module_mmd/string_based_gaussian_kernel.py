import typing

import torch
import numpy as np
import logging

from distributed import Client

from .module_distance import (JaccardDistanceModule, MeteorDistanceModule)

from mmd_tst_variable_detector.kernels.base import BaseKernel, BaseDataset, KernelMatrixObject
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel, QuadraticKernelMatrixContainer

logger = logging.getLogger(f'{__package__}.{__name__}')


PossibleDistanceModules = typing.Union[JaccardDistanceModule, MeteorDistanceModule]

InputTypeDataPointString = typing.List[str]
InputTypeString = typing.List[InputTypeDataPointString]


class StringBasedGaussianKernel(QuadraticKernelGaussianKernel):
    """Note: ARD weights does nothing in this Kernel."""
    def __init__(self,
                 distance_module: PossibleDistanceModules,
                 bandwidth: typing.Optional[torch.Tensor] = None,
                 ard_weights: typing.Optional[torch.Tensor] = None,
                 ard_weight_shape: typing.Optional[typing.Tuple[int, ...]] = None,
                 is_force_cutoff: bool = False,
                 ratio_cutoff: float = -1,
                 heuristic_operation: str = 'median',
                 is_auto_adjust_gamma: bool = False,
                 opt_bandwidth: bool = False):
        """
        Parameters
        ----------
        distance_module: BaseDistanceModule
            distance module object.
        bandwidth: torch.Tensor
            bandwidth for Gaussian kernel.
        ard_weights: torch.Tensor
            ARD weights for Gaussian kernel.
        ard_weight_shape: typing.Optional[typing.Tuple[int, ...]]
            ARD weight shape. If None, the shape is automatically determined.
        is_force_cutoff: bool
            If True, the kernel matrix is forced to be positive definite.
        ratio_cutoff: float
            If is_force_cutoff is True, the kernel matrix is forced to be positive definite.
            The ratio_cutoff is a ratio of the minimum eigenvalue to the maximum eigenvalue.
            If the ratio is smaller than the ratio_cutoff, the kernel matrix is forced to be positive definite.
        heuristic_operation: str
            'median' or 'mean'. The heuristic operation for median heuristic.
        is_auto_adjust_gamma: bool
            If True, the gamma is automatically adjusted to realize acceptable K_xy values.
        is_dimension_median_heuristic: bool
            If True, the median heuristic is computed for each dimension.
        opt_bandwidth: bool
            If True, the bandwidth is optimized.
        dask_client: typing.Optional[Client]
            Dask client object. Used for computing the initial length scale (bandwidth).
        """
        assert isinstance(distance_module, typing.get_args(PossibleDistanceModules))
        super().__init__(
            distance_module=distance_module,
            bandwidth=bandwidth,
            ard_weights=ard_weights,
            ard_weight_shape=(1,),
            is_force_cutoff=is_force_cutoff,
            ratio_cutoff=ratio_cutoff,
            heuristic_operation=heuristic_operation,
            is_auto_adjust_gamma=is_auto_adjust_gamma,
            is_dimension_median_heuristic=False,
            opt_bandwidth=opt_bandwidth,
        )
        self.distance_module: PossibleDistanceModules
        
    @classmethod
    def from_dataset(cls, 
                     dataset: BaseDataset,
                     distance_module: PossibleDistanceModules,
                     bandwidth: typing.Optional[torch.Tensor] = None,
                     ard_weights: typing.Optional[torch.Tensor] = None,
                     heuristic_operation: str = 'median',
                     is_dimension_median_heuristic: bool = True,
                     dask_client: typing.Optional[Client] = None
                     ) -> "StringBasedGaussianKernel":
        """Public API. Create a kernel object from a dataset.
        """
        # do kernel length initialization.
        if ard_weights is None:
            _t_data_dims = dataset.get_dimension_data_space()
            ard_weights = torch.ones(_t_data_dims)
        # end if
        
        assert ard_weights is not None, 'ard_weights is None.'
        
        
        kernel_obj = cls(
            distance_module=distance_module,
            bandwidth=bandwidth,
            ard_weights=ard_weights,
            heuristic_operation=heuristic_operation,
            is_dimension_median_heuristic=is_dimension_median_heuristic,
            opt_bandwidth=False,)
        
        # do kernel length initialization.
        kernel_obj.compute_length_scale_dataset(dataset)
        
        return kernel_obj
        
        
    def get_hyperparameters(self) -> typing.Dict[str, typing.Any]:
        return {
            'ard_weight_shape': list(self.ard_weights.shape),
            'is_force_cutoff': self.is_force_cutoff,
            'ratio_cutoff': self.ratio_cutoff,
            'heuristic_operation': self.heuristic_operation,
            'is_auto_adjust_gamma': self.is_auto_adjust_gamma,
            'is_dimension_median_heuristic': self.is_dimension_median_heuristic,
            'opt_bandwidth': self.opt_bandwidth
        }

    def _get_trainable_parameters(self) -> typing.List[torch.nn.Parameter]:
        return [self.ard_weights]

    def _get_median_single(self,
                           x: InputTypeString,
                           y: InputTypeString) -> torch.Tensor:
        """Get a median value for kernel functions.
        The approach is shown in 'Large sample analysis of the median heuristic'
        Args:
            x: (samples, features)
            y: (samples, features)
            minimum_sample: a minimum value for sampling.
            heuristic_operation: 'median' or 'mean'
        Returns:
            computed median
        """
        samp = x + y
        assert isinstance(self.distance_module, typing.get_args(PossibleDistanceModules))
        with torch.no_grad():
            distance_container = self.distance_module.compute_distance(samp, samp)
        # TODO: I want to make a cache;
        matrix_distance = distance_container.d_xy

        if isinstance(matrix_distance, torch.Tensor):
            matrix_distance = matrix_distance.numpy()
        # end if

        if self.heuristic_operation == 'median':
            med_sqdist = np.median(matrix_distance)
        elif self.heuristic_operation == 'mean':
            med_sqdist = np.mean(matrix_distance)
        else:
            raise Exception(f'No heuristic_operation == {self.heuristic_operation}.')
        # end if

        bandwidth = np.sqrt(med_sqdist / 2)
        # end if
        logger.debug("initial by median-heuristics {:.3g}".format(bandwidth))

        return torch.tensor([bandwidth])
    
    def _get_median_dim(self, 
                        x: torch.Tensor, 
                        y: torch.Tensor, 
                        is_completion_missing: bool = True, 
                        is_safe_guard_same_xy: bool = True) -> typing.Optional[torch.Tensor]:
        raise NotImplementedError('This kernel does not have the dim-wise.')
    

    def _compute_kernel_matrix_single(self,
                                      x: InputTypeString,
                                      y: InputTypeString,
                                      bandwidth: typing.Optional[torch.Tensor] = None
                                      ) -> KernelMatrixObject:
        # comment: I do not maintain this method anymore. Multi-dim length scale is the default.
        # Basically, I do not need this method anymore.
        if bandwidth is None:
            bandwidth = self.bandwidth
            assert bandwidth is not None
        # end if
        sigma = torch.exp(bandwidth)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))

        d_container = self.distance_module.compute_distance(x, y, False)

        k_xy = torch.exp(-1 * gamma * d_container.d_xy)
        k_xx = torch.exp(-1 * gamma * d_container.d_xx)
        k_yy = torch.exp(-1 * gamma * d_container.d_yy)

        k_container = QuadraticKernelMatrixContainer(k_xx, k_yy, k_xy)
        return KernelMatrixObject(kernel_computation_type=self.kernel_computation_type, x_size=len(x), y_size=len(y),
                                  kernel_matrix_container=k_container)

    def _compute_kernel_matrix_dim(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   bandwidth: typing.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        raise NotImplementedError('This kernel does not have the dim-wise.')