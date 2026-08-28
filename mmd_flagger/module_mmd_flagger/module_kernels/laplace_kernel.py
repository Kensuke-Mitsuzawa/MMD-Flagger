import torch
import typing as ty
import typing

from mmd_tst_variable_detector.kernels.base import BaseKernel
from mmd_tst_variable_detector.kernels import (KernelMatrixObject, QuadraticKernelMatrixContainer)

from mmd_tst_variable_detector.datasets import BaseDataset

from .base_custom_kernel import BaseCustomKernel


class LaplaceKernel(BaseCustomKernel):
    """
    $$k(x, y; \sigma) = \exp(\frac{-||x-y||_1}{\sigma})$$
    """
    def __init__(self,
                 sigma: float = 1.0
                 ):
        # note: `ard_weight_shape` is a dummy variable. It's not used.
        super().__init__(distance_module='L1')
        self.sigma = sigma
        self.kernel_computation_type = "quadratic" 

    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> "LaplaceKernel":
        """Public API method to create a kernel object from a dataset.
        
        Must be implemented in a subclass.
        """
        dim_shape = dataset.get_dimension_flattened()
        return LaplaceKernel()

    # --------------------------------------------------------------------------------------------------
    # methods to be implemented.
    # private methods

    def _get_median_single(self,
                           x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _get_median_dim(self,
                        x: torch.Tensor,
                        y: torch.Tensor) -> typing.Optional[torch.Tensor]:
        raise NotImplementedError()

    def _compute_kernel_matrix_single(self,
                                      x: torch.Tensor,
                                      y: torch.Tensor,
                                      bandwidth: typing.Optional[torch.Tensor]) -> KernelMatrixObject:
        raise NotImplementedError()

    def _compute_kernel_matrix_dim(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   bandwidth: typing.Optional[torch.Tensor]) -> KernelMatrixObject:
        raise NotImplementedError()

    def _get_trainable_parameters(self) -> typing.List[torch.nn.Parameter]:
        """An abstract method that returns a list of trainable parameters.

        :return:
        """
        raise NotImplementedError()
    
    # -----------------------------------------------------------------------------    
    # methods to be implemented.
    # public API methods

    def get_hyperparameters(self) -> typing.Dict[str, typing.Any]:
        """A method to return a dictionary of hyperparameters.

        :return:
        """
        raise NotImplementedError()

    def laplace_kernel(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Laplace Kernel (Exponential Kernel).
        
        Args:
            x1 (torch.Tensor): A tensor of shape (N, D).
            x2 (torch.Tensor): A tensor of shape (M, D).
            sigma (float): The kernel hyperparameter.
            
        Returns:
            torch.Tensor: The kernel matrix of shape (N, M).
        """
        # Manhattan distance (L1 norm)
        dist = torch.cdist(x1, x2, p=1)
        return torch.exp(-dist / self.sigma)


    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              bandwidth: ty.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        k_xx = self.laplace_kernel(x, x)
        k_yy = self.laplace_kernel(y, y)
        k_xy = self.laplace_kernel(x, y)

        k_obj = KernelMatrixObject(
            kernel_computation_type='quadratic',
            x_size=x.shape[0],
            y_size=y.shape[0],
            kernel_matrix_container=QuadraticKernelMatrixContainer(
                k_xx=k_xx,
                k_yy=k_yy,
                k_xy=k_xy
            )
        )
        return k_obj
