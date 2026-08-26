import torch
import typing as ty
import typing

# from mmd_tst_variable_detector.kernels.base import BaseKernel
from .base_custom_kernel import BaseCustomKernel
from mmd_tst_variable_detector.kernels import (KernelMatrixObject, QuadraticKernelMatrixContainer)

from mmd_tst_variable_detector.datasets import BaseDataset


class DotProductKernel(BaseCustomKernel):
    def __init__(self):
        super().__init__(distance_module=None)
        self.kernel_computation_type = "quadratic" 

    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> "DotProductKernel":
        """Public API method to create a kernel object from a dataset.
        
        Must be implemented in a subclass.
        """
        dim_shape = dataset.get_dimension_flattened()
        return DotProductKernel()

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


    def dot_product_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        return torch.matmul(X, Y.T)

    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              bandwidth: ty.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        k_xx = self.dot_product_kernel(x, x)
        k_yy = self.dot_product_kernel(y, y)
        k_xy = self.dot_product_kernel(x, y)

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
