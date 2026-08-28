import torch
import typing as ty
import typing

from mmd_tst_variable_detector.kernels.base import BaseKernel
from mmd_tst_variable_detector.kernels import (KernelMatrixObject, QuadraticKernelMatrixContainer)

from mmd_tst_variable_detector.datasets import BaseDataset
from .base_custom_kernel import BaseCustomKernel


class PolynomialKernel(BaseCustomKernel):
    """
    $$k(x, y; d, \gamma) = (\gamma x ^\intercal y + r)^d$$
    """
    def __init__(self,
                 degree: int = 3, 
                 constant: float = 1.0
                 ):
        super().__init__(distance_module=None)
        self.kernel_computation_type = "quadratic"         
        self.degree = degree
        self.constant = constant

    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> "PolynomialKernel":
        """Public API method to create a kernel object from a dataset.
        
        Must be implemented in a subclass.
        """
        dim_shape = dataset.get_dimension_flattened()
        return PolynomialKernel()

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

    def polynomial_kernel(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Polynomial Kernel.
        
        Args:
            x1 (torch.Tensor): A tensor of shape (N, D).
            x2 (torch.Tensor): A tensor of shape (M, D).
            degree (int): The polynomial degree.
            constant (float): The constant term.
            
        Returns:
            torch.Tensor: The kernel matrix of shape (N, M).
        """
        return (torch.matmul(x1, x2.T) + self.constant)**self.degree

    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              bandwidth: ty.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        k_xx = self.polynomial_kernel(x, x)
        k_yy = self.polynomial_kernel(y, y)
        k_xy = self.polynomial_kernel(x, y)

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
