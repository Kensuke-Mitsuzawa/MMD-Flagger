import typing as ty

from mmd_tst_variable_detector.kernels.base import BaseKernel
from mmd_tst_variable_detector.distance_module import BaseDistanceModule


class CustomDistanceModule():
    def __init__(self, distance_module: ty.Optional[str]):
        self.distance_module = distance_module

    def get_hyperparameters(self):
        return self.distance_module


class BaseCustomKernel(BaseKernel):
    def __init__(self, distance_module: ty.Optional[BaseDistanceModule | str]):
        super().__init__()
        self.distance_module: BaseDistanceModule | CustomDistanceModule
        if isinstance(distance_module, BaseDistanceModule):
             self.distance_module = distance_module
        else:
             self.distance_module = CustomDistanceModule(distance_module)
        # end if
        self.kernel_computation_type = "quadratic"

    def get_hyperparameters(self) -> ty.Dict[str, ty.Any]:
        return {
            'distance_module': self.distance_module.get_hyperparameters()
        }

         
