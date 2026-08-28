import typing as ty
from pydantic import BaseModel

class MmdErrorFlagResultVer3(BaseModel):
    n_sample: int

    tau_parameter: ty.List[float]
    mmd_distances: ty.List[float]
    variance_mmd: ty.List[float]  # empiriclly estimated variance of MMD. See Sutherland, 2017.
    test_power_approximation: ty.List[float]  # empiriclly approximation of Test-Power regarding Two Sample Testing. See Sutherland, 2017.

    trajectory_shape: str
    is_hallucination: bool

    kernel_containers: ty.Optional[ty.List[ty.Dict]] = None  # list of Kernel matrices `KernelMatrixObject` for each \tau value.
