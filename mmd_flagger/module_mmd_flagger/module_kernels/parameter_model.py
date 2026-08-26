import typing as ty
from pydantic import BaseModel, ValidationInfo, Field, field_validator
from enum import Enum
import torch

class KernelType(str, Enum):
    Gaussian = "QuadraticKernelGaussianKernelCustom"
    Laplace = "LaplaceKernel"
    Matern = "MaternKernel"
    Polynominal = "PolynomialKernel"
    GaussianString = "StringBasedGaussianKernel"
    DotProductKernel = "DotProductKernel"

