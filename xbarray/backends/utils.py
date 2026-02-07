from .base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

try:
    import torch
    from .pytorch import PyTorchComputeBackend
except ImportError:
    torch = None
    PyTorchComputeBackend = None

try:
    import numpy as np
    from .numpy import NumpyComputeBackend
except ImportError:
    np = None
    NumpyComputeBackend = None

try:
    import jax
    from .jax import JaxComputeBackend
except ImportError:
    jax = None
    JaxComputeBackend = None

__all__ = [
    'get_backend_from_tensor',
]

def get_backend_from_tensor(tensor : BArrayType) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
    if torch is not None and isinstance(tensor, torch.Tensor):
        return PyTorchComputeBackend
    elif np is not None and isinstance(tensor, np.ndarray):
        return NumpyComputeBackend
    elif jax is not None and isinstance(tensor, jax.Array):
        return JaxComputeBackend
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")