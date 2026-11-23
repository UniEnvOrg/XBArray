from typing import Any, Union, Optional
import numpy as np
import torch
from ._typing import ARRAY_TYPE, DTYPE_TYPE, DEVICE_TYPE, RNG_TYPE
from xbarray.backends.base import ComputeBackend, SupportsDLPack

# Dtypes with little operator support in PyTorch, widened to a dtype that holds every value
# (uint64 -> int64 is checked for overflow in `from_numpy`)
PYTORCH_DTYPE_CAST_MAP = {
    torch.uint16: torch.int32,
    torch.uint32: torch.int64,
    torch.uint64: torch.int64,
    torch.float8_e4m3fn: torch.float16,
    torch.float8_e5m2: torch.float16,
}
_UNSIGNED_DTYPES = tuple(getattr(torch, name) for name in ("uint16", "uint32", "uint64") if hasattr(torch, name))

__all__ = [
    "default_integer_dtype",
    "default_index_dtype",
    "default_floating_dtype",
    "default_boolean_dtype",
    "serialize_device",
    "deserialize_device",
    "is_backendarray",
    "from_numpy",
    "from_other_backend",
    "to_numpy",
    "to_dlpack",
    "dtype_is_real_integer",
    "dtype_is_real_floating",
    "dtype_is_boolean",
    "abbreviate_array",
    "map_fn_over_arrays",
    "map_fn_over_arrays_ex",
    "pad_dim",
]

default_integer_dtype = torch.int32
default_index_dtype = torch.long
default_floating_dtype = torch.float32
default_boolean_dtype = torch.bool

def serialize_device(device : Optional[DEVICE_TYPE]) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, str):
        return device
    assert isinstance(device, torch.device)
    if device.index is None:
        # e.g. torch.device("cuda") means the *current* CUDA device, not necessarily cuda:0
        return device.type
    return f"{device.type}:{device.index}"

def deserialize_device(device_str : Optional[str]) -> Optional[DEVICE_TYPE]:
    if device_str is None:
        return None
    if ":" in device_str:
        device_type, index_str = device_str.split(":")
        index = int(index_str)
        return torch.device(device_type, index)
    else:
        return torch.device(device_str)

def is_backendarray(data : Any) -> bool:
    return isinstance(data, torch.Tensor)

def from_numpy(
    data : np.ndarray,
    /,
    *,
    dtype : Optional[DTYPE_TYPE] = None,
    device : Optional[DEVICE_TYPE] = None
) -> ARRAY_TYPE:
    data = np.asarray(data)  # numpy scalars
    if any(stride < 0 for stride in data.strides) or not data.dtype.isnative:
        # torch.from_numpy rejects negative strides and non-native byte orders
        data = np.ascontiguousarray(data, dtype=data.dtype.newbyteorder("="))
    if dtype is None and data.dtype == np.uint64 and data.size > 0 and int(data.max()) > np.iinfo(np.int64).max:
        raise OverflowError("uint64 values above the int64 range cannot be converted to a PyTorch int64 tensor; pass dtype= explicitly")
    t = torch.from_numpy(data)
    target_dtype = dtype if dtype is not None else PYTORCH_DTYPE_CAST_MAP.get(t.dtype, t.dtype)
    if target_dtype is not None or device is not None:
        t = t.to(device=device, dtype=target_dtype)
    return t

def from_other_backend(
    other_backend: ComputeBackend,
    data: Any,
    /,
) -> ARRAY_TYPE:
    if other_backend.simplified_name == "numpy":
        # Handles numpy scalars, negative strides and byte order, which DLPack does not
        return from_numpy(data)
    dat_dlpack = other_backend.to_dlpack(data)
    t = torch.from_dlpack(dat_dlpack)
    if other_backend.simplified_name == "jax":
        # JAX arrays are immutable: an in-place write to a tensor sharing their buffer would
        # silently change the JAX array, so the tensor gets its own copy
        t = t.clone()
    return t

def to_numpy(
    data : ARRAY_TYPE
) -> np.ndarray:
    # Tensors that require grad, or have the conjugate / negative bit set, cannot call .numpy() directly
    data = data.detach().resolve_conj().resolve_neg()
    # Torch bfloat16 is not supported by numpy
    if data.dtype == torch.bfloat16:
        data = data.to(torch.float32)
    return data.cpu().numpy()

def to_dlpack(
    data: ARRAY_TYPE,
    /,
) -> SupportsDLPack:
    # DLPack refuses tensors that require grad
    return data.detach()

def dtype_is_real_integer(
    dtype: DTYPE_TYPE
) -> bool:
    # https://pytorch.org/docs/stable/tensors.html#id12
    return dtype in [
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, 
        torch.int,
        torch.long,
        *_UNSIGNED_DTYPES,
    ]

def dtype_is_real_floating(
    dtype: DTYPE_TYPE
) -> bool:
    return dtype in [
        torch.float16, torch.float32, torch.float64, 
        torch.float, torch.double, 
        torch.bfloat16
    ]

def dtype_is_boolean(
    dtype: DTYPE_TYPE
) -> bool:
    return dtype == torch.bool

from .._common.implementations import *
from array_api_compat import torch as compat_module
abbreviate_array = get_abbreviate_array_function(
    compat_module, 
    default_integer_dtype=default_integer_dtype, 
    func_dtype_is_real_floating=dtype_is_real_floating,
    func_dtype_is_real_integer=dtype_is_real_integer,
    func_dtype_is_boolean=dtype_is_boolean
)
map_fn_over_arrays = get_map_fn_over_arrays_function(
    is_backendarray=is_backendarray,
)
map_fn_over_arrays_ex = get_map_fn_over_arrays_ex_function(
    is_backendarray=is_backendarray,
)
pad_dim = get_pad_dim_function(
    backend=compat_module,
)