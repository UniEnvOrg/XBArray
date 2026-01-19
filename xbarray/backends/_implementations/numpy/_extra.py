from typing import Any, Union, Optional
import numpy as np
from ._typing import ARRAY_TYPE, DTYPE_TYPE, DEVICE_TYPE, RNG_TYPE
from xbarray.backends.base import ComputeBackend, SupportsDLPack

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
    "pad_dim",
]

default_integer_dtype = int
default_index_dtype = int
default_floating_dtype = float
default_boolean_dtype = bool

def serialize_device(device : Optional[DEVICE_TYPE]) -> Optional[str]:
    return None  # NumPy does not have device concept

def deserialize_device(device_str : Optional[str]) -> Optional[DEVICE_TYPE]:
    return None  # NumPy does not have device concept

def is_backendarray(data : Any) -> bool:
    return isinstance(data, np.ndarray)

def from_numpy(
    data : np.ndarray,
    /,
    *,
    dtype : Optional[DTYPE_TYPE] = None,
    device : Optional[DEVICE_TYPE] = None
) -> ARRAY_TYPE:
    return data

def from_other_backend(
    other_backend: ComputeBackend,
    data: Any,
    /,
) -> ARRAY_TYPE:
    return other_backend.to_numpy(data)

def to_numpy(
    data : ARRAY_TYPE
) -> np.ndarray:
    return data

def to_dlpack(
    data: ARRAY_TYPE,
    /,
) -> SupportsDLPack:
    return data

def dtype_is_real_integer(
    dtype: DTYPE_TYPE
) -> bool:
    return np.issubdtype(dtype, np.integer)

def dtype_is_real_floating(
    dtype: DTYPE_TYPE
) -> bool:
    return np.issubdtype(dtype, np.floating)

def dtype_is_boolean(
    dtype: DTYPE_TYPE
) -> bool:
    return dtype == np.bool_ or dtype == bool

from .._common.implementations import *
from array_api_compat import numpy as compat_module
abbreviate_array = get_abbreviate_array_function(
    backend=compat_module,
    default_integer_dtype=default_integer_dtype,
    func_dtype_is_real_floating=dtype_is_real_floating,
    func_dtype_is_real_integer=dtype_is_real_integer,
    func_dtype_is_boolean=dtype_is_boolean,
)

map_fn_over_arrays = get_map_fn_over_arrays_function(
    is_backendarray=is_backendarray,
)

pad_dim = get_pad_dim_function(
    backend=compat_module,
)