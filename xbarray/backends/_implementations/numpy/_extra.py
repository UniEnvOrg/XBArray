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
    "map_fn_over_arrays_ex",
    "pad_dim",
    "clip",
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

map_fn_over_arrays_ex = get_map_fn_over_arrays_ex_function(
    is_backendarray=is_backendarray,
)

pad_dim = get_pad_dim_function(
    backend=compat_module,
)

_compat_clip = compat_module.clip
_INTEGER_LIMITS = {}

def _as_clip_bound(bound : Any, dtype : np.dtype, is_lower : bool) -> Any:
    """
    Converts a `clip` bound so that comparing it with an array of `dtype` keeps `dtype`,
    following `array_api_compat`'s conversion rules (the bound is cast to `dtype`).
    Returns None if the bound can never be active.
    """
    bound_type = type(bound)
    if bound_type is int:
        if dtype.kind != "f":
            limits = _INTEGER_LIMITS.get(dtype)
            if limits is None:
                info = np.iinfo(dtype)
                limits = _INTEGER_LIMITS[dtype] = (int(info.min), int(info.max))
            if (bound <= limits[0]) if is_lower else (bound >= limits[1]):
                return None
        # Python ints are weakly typed (NEP 50): numpy converts them to `dtype` itself
        return bound
    if bound_type is float and dtype.kind == "f":
        return bound
    if isinstance(bound, np.ndarray) and bound.dtype == dtype:
        return bound
    return np.asarray(bound, dtype=dtype)

def clip(
    x : ARRAY_TYPE,
    /,
    min : Optional[Union[int, float, ARRAY_TYPE]] = None,
    max : Optional[Union[int, float, ARRAY_TYPE]] = None,
    *,
    out : Optional[ARRAY_TYPE] = None,
) -> ARRAY_TYPE:
    """
    Array API `clip`: the result keeps the dtype of `x`, bounds broadcast against `x`,
    NaN in `x` or in a bound propagates, and 0-d results are returned as numpy scalars.
    Real integer / floating arrays (and numpy scalars) call numpy's `clip` / `maximum` / `minimum`
    directly, which is much faster than `array_api_compat`'s generic implementation (still used
    for `out=`, other dtypes and non-numpy inputs).
    """
    if (
        out is not None
        or not (type(x) is np.ndarray or isinstance(x, np.generic))
        or x.dtype.kind not in "fiu"
    ):
        return _compat_clip(x, min, max, out=out)
    dtype = x.dtype
    if min is not None:
        min = _as_clip_bound(min, dtype, True)
    if max is not None:
        max = _as_clip_bound(max, dtype, False)
    if min is None:
        return x.copy()[()] if max is None else np.minimum(x, max)
    if max is None:
        return np.maximum(x, min)
    if type(min) is not np.ndarray and type(max) is not np.ndarray:
        # numpy's fused clip loop is fastest with scalar bounds (but slow with array bounds)
        return x.clip(min, max)
    return np.minimum(np.maximum(x, min), max)
