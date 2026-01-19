from typing import Any, Union, Optional, Callable
import jax
import jax.numpy as jnp
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

def serialize_device(device : Optional[Any]) -> Optional[str]:
    if device is None:
        return None
    assert hasattr(device, "platform") and hasattr(device, "id")
    return device.platform + ":" + str(device.id)

def deserialize_device(device_str : Optional[str]) -> Optional[Any]:
    if device_str is None:
        return None
    platform, id_str = device_str.split(":")
    id = int(id_str)
    # Find the device with matching platform and id
    for device in jax.devices(platform):
        if device.platform == platform and device.id == id:
            return device
    # Fallback with the same platform
    for device in jax.devices(platform):
        if device.platform == platform:
            return device
    raise ValueError(f"Device with platform '{platform}' and id '{id}' not found.")

def is_backendarray(data : Any) -> bool:
    return isinstance(data, jax.Array)

def from_numpy(
    data : np.ndarray,
    /,
    *,
    dtype : Optional[DTYPE_TYPE] = None,
    device : Optional[DEVICE_TYPE] = None
) -> ARRAY_TYPE:
    return jax.numpy.asarray(data, dtype=dtype, device=device)

def from_other_backend(
    other_backend: ComputeBackend,
    data: Any,
    /,
) -> ARRAY_TYPE:
    data_dlpack = other_backend.to_dlpack(data)
    return jax.dlpack.from_dlpack(data_dlpack)
    # except Exception as e:
    #     # jax sometimes has tiling issues with dlpack converted data
    #     np = other_backend.to_numpy(data)
    #     return from_numpy(np)

def to_numpy(
    data : ARRAY_TYPE
) -> np.ndarray:
    if data.dtype == jax.dtypes.bfloat16:
        data = data.astype(np.float32)
    return np.asarray(data)

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
    return dtype == jax.dtypes.bfloat16 or np.issubdtype(dtype, np.floating)

def dtype_is_boolean(
    dtype: DTYPE_TYPE
) -> bool:
    return dtype == np.bool_ or dtype == bool

from .._common.implementations import *
if hasattr(jax.numpy, "__array_api_version__"):
    compat_module = jax.numpy
else:
    import jax.experimental.array_api as compat_module
abbreviate_array = get_abbreviate_array_function(
    backend=compat_module,
    default_integer_dtype=default_integer_dtype,
    func_dtype_is_real_floating=dtype_is_real_floating,
    func_dtype_is_real_integer=dtype_is_real_integer,
    func_dtype_is_boolean=dtype_is_boolean
)
def map_fn_over_arrays(
    data : Any, func : Callable[[ARRAY_TYPE], ARRAY_TYPE]
):
    return jax.tree.map(
        func,
        data
    )
pad_dim = get_pad_dim_function(
    backend=compat_module,
)