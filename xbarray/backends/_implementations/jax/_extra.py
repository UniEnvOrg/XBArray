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
    "map_fn_over_arrays_ex",
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
    if isinstance(data, np.ndarray) and not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder("="))  # JAX rejects non-native byte orders
    return jax.numpy.asarray(data, dtype=dtype, device=device)

def from_other_backend(
    other_backend: ComputeBackend,
    data: Any,
    /,
) -> ARRAY_TYPE:
    if other_backend.simplified_name == "jax":
        return data
    device_str = other_backend.serialize_device(other_backend.device(data))
    if device_str is None or device_str.startswith("cpu"):
        # Host data goes to JAX's default device (as `from_numpy` does) instead of staying
        # committed to the CPU, and is copied: a JAX array must not change when the source does
        return from_numpy(other_backend.to_numpy(data))
    # Import a private copy made by the source library (on its own stream, so it is ordered
    # before any later in-place write there): JAX arrays are immutable and must not alias a
    # buffer the other library may modify. The copy is also compact, which JAX's DLPack
    # import requires (it rejects e.g. tensors sliced with a step or broadcast).
    private_copy = other_backend.asarray(data, copy=True)
    return jax.dlpack.from_dlpack(other_backend.to_dlpack(private_copy))

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
# Same traversal as the other backends (mappings, sequences, namedtuples, dataclasses; func is
# applied to JAX arrays only); other registered pytrees are traversed with jax.tree.map.
map_fn_over_arrays = get_map_fn_over_arrays_function(
    is_backendarray=is_backendarray,
    fallback=lambda data, func: jax.tree.map(
        lambda x: func(x) if is_backendarray(x) else x,
        data
    ),
)

map_fn_over_arrays_ex = get_map_fn_over_arrays_ex_function(
    is_backendarray=is_backendarray,
    fallback=lambda data, *supp_data, func: jax.tree.map(
        lambda x, *s: func(x, *s) if is_backendarray(x) else x,
        data,
        *supp_data
    ),
)

pad_dim = get_pad_dim_function(
    backend=compat_module,
)