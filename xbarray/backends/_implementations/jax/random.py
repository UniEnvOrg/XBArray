from typing import Union, Optional, Tuple, Any
import operator
from ._typing import ARRAY_TYPE, DTYPE_TYPE, DEVICE_TYPE, RNG_TYPE
import numpy as np
import jax

__all__ = [
    "random_number_generator",
    "random_discrete_uniform",
    "random_uniform",
    "random_exponential",
    "random_normal",
    "random_geometric",
    "random_permutation"
]

def _as_shape(shape : Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Normalise an int (incl. numpy integers) or a sequence of ints to a tuple of Python ints."""
    try:
        return (operator.index(shape),)
    except TypeError:
        return tuple(operator.index(dim) for dim in shape)

def _is_integer_dtype(dtype : Optional[DTYPE_TYPE]) -> bool:
    return dtype is None or np.dtype(dtype).kind in "iu"

def random_number_generator(
    seed : Optional[int] = None,
    *,
    device : Optional[DEVICE_TYPE] = None
) -> RNG_TYPE:
    # Unseeded: draw a 32-bit seed from numpy's global generator (so np.random.seed still
    # makes it reproducible), not one of only 65535 seeds
    rng_seed = int(np.random.randint(2 ** 32, dtype=np.uint64)) if seed is None else seed
    rng = jax.random.key(
        seed=rng_seed
    )
    if device is not None:
        rng = jax.device_put(rng, device)
    return rng

def random_discrete_uniform(
    shape : Union[int, Tuple[int, ...]], 
    /,
    from_num : int, 
    to_num : int, 
    *,
    rng : RNG_TYPE, 
    dtype : Optional[DTYPE_TYPE] = None, 
    device : Optional[DEVICE_TYPE] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    new_rng, rng = jax.random.split(rng)
    from_num, to_num = int(from_num), int(to_num)
    t = None
    if dtype is not None:
        # Materialised dtype: without x64, jax stores int64 as int32
        np_dtype = np.dtype(jax.numpy.empty((0,), dtype=dtype).dtype)
        if np_dtype.kind in "iu" and np_dtype.itemsize <= 4:
            info = np.iinfo(np_dtype)
            if from_num == int(info.min) and to_num == int(info.max) + 1:
                # The whole dtype range: raw random bits (``randint`` cannot take ``maxval``
                # = int32 max + 1, and is ~3x slower for uint8 images)
                t = jax.random.bits(rng, _as_shape(shape), dtype=np.dtype(f"uint{np_dtype.itemsize * 8}"))
                if np_dtype.kind == "i":
                    t = jax.lax.bitcast_convert_type(t, np_dtype)
    if t is None:
        if _is_integer_dtype(dtype):
            t = jax.random.randint(rng, _as_shape(shape), minval=from_num, maxval=to_num, dtype=dtype or int)
        else:
            # randint only produces integers: draw them, then cast (bool / floating dtypes, like numpy and PyTorch)
            t = jax.random.randint(rng, _as_shape(shape), minval=from_num, maxval=to_num, dtype=int).astype(dtype)
    if device is not None:
        t = jax.device_put(t, device)
    return new_rng, t

def random_uniform(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    rng : RNG_TYPE, 
    low : float = 0.0, high : float = 1.0,
    dtype : Optional[DTYPE_TYPE] = None, 
    device : Optional[DEVICE_TYPE] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    new_rng, rng = jax.random.split(rng)
    data = jax.random.uniform(rng, _as_shape(shape), dtype=dtype or float, minval=low, maxval=high)
    if (low != 0.0 or high != 1.0) and high > low:
        # Rounding can turn the largest draws into `high` itself: keep the interval half-open
        largest = jax.numpy.nextafter(jax.numpy.asarray(high, dtype=data.dtype), jax.numpy.asarray(low, dtype=data.dtype))
        data = jax.numpy.minimum(data, largest)
    if device is not None:
        data = jax.device_put(data, device)
    return new_rng, data

def random_exponential(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    rng : RNG_TYPE,  
    lambd : float = 1.0, 
    dtype : Optional[DTYPE_TYPE] = None, 
    device : Optional[DEVICE_TYPE] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    new_rng, rng = jax.random.split(rng)
    data = jax.random.exponential(rng, _as_shape(shape), dtype=dtype or float) / lambd
    if device is not None:
        data = jax.device_put(data, device)
    return new_rng, data

def random_normal(
    shape: Union[int, Tuple[int, ...]],
    /,
    *, 
    rng : RNG_TYPE,
    mean : float = 0.0, std : float = 1.0, 
    dtype : Optional[DTYPE_TYPE] = None, 
    device : Optional[Any] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    new_rng, rng = jax.random.split(rng)
    data = jax.random.normal(rng, _as_shape(shape), dtype=dtype or float) * std + mean
    if device is not None:
        data = jax.device_put(data, device)
    return new_rng, data

def random_geometric(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    p: float, 
    rng: RNG_TYPE,
    dtype: Optional[DTYPE_TYPE] = None, 
    device: Optional[Any] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    new_rng, rng = jax.random.split(rng)
    if _is_integer_dtype(dtype):
        data = jax.random.geometric(rng, p=p, shape=_as_shape(shape), dtype=dtype or int)
    else:
        data = jax.random.geometric(rng, p=p, shape=_as_shape(shape), dtype=int).astype(dtype)
    if device is not None:
        data = jax.device_put(data, device)
    return new_rng, data

def random_permutation(
    n : int,
    /,
    *,
    rng: RNG_TYPE,
    dtype: Optional[DTYPE_TYPE] = None,
    device: Optional[DEVICE_TYPE] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    new_rng, rng = jax.random.split(rng)
    # jax.random.permutation has no dtype argument
    data = jax.random.permutation(rng, n)
    if dtype is not None:
        data = data.astype(dtype)
    if device is not None:
        data = jax.device_put(data, device)
    return new_rng, data