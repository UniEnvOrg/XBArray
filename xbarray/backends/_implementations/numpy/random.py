from typing import Union, Optional, Tuple, Any
from ._typing import ARRAY_TYPE, DTYPE_TYPE, DEVICE_TYPE, RNG_TYPE
import math
import numpy as np

__all__ = [
    "random_number_generator",
    "random_discrete_uniform",
    "random_uniform",
    "random_exponential",
    "random_normal",
    "random_geometric",
    "random_permutation"
]

_FLOAT32 = np.dtype(np.float32)
_FLOAT32_MAX = float(np.finfo(_FLOAT32).max)
# Below this many elements, scaling a float32 draw in place costs more (ufunc overhead)
# than drawing float64 with `Generator.uniform` and casting.
_MIN_SIZE_FOR_FLOAT32_AFFINE = 128

def random_number_generator(
    seed : Optional[int] = None,
    *,
    device : Optional[DEVICE_TYPE] = None
) -> RNG_TYPE:
    return np.random.default_rng(seed)

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
    if dtype is not None and np.dtype(dtype).kind in "iub":
        # Generate directly in the requested dtype; raises if the bounds do not fit
        t = rng.integers(int(from_num), int(to_num), size=shape, dtype=dtype)
    else:
        t = rng.integers(int(from_num), int(to_num), size=shape)
        if dtype is not None:
            t = t.astype(dtype, copy=False)
    return rng, t

def random_uniform(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    rng : RNG_TYPE, 
    low : float = 0.0, high : float = 1.0,
    dtype : Optional[DTYPE_TYPE] = None, 
    device : Optional[DEVICE_TYPE] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    low, high = float(low), float(high)
    if dtype is not None and dtype == _FLOAT32:
        # Generator.random supports float32 natively (other float dtypes go through float64 below)
        if low == 0.0 and high == 1.0:
            return rng, rng.random(size=shape, dtype=np.float32)
        if (
            (shape if isinstance(shape, int) else math.prod(shape)) >= _MIN_SIZE_FOR_FLOAT32_AFFINE
            and max(abs(low), abs(high), abs(high - low)) <= _FLOAT32_MAX
        ):
            t = rng.random(size=shape, dtype=np.float32)
            t *= high - low
            t += low
            return rng, t
    t = rng.uniform(low, high, size=shape)
    if dtype is not None:
        t = t.astype(dtype, copy=False)
    return rng, t

def random_exponential(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    rng : RNG_TYPE,  
    lambd : float = 1.0, 
    dtype : Optional[DTYPE_TYPE] = None, 
    device : Optional[DEVICE_TYPE] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    t = rng.exponential(1.0 / float(lambd), size=shape)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t

def random_normal(
    shape: Union[int, Tuple[int, ...]],
    /,
    *, 
    rng : RNG_TYPE,
    mean : float = 0.0, std : float = 1.0, 
    dtype : Optional[DTYPE_TYPE] = None, 
    device : Optional[Any] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    t = rng.normal(mean, std, size=shape)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t


def random_geometric(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    p: float, 
    rng: RNG_TYPE,
    dtype: Optional[DTYPE_TYPE] = None, 
    device: Optional[Any] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    t = rng.geometric(p, size=shape)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t

def random_permutation(
    n : int,
    /,
    *,
    rng: RNG_TYPE,
    dtype: Optional[DTYPE_TYPE] = None,
    device: Optional[DEVICE_TYPE] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    t = rng.permutation(n)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t