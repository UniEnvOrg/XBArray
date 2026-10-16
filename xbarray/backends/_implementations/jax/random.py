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

def random_number_generator(
    seed : Optional[int] = None,
    *,
    device : Optional[DEVICE_TYPE] = None
) -> RNG_TYPE:
    rng_seed = np.random.randint(65535) if seed is None else seed
    rng = jax.random.key(
        seed=rng_seed
    )
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
    t = jax.random.randint(rng, _as_shape(shape), minval=int(from_num), maxval=int(to_num), dtype=dtype or int)
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
    data = jax.random.geometric(rng, p=p, shape=_as_shape(shape), dtype=dtype or int)
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
    data = jax.random.permutation(rng, n, dtype=dtype or int)
    if device is not None:
        data = jax.device_put(data, device)
    return new_rng, data