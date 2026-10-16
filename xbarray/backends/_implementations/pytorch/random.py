from typing import Union, Optional, Tuple, Any
import operator
from ._typing import ARRAY_TYPE, DTYPE_TYPE, DEVICE_TYPE, RNG_TYPE
import torch

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
    rng = torch.Generator(
        device=device
    )
    if seed is not None:
        rng = rng.manual_seed(seed)
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
    t = torch.randint(int(from_num), int(to_num), _as_shape(shape), generator=rng, dtype=dtype, device=device)
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
    t = torch.rand(_as_shape(shape), generator=rng, dtype=dtype, device=device)
    t = t * (high - low) + low
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
    t = torch.empty(_as_shape(shape), dtype=dtype, device=device)
    t = t.exponential_(lambd, generator=rng)
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
    t = torch.normal(mean, std, _as_shape(shape), generator=rng, dtype=dtype, device=device)
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
    t = torch.empty(_as_shape(shape), dtype=dtype, device=device)
    t = t.geometric_(p, generator=rng)
    return rng, t

def random_permutation(
    n : int,
    /,
    *,
    rng: RNG_TYPE,
    dtype: Optional[DTYPE_TYPE] = None,
    device: Optional[DEVICE_TYPE] = None
) -> Tuple[RNG_TYPE, ARRAY_TYPE]:
    t = torch.randperm(n, generator=rng, dtype=dtype, device=device)
    return rng, t