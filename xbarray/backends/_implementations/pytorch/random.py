from typing import Union, Optional, Tuple, Any
import math
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

# Integer dtypes whose full range is drawn as raw random bits (see ``random_discrete_uniform``)
_FULL_RANGE_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

def _random_bits(
    shape : Tuple[int, ...],
    rng : RNG_TYPE,
    dtype : DTYPE_TYPE,
    device : Optional[DEVICE_TYPE],
) -> ARRAY_TYPE:
    """Uniform values over the whole range of the integer ``dtype``, drawn as 64-bit words.

    ``torch.randint`` costs the same per element whatever the dtype, so drawing
    ``itemsize / 8`` as many int64 words and reinterpreting their bytes is up to
    8x faster on CPU (uint8 images). ``randint``'s exclusive upper bound leaves
    out the single word ``2**63 - 1``, a bias of ``2**-64``.
    """
    n = math.prod(shape)
    itemsize = torch.empty((), dtype=dtype).element_size()
    words = torch.randint(
        -2 ** 63, 2 ** 63 - 1, (-(-n * itemsize // 8),),
        generator=rng, dtype=torch.int64, device=device,
    )
    return words.view(dtype)[:n].reshape(shape)

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
    from_num, to_num = int(from_num), int(to_num)
    if dtype in _FULL_RANGE_DTYPES:
        info = torch.iinfo(dtype)
        if from_num == info.min and to_num == info.max + 1:
            return rng, _random_bits(_as_shape(shape), rng, dtype, device)
    t = torch.randint(from_num, to_num, _as_shape(shape), generator=rng, dtype=dtype, device=device)
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
    if low != 0.0 or high != 1.0:
        # Two scalar ops cost ~3x the draw itself for small tensors: skip them for [0, 1)
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