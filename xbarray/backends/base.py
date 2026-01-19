from typing import Optional, Generic, TypeVar, Dict, Union, Any, Sequence, SupportsFloat, Tuple, Type, Callable, Mapping, Protocol, Literal
import abc
from array_api_typing.typing_extra import *
import numpy as np

ArrayAPISetIndex = SetIndex
ArrayAPIGetIndex = GetIndex

__all__ = [
    "RNGBackend",
    "ComputeBackend",
    "BArrayType",
    "BDeviceType",
    "BDtypeType",
    "BRNGType",
    "SupportsDLPack",
    "ArrayAPIArray",
    "ArrayAPIDevice",
    "ArrayAPIDType",
    "ArrayAPINamespace",
    "ArrayAPISetIndex",
    "ArrayAPIGetIndex",
]

BArrayType = TypeVar("BArrayType", covariant=True, bound=ArrayAPIArray)
BDeviceType = TypeVar("BDeviceType", covariant=True, bound=ArrayAPIDevice)
BDtypeType = TypeVar("BDtypeType", covariant=True, bound=ArrayAPIDType)
BRNGType = TypeVar("BRNGType", covariant=True)
class RNGBackend(Protocol[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    @abc.abstractmethod
    def random_number_generator(
        self, 
        seed : Optional[int] = None, 
        *,
        device : Optional[BDeviceType] = None
    ) -> BRNGType:
        raise NotImplementedError
    
    @abc.abstractmethod
    def random_discrete_uniform(
        self, 
        shape : Union[int, Tuple[int, ...]], 
        /,
        from_num : int, 
        to_num : int,
        *,
        rng : BRNGType, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None, 
    ) -> Tuple[BRNGType, BArrayType]:
        """
        Sample from a discrete uniform distribution [from_num, to_num) with shape `shape`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def random_uniform(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        /,
        *,
        rng : BRNGType, 
        low : float = 0.0, high : float = 1.0, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None,
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_exponential(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        /,
        *,
        rng : BRNGType, 
        lambd : float = 1.0, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None, 
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_normal(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        /,
        *,
        rng : BRNGType, 
        mean : float = 0.0, std : float = 1.0, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None,
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_geometric(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        /,
        *, 
        p : float, 
        rng : BRNGType, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None,
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_permutation(
        self, 
        n : int, 
        /,
        *, 
        rng : BRNGType, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

class ComputeBackend(ArrayAPINamespace[BArrayType, BDeviceType, BDtypeType], Protocol[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    simplified_name : Literal['numpy', 'jax', 'pytorch']
    ARRAY_TYPE : Type[BArrayType]
    DEVICE_TYPE : Type[BDeviceType]
    DTYPE_TYPE : Type[BDtypeType]
    RNG_TYPE : Type[BRNGType]
    default_integer_dtype : BDtypeType
    default_index_dtype : BDtypeType
    default_floating_dtype : BDtypeType
    default_boolean_dtype : BDtypeType
    random : RNGBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]

    @abc.abstractmethod
    def serialize_device(self, device : Optional[BDeviceType]) -> Optional[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def deserialize_device(self, device_str : Optional[str]) -> Optional[BDeviceType]:
        raise NotImplementedError

    @abc.abstractmethod
    def is_backendarray(self, data : Any) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def from_numpy(
        self, 
        data : np.ndarray, 
        /,
        *,
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None
    ) -> BArrayType:
        raise NotImplementedError
    
    @abc.abstractmethod
    def from_other_backend(
        self,
        other_backend : "ComputeBackend",
        data : ArrayAPIArray,
        /
    ) -> BArrayType:
        """
        Convert an array from another backend to this backend.
        The other backend must be compatible with the ArrayAPI.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_numpy(
        self, 
        data : BArrayType
    ) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dlpack(
        self, 
        data : BArrayType, 
        /,
    ) -> SupportsDLPack:
        raise NotImplementedError

    @abc.abstractmethod
    def dtype_is_real_integer(self, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def dtype_is_real_floating(self, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def dtype_is_boolean(self, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def abbreviate_array(self, x : BArrayType, try_cast_scalar: bool = True) -> Union[float, int, BArrayType]:
        """
        Abbreviates an array to a single element if possible.
        Or, if some dimensions are the same, abbreviates to a smaller array (but with the same number of dimensions).
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def map_fn_over_arrays(self, data : Any, func : Callable[[BArrayType], BArrayType]) -> Any:
        """
        Map a function over arrays in a data structure and produce a new data structure with the same shape.
        This is useful for applying a function to all arrays in a nested structure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pad_dim(self, x : BArrayType, dim : int, target_size : int, value : Union[float, int] = 0) -> BArrayType:
        """
        Pad a dimension of an array to a target size with a given value.
        If the dimension is already the target size, return the original array.
        If the dimension is larger than the target size, raise an error.
        """
        raise NotImplementedError