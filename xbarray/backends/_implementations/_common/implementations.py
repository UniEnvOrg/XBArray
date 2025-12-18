from typing import Any, Union, Callable, Mapping, Sequence
from array_api_typing.typing_compat import ArrayAPINamespace as CompatNamespace, ArrayAPIArray as CompatArray, ArrayAPIDType as CompatDType
import array_api_compat
import dataclasses

__all__ = [
    "get_abbreviate_array_function",
    "get_map_fn_over_arrays_function",
    "get_pad_dim_function",
]

def get_abbreviate_array_function(
    backend : CompatNamespace[CompatArray, Any, Any],
    default_integer_dtype : CompatDType,
    func_dtype_is_real_floating : Callable[[CompatDType], bool],
    func_dtype_is_real_integer : Callable[[CompatDType], bool],
    func_dtype_is_boolean : Callable[[CompatDType], bool],
):
    def abbreviate_array(array : CompatArray, try_cast_scalar : bool = True) -> Union[float, int, CompatArray]:
        """
        Abbreivates an array to a single element if possible.
        Or, if some dimensions are the same, abbreviates to a smaller array (but with the same number of dimensions).
        """
        abbr_array = array
        idx = backend.zeros(1, dtype=default_integer_dtype, device=array_api_compat.device(abbr_array))
        for dim_i in range(len(array.shape)):
            first_elem = backend.take(abbr_array, idx, axis=dim_i)
            if backend.all(abbr_array == first_elem):
                abbr_array = first_elem
            else:
                continue
        if try_cast_scalar:
            if all(i == 1 for i in abbr_array.shape):
                elem = abbr_array[tuple([0] * len(abbr_array.shape))]
                if func_dtype_is_real_floating(elem.dtype):
                    return float(elem)
                elif func_dtype_is_real_integer(elem.dtype):
                    return int(elem)
                elif func_dtype_is_boolean(elem.dtype):
                    return bool(elem)
                else:
                    raise ValueError(f"Abbreviated array element dtype must be a real floating or integer or boolean type, actual dtype: {elem.dtype}")
        else:
            return array
    return abbreviate_array

def get_map_fn_over_arrays_function(
    is_backendarray : Callable[[Any], bool],
):
    def map_fn_over_arrays(data : Any, func : Callable[[CompatArray], CompatArray]) -> Any:
        """
        Map a function to the data.
        """
        if is_backendarray(data):
            return func(data)
        elif isinstance(data, Mapping):
            ret = {k: map_fn_over_arrays(v, func) for k, v in data.items()}
            try:
                return type(data)(**ret)  # try to keep the same mapping type
            except:
                return ret
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            ret = [map_fn_over_arrays(i, func) for i in data]
            try:
                return type(data)(ret)  # try to keep the same sequence type
            except:
                return ret
        elif dataclasses.is_dataclass(data):
            return type(data)(**map_fn_over_arrays(dataclasses.asdict(data), func))
        else:
            return data
    return map_fn_over_arrays

def get_pad_dim_function(
    backend : CompatNamespace[CompatArray, Any, Any],
):
    def pad_dim(x : CompatArray, dim : int, target_size : int, value : Union[float, int] = 0) -> CompatArray:
        size_at_target = x.shape[dim]
        if size_at_target > target_size:
            raise ValueError(f"Cannot pad dimension {dim} of size {size_at_target} to smaller target size {target_size}")
        if size_at_target == target_size:
            return x
        target_shape = list(x.shape)
        target_shape[dim] = target_size - size_at_target
        pad_value = backend.full(target_shape, value, dtype=x.dtype, device=array_api_compat.device(x))
        return backend.concat([x, pad_value], axis=dim)
    return pad_dim