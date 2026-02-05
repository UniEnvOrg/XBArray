from ._cls_base import ComputeBackendImplCls
from ._implementations import pytorch as pytorch_impl

class PyTorchComputeBackend(metaclass=ComputeBackendImplCls[pytorch_impl.ARRAY_TYPE, pytorch_impl.DEVICE_TYPE, pytorch_impl.DTYPE_TYPE, pytorch_impl.RNG_TYPE]):
    ARRAY_TYPE = pytorch_impl.ARRAY_TYPE
    DTYPE_TYPE = pytorch_impl.DTYPE_TYPE
    DEVICE_TYPE = pytorch_impl.DEVICE_TYPE
    RNG_TYPE = pytorch_impl.RNG_TYPE

for name in dir(pytorch_impl):
    if not name.startswith('_') or name in [
        '__array_namespace_info__',
        '__array_api_version__',
    ]:
        try:
            setattr(PyTorchComputeBackend, name, getattr(pytorch_impl, name))
        except AttributeError:
            pass

__all__ = [
    'PyTorchComputeBackend',
]