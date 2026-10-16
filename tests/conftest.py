import importlib

import pytest

_BACKENDS = {
    "numpy": ("xbarray.backends.numpy", "NumpyComputeBackend", "numpy"),
    "pytorch": ("xbarray.backends.pytorch", "PyTorchComputeBackend", "torch"),
    "jax": ("xbarray.backends.jax", "JaxComputeBackend", "jax"),
}


def load_backend(name):
    module_name, cls_name, requirement = _BACKENDS[name]
    pytest.importorskip(requirement)
    return getattr(importlib.import_module(module_name), cls_name)


@pytest.fixture(params=list(_BACKENDS))
def backend(request):
    return load_backend(request.param)


@pytest.fixture
def np_backend():
    return load_backend("numpy")
