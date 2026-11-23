import pytest

from conftest import load_backend


def test_torch_serialize_device_roundtrip():
    torch = pytest.importorskip("torch")
    B = load_backend("pytorch")
    for device in [torch.device("cpu"), torch.device("cuda"), torch.device("cuda", 0), torch.device("cpu", 0)]:
        assert B.deserialize_device(B.serialize_device(device)) == device
    assert B.serialize_device(torch.device("cuda")) == "cuda"  # the current device, not cuda:0
    assert B.serialize_device(torch.device("cuda", 0)) == "cuda:0"


def test_jax_serialize_device_roundtrip():
    jax = pytest.importorskip("jax")
    B = load_backend("jax")
    for device in jax.devices() + jax.devices("cpu"):
        assert B.deserialize_device(B.serialize_device(device)) == device
