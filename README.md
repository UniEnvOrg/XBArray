# xbarray
Cross-backend Python array library based on the Array API Standard.

This allows you to write array transformations that can run on different backends like NumPy, PyTorch, and Jax.

## Installation

```bash
pip install xbarray
```

## Usage:

Abstract typing:

```python
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from typing import Generic

class Behavior(Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    def __init__(self, backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]) -> None:
        self.backend = backend

    def create_array(self) -> BArrayType:
        return self.backend.zeros(5, dtype=self.backend.default_floating_dtype)
```

Concrete usage:

```python
from xbarray.backends.pytorch import PyTorchComputeBackend

behavior_pytorch_instance = Behavior(PyTorchComputeBackend)
behavior_pytorch_array = behavior_pytorch_instance.create_array()
```

## Cite

XBArray is developed as part of the UniEnv project.
If you use XBArray in your research, please cite it as follows:

```bibtex
@software{cao_unienv,
  author = {Cao, Yunhao AND Fang, Kuan},
  title = {{UniEnv: Unifying Robot Environments and Data APIs}},
  year = {2025},
  month = oct,
  url = {https://github.com/UniEnvOrg/UniEnv},
  license = {MIT}
}
```