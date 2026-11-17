# Development

## Set up

Clone the repository and install it in editable mode with the backends you want to
test. The test suite also needs `pytest`, which is not a declared dependency:

```bash
git clone https://github.com/UniEnvOrg/XBArray.git
cd XBArray
python -m venv .venv
source .venv/bin/activate
pip install -e ".[torch,jax]" pytest
```

Backends you leave out are skipped by the tests, so `pip install -e . pytest` is enough
to work on NumPy-only code.

## Repository layout

| Path | Contents |
| --- | --- |
| `xbarray/backends/base.py` | The `ComputeBackend` and `RNGBackend` protocols and the type variables |
| `xbarray/backends/numpy.py`, `pytorch.py`, `jax.py` | The backend classes |
| `xbarray/backends/_implementations/<backend>/` | The functions that become attributes of each backend class; `random.py` holds the random functions |
| `xbarray/backends/_implementations/_common/` | Shared implementations of `abbreviate_array`, `map_fn_over_arrays(_ex)` and `pad_dim` |
| `xbarray/backends/utils.py` | `get_backend_from_tensor` |
| `xbarray/transformations/<name>/base.py` | Generic transformations that take a backend argument |
| `xbarray/transformations/<name>/{numpy,pytorch,jax}.py` | The same functions bound to one backend |
| `array_api_typing/` | Typing protocols for the Array API, shipped with the package |
| `tests/` | The pytest suite |
| `benchmarks/` | Micro-benchmarks |

A backend class copies every public name of its implementation package onto itself
when the module is imported. To add a function to a backend, add it to that backend's
`_implementations/<backend>/_extra.py` (or `random.py`) and its `__all__`, and add it
to the protocol in `base.py`.

## Run the tests

From the repository root:

```bash
pytest
```

`pyproject.toml` sets `testpaths = ["tests"]` and puts the repository root on
`sys.path`, so the tests import the working copy.

The `backend` fixture in `tests/conftest.py` runs each test once per backend
(`numpy`, `pytorch`, `jax`). A backend whose library is not installed is skipped
through `pytest.importorskip`. Tests that need one backend use the `np_backend`
fixture or import the backend directly. The backend name is part of each
parametrised test ID, so `-k` selects by backend:

```bash
pytest -k jax                  # the JAX cases of the parametrised tests
pytest tests/test_random.py    # one file
```

The test files are:

| File | Covers |
| --- | --- |
| `tests/test_random.py` | Dtypes, shapes, bounds and distributions of the random functions; same-seed reproducibility |
| `tests/test_clip.py` | `clip` on every backend, and the NumPy `clip` against `array_api_compat` |
| `tests/test_point_sampling.py` | `random_point_sampling` with and without a validity mask |

`test_torch_full_range_on_device[cuda]` runs only when PyTorch sees a CUDA device.
To keep JAX on the CPU, set `JAX_PLATFORMS=cpu`.

## Run the benchmarks

The benchmarks are plain scripts. Run them from the repository root:

```bash
python benchmarks/bench_clip.py
python benchmarks/bench_random.py
```

Both measure the NumPy backend with `timeit` and print microseconds per call for a
scalar, a small array and a 100,000-element array:

- `bench_clip.py` compares `NumpyComputeBackend.clip` with `array_api_compat`'s `clip`
  and with `np.clip`.
- `bench_random.py` times `random_discrete_uniform` and `random_uniform` for several
  dtypes.

They insert the repository root into `sys.path`, so they measure the working copy
even without an install.

## Changelog and releases

Record user-visible changes under **Unreleased** in `CHANGELOG.md`. The version
number is in `pyproject.toml`. Publishing a GitHub release runs
`.github/workflows/python-publish.yml`, which builds the package and uploads it to
PyPI.
