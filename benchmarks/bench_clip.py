"""Micro-benchmark for `clip` on the numpy backend, against the array_api_compat
implementation it used before (and plain `np.clip` for reference).

Run from the repository root:

    python benchmarks/bench_clip.py
"""
import os
import sys
import timeit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from array_api_compat import numpy as compat_numpy
from xbarray.backends.numpy import NumpyComputeBackend as B


def bench(fn, number=20_000):
    return min(timeit.repeat(fn, number=number, repeat=7)) / number * 1e6


if __name__ == "__main__":
    print(f"numpy {np.__version__}")
    for size in [(), (16,), (100_000,)]:
        number = 300 if size == (100_000,) else 20_000
        x = np.asarray(np.random.default_rng(0).uniform(-2, 2, size=size), dtype=np.float32)
        low = np.full(size, -1.0, dtype=np.float32)
        high = np.full(size, 1.0, dtype=np.float32)
        xi = np.arange(int(np.prod(size)), dtype=np.int64).reshape(size)
        cases = [
            ("f32, scalar bounds", lambda f: f(x, -1.0, 1.0)),
            ("f32, min only", lambda f: f(x, 0.0, None)),
            ("f32, array bounds", lambda f: f(x, low, high)),
            ("i64, int bounds", lambda f: f(xi, 2, 10)),
        ]
        for name, call in cases:
            old = bench(lambda: call(compat_numpy.clip), number)
            new = bench(lambda: call(B.clip), number)
            ref = bench(lambda: call(np.clip), number)
            print(
                f"{str(size):<10} {name:<20} compat {old:8.2f} us -> backend {new:8.2f} us "
                f"({old / new:4.1f}x)   np.clip {ref:8.2f} us"
            )
