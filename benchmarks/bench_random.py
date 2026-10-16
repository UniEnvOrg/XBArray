"""Micro-benchmark for typed random draws on the numpy backend.

Run from the repository root:

    python benchmarks/bench_random.py
"""
import os
import sys
import timeit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import xbarray
from xbarray.backends.numpy import NumpyComputeBackend as B

CASES = [
    ("random_discrete_uniform", np.int32, lambda rng, shape, dtype: B.random.random_discrete_uniform(shape, 0, 10, rng=rng, dtype=dtype)),
    ("random_discrete_uniform", np.int64, lambda rng, shape, dtype: B.random.random_discrete_uniform(shape, 0, 10, rng=rng, dtype=dtype)),
    ("random_uniform", np.float32, lambda rng, shape, dtype: B.random.random_uniform(shape, rng=rng, dtype=dtype)),
    ("random_uniform", np.float32, lambda rng, shape, dtype: B.random.random_uniform(shape, rng=rng, low=-1.0, high=1.0, dtype=dtype)),
    ("random_uniform", np.float64, lambda rng, shape, dtype: B.random.random_uniform(shape, rng=rng, dtype=dtype)),
]
SHAPES = [(), (16,), (100_000,)]


def bench(fn, number):
    t = min(timeit.repeat(fn, number=number, repeat=7))
    return t / number * 1e6


if __name__ == "__main__":
    print(f"numpy {np.__version__}, xbarray from {os.path.dirname(xbarray.__file__)}")
    for idx, (name, dtype, call) in enumerate(CASES):
        for shape in SHAPES:
            rng = B.random.random_number_generator(0)
            number = 200 if shape == (100_000,) else 20_000
            us = bench(lambda: call(rng, shape, dtype), number)
            print(f"case {idx} {name:<24} {np.dtype(dtype).name:<8} shape={str(shape):<10} {us:9.2f} us")
