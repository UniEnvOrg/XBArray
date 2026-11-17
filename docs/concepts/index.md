# Concepts

These pages explain the parts of XBArray you use in every program.

- [Backends and the `ComputeBackend` interface](backends.md): what a backend class
  contains, where its functions come from, and how to pick one.
- [Generic typing](typing.md): the `BArrayType`, `BDeviceType`, `BDtypeType` and
  `BRNGType` type variables, and how to annotate backend-agnostic code.
- [Devices and dtypes](devices-and-dtypes.md): dtype objects, the default dtypes of
  each backend, and how devices are passed, read and serialized.
- [Random number generation](random.md): explicit generator objects and the
  `(rng, array)` return convention.
- [Converting between backends](interop.md): NumPy round trips, DLPack transfers, and
  mapping functions over nested data.

If you have not installed XBArray yet, start with [Getting started](../getting-started.md).
