"""Dialect Python bindings.

Exposes the ODS-generated op classes for the nexus ``nn`` dialect and the
vendored upstream MLIR dialect bindings. Submodules are imported lazily so that
``import nexus`` stays cheap.
"""

# vendored upstream MLIR dialects + nexus dialects
__all__ = [
    "affine",
    "amdgpu",
    "arith",
    "async_dialect",
    "bufferization",
    "builtin",
    "cf",
    "complex",
    "emitc",
    "func",
    "gpu",
    "index",
    "irdl",
    "linalg",
    "llvm",
    "math",
    "memref",
    "ml_program",
    "nn",  # nexus dialect
    "nvgpu",
    "nvvm",
    "openacc",
    "openmp",
    "pdl",
    "quant",
    "rocdl",
    "scf",
    "shape",
    "shard",
    "smt",
    "sparse_tensor",
    "spirv",
    "tensor",
    "tosa",
    "transform",
    "ub",
    "vector",
    "x86",
]


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
