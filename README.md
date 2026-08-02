# nexus

## Build

Requires:
- MLIR/LLVM installation, with `MLIR_DIR` and `LLVM_DIR` environment variables set
- clang, lld, ccache, and ninja

Configure and build:

```bash
cmake --workflow --preset debug   # or release
```

## Test

Run the configured build's regression tests (lit + FileCheck) directly:

```bash
cmake --build --preset debug-check   # or release-check
```

Or configure, build, and test in one step:

```bash
cmake --workflow --preset debug-test   # or release-test
```

Run a single test file:

```bash
llvm-lit -v build/compiler/test/Dialect/NN/show-dialects.mlir
```
