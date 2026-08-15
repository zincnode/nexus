#ifndef CAPI_REGISTEREVERYTHING_H
#define CAPI_REGISTEREVERYTHING_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Appends all nexus dialects to the dialect registry.
MLIR_CAPI_EXPORTED void mlirRegisterNexusDialects(MlirDialectRegistry registry);

/// Registers all nexus passes.
MLIR_CAPI_EXPORTED void mlirRegisterNexusPasses();

#ifdef __cplusplus
}
#endif

#endif // CAPI_REGISTEREVERYTHING_H
