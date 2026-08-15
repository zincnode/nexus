#include "CAPI/RegisterEverything.h"
#include "InitNexusDialects.h"
#include "InitNexusPasses.h"

#include "mlir/CAPI/IR.h"

void mlirRegisterNexusDialects(MlirDialectRegistry registry) {
  mlir::nxs::registerNexusDialects(*unwrap(registry));
}

void mlirRegisterNexusPasses() { mlir::nxs::registerNexusPasses(); }
