#include "CAPI/RegisterEverything.h"
#include "InitNexusDialects.h"

#include "mlir/CAPI/IR.h"

void mlirRegisterNexusDialects(MlirDialectRegistry registry) {
  mlir::nxs::registerNexusDialects(*unwrap(registry));
}
