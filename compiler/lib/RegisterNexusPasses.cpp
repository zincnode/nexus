#include "InitNexusPasses.h"

#include "Conversion/Passes.h"

void mlir::nxs::registerNexusPasses() {
  mlir::nxs::registerConversionPasses();
}
