#include "InitNexusDialects.h"
#include "InitNexusPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "torch-mlir/InitAll.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::nxs::registerNexusPasses();
  mlir::torch::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::nxs::registerNexusDialects(registry);

  mlir::torch::registerAllDialects(registry);
  mlir::torch::registerAllExtensions(registry);
  mlir::torch::registerOptionalInputDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "nexus optimizer driver\n", registry));
}
