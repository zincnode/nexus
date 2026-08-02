#include "InitNexusDialects.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::nxs::registerNexusDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "nexus optimizer driver\n", registry));
}
