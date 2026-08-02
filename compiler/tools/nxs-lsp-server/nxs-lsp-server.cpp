#include "InitNexusDialects.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::nxs::registerNexusDialects(registry);

  return failed(mlir::MlirLspServerMain(argc, argv, registry));
}
