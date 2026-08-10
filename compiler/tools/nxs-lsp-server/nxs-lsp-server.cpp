#include "InitNexusDialects.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "torch-mlir/InitAll.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::nxs::registerNexusDialects(registry);
  mlir::torch::registerAllDialects(registry);
  mlir::torch::registerAllExtensions(registry);
  mlir::torch::registerOptionalInputDialects(registry);

  return failed(mlir::MlirLspServerMain(argc, argv, registry));
}
