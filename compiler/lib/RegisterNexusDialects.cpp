#include "Dialect/NN/IR/NNOps.h"
#include "InitNexusDialects.h"

/// Add all the MLIR dialects to the provided registry.
void mlir::nxs::registerNexusDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    mlir::nxs::NNDialect
  >();
  // clang-format on
}

/// Append all the MLIR dialects to the registry contained in the given context.
void mlir::nxs::registerNexusDialects(MLIRContext &context) {
  DialectRegistry registry;
  registerNexusDialects(registry);
  context.appendDialectRegistry(registry);
}
