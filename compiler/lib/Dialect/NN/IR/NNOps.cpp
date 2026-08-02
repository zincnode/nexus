#include "Dialect/NN/IR/NNOps.h"
#include "Dialect/NN/IR/NNOpsDialect.cpp.inc"

void mlir::nxs::NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/NN/IR/NNOps.cpp.inc"
      >();
}
