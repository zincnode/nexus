#include "Dialect/NN/IR/NNOps.h"
#include "Dialect/NN/IR/NNOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/NN/IR/NNOps.cpp.inc"

void mlir::nxs::NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/NN/IR/NNOps.cpp.inc"
      >();
}
