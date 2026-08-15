#ifndef DIALECT_NN_IR_NNOPS_H
#define DIALECT_NN_IR_NNOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/NN/IR/NNOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Dialect/NN/IR/NNOps.h.inc"

#endif // DIALECT_NN_IR_NNOPS_H
