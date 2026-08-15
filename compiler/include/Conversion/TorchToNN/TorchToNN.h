#ifndef CONVERSION_TORCHTONN_TORCHTONN_H
#define CONVERSION_TORCHTONN_TORCHTONN_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
namespace nxs {

#define GEN_PASS_DECL_CONVERTTORCHTONN
#include "Conversion/Passes.h.inc"

} // namespace nxs
} // namespace mlir

#endif // CONVERSION_TORCHTONN_TORCHTONN_H
