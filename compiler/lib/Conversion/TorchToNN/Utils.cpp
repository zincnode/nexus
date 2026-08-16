#include "Conversion/TorchToNN/TorchToNN.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {
namespace nxs {

FailureOr<DenseI64ArrayAttr>
getI64ListAttr(ConversionPatternRewriter &rewriter, Location loc, Value v) {
  SmallVector<Value> elems;
  if (!getListConstructElements(v, elems))
    return rewriter.notifyMatchFailure(loc, "expected a ListConstruct");
  SmallVector<int64_t> ints;
  ints.reserve(elems.size());
  for (Value e : elems) {
    auto cst = e.getDefiningOp<ConstantIntOp>();
    if (!cst)
      return rewriter.notifyMatchFailure(loc, "expected constant int");
    ints.push_back(static_cast<int64_t>(cst.getValue()));
  }
  return rewriter.getDenseI64ArrayAttr(ints);
}

LogicalResult getConstantInt(ConversionPatternRewriter &rewriter, Location loc,
                             Value v, int64_t &result) {
  auto cst = v.getDefiningOp<ConstantIntOp>();
  if (!cst)
    return rewriter.notifyMatchFailure(loc, "expected constant int");
  result = static_cast<int64_t>(cst.getValue());
  return success();
}

Value optionalTensorOrNone(Value v) {
  if (!v)
    return Value();
  if (isa_and_nonnull<ConstantNoneOp>(v.getDefiningOp()))
    return Value();
  return v;
}

Value materializeScalarTensor(ConversionPatternRewriter &rewriter,
                              Location loc, Type elemType, Value scalar) {
  auto type = RankedTensorType::get({1}, elemType);
  Attribute attr;
  if (auto cst = scalar.getDefiningOp<ConstantFloatOp>()) {
    double value = cst.getValue().convertToDouble();
    if (auto intTy = dyn_cast<mlir::IntegerType>(elemType))
      attr = rewriter.getIntegerAttr(intTy, static_cast<int64_t>(value));
    else
      attr = rewriter.getFloatAttr(elemType, value);
  } else if (auto cst = scalar.getDefiningOp<ConstantIntOp>()) {
    int64_t value = static_cast<int64_t>(cst.getValue());
    if (isa<mlir::FloatType>(elemType))
      attr = rewriter.getFloatAttr(elemType, static_cast<double>(value));
    else
      attr = rewriter.getIntegerAttr(elemType, cst.getValue());
  } else {
    return Value();
  }
  return arith::ConstantOp::create(rewriter, loc, type,
                                   DenseElementsAttr::get(type, {attr}));
}

} // namespace nxs
} // namespace mlir
