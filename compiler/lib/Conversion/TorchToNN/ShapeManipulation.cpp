#include "Conversion/TorchToNN/TorchToNN.h"

#include "Dialect/NN/IR/NNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include <climits>

using namespace mlir;
namespace nxs = mlir::nxs;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace nxs;

namespace {

class ConvertAtenFlattenUsingIntsOp
    : public OpConversionPattern<AtenFlattenUsingIntsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenFlattenUsingIntsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t startDim = 0;
    int64_t endDim = 0;
    if (failed(
            getConstantInt(rewriter, loc, adaptor.getStartDim(), startDim)) ||
        failed(getConstantInt(rewriter, loc, adaptor.getEndDim(), endDim)))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::FlattenOp>(
        op, resultType, adaptor.getSelf(), rewriter.getI64IntegerAttr(startDim),
        rewriter.getI64IntegerAttr(endDim));
    return success();
  }
};

class ConvertAtenCatOp : public OpConversionPattern<AtenCatOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenCatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> elems;
    if (!getListConstructElements(adaptor.getTensors(), elems))
      return rewriter.notifyMatchFailure(loc, "expected a ListConstruct");
    int64_t dim = 0;
    if (failed(getConstantInt(rewriter, loc, adaptor.getDim(), dim)))
      return failure();
    // List elements are not direct operands, so the framework does not
    // convert their types; materialize the tensor type for each element. Any
    // cast that remains becomes an identity once the producing op converts,
    // and is folded away by canonicalize.
    SmallVector<Value> converted;
    converted.reserve(elems.size());
    for (Value e : elems) {
      Type tensorType = getTypeConverter()->convertType(e.getType());
      if (!tensorType)
        return rewriter.notifyMatchFailure(loc, "unconvertible element type");
      Value c = getTypeConverter()->materializeTargetConversion(
          rewriter, loc, tensorType, e);
      if (!c)
        return rewriter.notifyMatchFailure(loc, "failed to materialize");
      converted.push_back(c);
    }
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::ConcatOp>(
        op, resultType, converted, rewriter.getI64IntegerAttr(dim));
    return success();
  }
};

class ConvertAtenSliceTensorOp : public OpConversionPattern<AtenSliceTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenSliceTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t dim = 0;
    if (failed(getConstantInt(rewriter, loc, adaptor.getDim(), dim)))
      return failure();
    // `start`/`end` default to 0 and the input extent when absent. Only
    // constant bounds are supported.
    int64_t start = 0;
    if (auto startOp = adaptor.getStart().getDefiningOp<ConstantIntOp>())
      start = static_cast<int64_t>(startOp.getValue());
    else if (!isa_and_nonnull<ConstantNoneOp>(adaptor.getStart().getDefiningOp()))
      return rewriter.notifyMatchFailure(loc, "expected constant start");
    int64_t end = INT64_MAX;
    if (auto endOp = adaptor.getEnd().getDefiningOp<ConstantIntOp>())
      end = static_cast<int64_t>(endOp.getValue());
    else if (!isa_and_nonnull<ConstantNoneOp>(adaptor.getEnd().getDefiningOp()))
      return rewriter.notifyMatchFailure(loc, "expected constant end");
    int64_t step = 1;
    if (failed(getConstantInt(rewriter, loc, adaptor.getStep(), step)))
      return failure();

    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::SliceOp>(
        op, resultType, adaptor.getSelf(), rewriter.getI64IntegerAttr(dim),
        rewriter.getI64IntegerAttr(start), rewriter.getI64IntegerAttr(end),
        rewriter.getI64IntegerAttr(step));
    return success();
  }
};

class ConvertAtenTransposeIntOp
    : public OpConversionPattern<AtenTransposeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenTransposeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    if (failed(getConstantInt(rewriter, loc, adaptor.getDim0(), dim0)) ||
        failed(getConstantInt(rewriter, loc, adaptor.getDim1(), dim1)))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::TransposeOp>(
        op, resultType, adaptor.getSelf(), rewriter.getI64IntegerAttr(dim0),
        rewriter.getI64IntegerAttr(dim1));
    return success();
  }
};

class ConvertAtenReshapeOp : public OpConversionPattern<AtenReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto shape = getI64ListAttr(rewriter, loc, adaptor.getShape());
    if (failed(shape))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::ReshapeOp>(op, resultType,
                                                adaptor.getSelf(), *shape);
    return success();
  }
};

class ConvertAtenViewOp : public OpConversionPattern<AtenViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto shape = getI64ListAttr(rewriter, loc, adaptor.getSize());
    if (failed(shape))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::ReshapeOp>(op, resultType,
                                                adaptor.getSelf(), *shape);
    return success();
  }
};

} // namespace

namespace mlir {
namespace nxs {

void populateTorchToNNShapeManipulationPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns) {
  patterns.add<ConvertAtenFlattenUsingIntsOp, ConvertAtenCatOp,
               ConvertAtenSliceTensorOp, ConvertAtenTransposeIntOp,
               ConvertAtenReshapeOp, ConvertAtenViewOp>(
      typeConverter, patterns.getContext());
}

} // namespace nxs
} // namespace mlir
