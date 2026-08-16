#include "Conversion/TorchToNN/TorchToNN.h"

#include "Dialect/NN/IR/NNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
namespace nxs = mlir::nxs;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace nxs;

namespace {

class ConvertAtenReluOp : public OpConversionPattern<AtenReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::ReluOp>(op, resultType, adaptor.getSelf());
    return success();
  }
};

class ConvertAtenAddTensorOp : public OpConversionPattern<AtenAddTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenAddTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Only alpha == 1 is supported; the op degenerates to a plain add.
    if (auto alpha = adaptor.getAlpha().getDefiningOp<ConstantIntOp>()) {
      if (static_cast<int64_t>(alpha.getValue()) != 1)
        return rewriter.notifyMatchFailure(loc, "expected alpha=1");
    } else if (auto alpha =
                   adaptor.getAlpha().getDefiningOp<ConstantFloatOp>()) {
      if (alpha.getValue().convertToDouble() != 1.0)
        return rewriter.notifyMatchFailure(loc, "expected alpha=1");
    } else {
      return rewriter.notifyMatchFailure(loc, "expected constant alpha");
    }

    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::AddOp>(op, resultType, adaptor.getSelf(),
                                            adaptor.getOther());
    return success();
  }
};

class ConvertAtenSiluOp : public OpConversionPattern<AtenSiluOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenSiluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::SiluOp>(op, resultType,
                                             adaptor.getSelf());
    return success();
  }
};

class ConvertAtenMulScalarOp : public OpConversionPattern<AtenMulScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenMulScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = getTypeConverter()->convertType(op.getType());
    auto tensorTy = dyn_cast<RankedTensorType>(resultType);
    if (!tensorTy)
      return rewriter.notifyMatchFailure(loc, "expected ranked tensor result");
    Value scalar =
        materializeScalarTensor(rewriter, loc, tensorTy.getElementType(),
                                adaptor.getOther());
    if (!scalar)
      return rewriter.notifyMatchFailure(loc, "expected constant scalar");
    rewriter.replaceOpWithNewOp<nxs::MulOp>(op, resultType, adaptor.getSelf(),
                                            scalar);
    return success();
  }
};

class ConvertAtenMulTensorOp : public OpConversionPattern<AtenMulTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenMulTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::MulOp>(op, resultType, adaptor.getSelf(),
                                            adaptor.getOther());
    return success();
  }
};

} // namespace

namespace mlir {
namespace nxs {

void populateTorchToNNElementwisePatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  patterns.add<ConvertAtenReluOp, ConvertAtenAddTensorOp, ConvertAtenSiluOp,
               ConvertAtenMulScalarOp, ConvertAtenMulTensorOp>(
      typeConverter, patterns.getContext());
}

} // namespace nxs
} // namespace mlir
