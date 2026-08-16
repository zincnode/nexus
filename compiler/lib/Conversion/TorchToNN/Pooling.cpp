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

class ConvertAtenMaxPool2dOp : public OpConversionPattern<AtenMaxPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenMaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto kernelSize = getI64ListAttr(rewriter, loc, adaptor.getKernelSize());
    auto paddings = getI64ListAttr(rewriter, loc, adaptor.getPadding());
    auto dilations = getI64ListAttr(rewriter, loc, adaptor.getDilation());
    if (failed(kernelSize) || failed(paddings) || failed(dilations))
      return failure();
    // An empty stride list means "use the kernel size" in torch semantics.
    FailureOr<DenseI64ArrayAttr> strides =
        getI64ListAttr(rewriter, loc, adaptor.getStride());
    if (failed(strides))
      return failure();
    if ((*strides).empty())
      strides = *kernelSize;
    auto ceilMode = adaptor.getCeilMode().getDefiningOp<ConstantBoolOp>();
    if (!ceilMode)
      return rewriter.notifyMatchFailure(loc, "expected constant ceil_mode");

    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::MaxPool2dOp>(
        op, resultType, adaptor.getSelf(), *kernelSize, *strides, *paddings,
        *dilations, ceilMode.getValue());
    return success();
  }
};

class ConvertAtenAdaptiveAvgPool2dOp
    : public OpConversionPattern<AtenAdaptiveAvgPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenAdaptiveAvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputSize =
        getI64ListAttr(rewriter, op.getLoc(), adaptor.getOutputSize());
    if (failed(outputSize))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::AdaptiveAvgPool2dOp>(
        op, resultType, adaptor.getSelf(), *outputSize);
    return success();
  }
};

} // namespace

namespace mlir {
namespace nxs {

void populateTorchToNNPoolingPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  patterns.add<ConvertAtenMaxPool2dOp, ConvertAtenAdaptiveAvgPool2dOp>(
      typeConverter, patterns.getContext());
}

} // namespace nxs
} // namespace mlir
