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

class ConvertAtenBatchNormOp : public OpConversionPattern<AtenBatchNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenBatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Only inference-mode batch norms with all parameters present are
    // supported.
    auto training = adaptor.getTraining().getDefiningOp<ConstantBoolOp>();
    if (!training || training.getValue())
      return rewriter.notifyMatchFailure(loc, "expected training=false");
    auto eps = adaptor.getEps().getDefiningOp<ConstantFloatOp>();
    if (!eps)
      return rewriter.notifyMatchFailure(loc, "expected constant eps");
    // Optional parameters arrive as torch.constant.none.
    if (!optionalTensorOrNone(adaptor.getWeight()) ||
        !optionalTensorOrNone(adaptor.getBias()) ||
        !optionalTensorOrNone(adaptor.getRunningMean()) ||
        !optionalTensorOrNone(adaptor.getRunningVar()))
      return rewriter.notifyMatchFailure(loc, "expected all BN parameters");

    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::BatchNormOp>(
        op, resultType, adaptor.getInput(), adaptor.getWeight(),
        adaptor.getBias(), adaptor.getRunningMean(), adaptor.getRunningVar(),
        rewriter.getF64FloatAttr(eps.getValue().convertToDouble()));
    return success();
  }
};

} // namespace

namespace mlir {
namespace nxs {

void populateTorchToNNNormalizationPatterns(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  patterns.add<ConvertAtenBatchNormOp>(typeConverter, patterns.getContext());
}

} // namespace nxs
} // namespace mlir
