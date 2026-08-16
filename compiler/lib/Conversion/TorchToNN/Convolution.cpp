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

class ConvertAtenConv2dOp : public OpConversionPattern<AtenConv2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenConv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto strides = getI64ListAttr(rewriter, loc, adaptor.getStride());
    auto paddings = getI64ListAttr(rewriter, loc, adaptor.getPadding());
    auto dilations = getI64ListAttr(rewriter, loc, adaptor.getDilation());
    if (failed(strides) || failed(paddings) || failed(dilations))
      return failure();
    int64_t groups = 1;
    if (failed(getConstantInt(rewriter, loc, adaptor.getGroups(), groups)))
      return failure();

    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::Conv2dOp>(
        op, resultType, adaptor.getInput(), adaptor.getWeight(),
        optionalTensorOrNone(adaptor.getBias()), *strides, *paddings,
        *dilations, rewriter.getI64IntegerAttr(groups));
    return success();
  }
};

} // namespace

namespace mlir {
namespace nxs {

void populateTorchToNNConvolutionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  patterns.add<ConvertAtenConv2dOp>(typeConverter, patterns.getContext());
}

} // namespace nxs
} // namespace mlir
