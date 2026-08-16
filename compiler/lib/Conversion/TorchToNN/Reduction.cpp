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

class ConvertAtenSoftmaxIntOp : public OpConversionPattern<AtenSoftmaxIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenSoftmaxIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t dim = 0;
    if (failed(getConstantInt(rewriter, loc, adaptor.getDim(), dim)))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::SoftmaxOp>(
        op, resultType, adaptor.getSelf(), rewriter.getI64IntegerAttr(dim));
    return success();
  }
};

} // namespace

namespace mlir {
namespace nxs {

void populateTorchToNNReductionPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  patterns.add<ConvertAtenSoftmaxIntOp>(typeConverter, patterns.getContext());
}

} // namespace nxs
} // namespace mlir
