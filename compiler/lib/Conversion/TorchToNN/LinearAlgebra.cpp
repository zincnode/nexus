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

class ConvertAtenLinearOp : public OpConversionPattern<AtenLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::LinearOp>(
        op, resultType, adaptor.getInput(), adaptor.getWeight(),
        optionalTensorOrNone(adaptor.getBias()));
    return success();
  }
};

class ConvertAtenMatmulOp : public OpConversionPattern<AtenMatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::MatmulOp>(op, resultType,
                                               adaptor.getSelf(),
                                               adaptor.getOther());
    return success();
  }
};

} // namespace

namespace mlir {
namespace nxs {

void populateTorchToNNLinearAlgebraPatterns(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  patterns.add<ConvertAtenLinearOp, ConvertAtenMatmulOp>(
      typeConverter, patterns.getContext());
}

} // namespace nxs
} // namespace mlir
