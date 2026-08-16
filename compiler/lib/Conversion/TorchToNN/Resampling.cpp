#include "Conversion/TorchToNN/TorchToNN.h"

#include "Dialect/NN/IR/NNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
namespace nxs = mlir::nxs;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace nxs;

namespace {

class ConvertAtenUpsampleNearest2dOp
    : public OpConversionPattern<AtenUpsampleNearest2dVecOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenUpsampleNearest2dVecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Only the scale-factors form is supported.
    SmallVector<Value> elems;
    if (!adaptor.getScaleFactors() ||
        !getListConstructElements(adaptor.getScaleFactors(), elems))
      return rewriter.notifyMatchFailure(loc, "expected scale factors list");
    SmallVector<double> scales;
    scales.reserve(elems.size());
    for (Value e : elems) {
      auto cst = e.getDefiningOp<ConstantFloatOp>();
      if (!cst)
        return rewriter.notifyMatchFailure(loc, "expected constant scale");
      scales.push_back(cst.getValue().convertToDouble());
    }
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<nxs::UpsampleOp>(
        op, resultType, adaptor.getInput(),
        DenseF64ArrayAttr::get(rewriter.getContext(), scales));
    return success();
  }
};

} // namespace

namespace mlir {
namespace nxs {

void populateTorchToNNResamplingPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<ConvertAtenUpsampleNearest2dOp>(typeConverter,
                                               patterns.getContext());
}

} // namespace nxs
} // namespace mlir
