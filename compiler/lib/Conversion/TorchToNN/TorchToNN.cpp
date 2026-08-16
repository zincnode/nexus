#include "Conversion/TorchToNN/TorchToNN.h"
#include "Dialect/NN/IR/NNOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
namespace nxs = mlir::nxs;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace nxs;

namespace mlir {
namespace nxs {
#define GEN_PASS_DEF_CONVERTTORCHTONN
#include "Conversion/Passes.h.inc"
} // namespace nxs
} // namespace mlir

// Converts !torch.vtensor<[sizes],dtype> to tensor<sizes x dtype>; all other
// types pass through unchanged.
static void populateTorchToNNTypeConverter(TypeConverter &typeConverter) {
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (auto vtensor = dyn_cast<ValueTensorType>(type)) {
      if (!vtensor.hasDtype() || !isBuiltInType(vtensor.getDtype()))
        return std::nullopt;
      return RankedTensorType::get(vtensor.getSizes(), vtensor.getDtype());
    }
    return type;
  });
}

namespace {

class ConvertValueTensorLiteralOp
    : public OpConversionPattern<ValueTensorLiteralOp> {
public:
  ConvertValueTensorLiteralOp(const TypeConverter &converter,
                              MLIRContext *context, bool inlineLiterals)
      : OpConversionPattern(converter, context), inlineLiterals(inlineLiterals) {}

  LogicalResult
  matchAndRewrite(ValueTensorLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    ElementsAttr value = op.getValue();
    // When requested, detach tensor-literal weights from their resource blob
    // so they print as inline `dense<...>` attrs instead of `dense_resource`.
    if (inlineLiterals) {
      if (auto resourceAttr =
              llvm::dyn_cast<DenseResourceElementsAttr>(value)) {
        value = DenseElementsAttr::getFromRawBuffer(resourceAttr.getShapedType(),
                                                    resourceAttr.getData());
      }
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, value);
    return success();
  }

private:
  bool inlineLiterals;
};

} // namespace

static void populateTorchToNNConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                bool inlineTensorLiterals) {
  populateTorchToNNConvolutionPatterns(typeConverter, patterns);
  populateTorchToNNNormalizationPatterns(typeConverter, patterns);
  populateTorchToNNElementwisePatterns(typeConverter, patterns);
  populateTorchToNNPoolingPatterns(typeConverter, patterns);
  populateTorchToNNShapeManipulationPatterns(typeConverter, patterns);
  populateTorchToNNLinearAlgebraPatterns(typeConverter, patterns);
  populateTorchToNNResamplingPatterns(typeConverter, patterns);
  populateTorchToNNReductionPatterns(typeConverter, patterns);
  patterns.add<ConvertValueTensorLiteralOp>(typeConverter, patterns.getContext(),
                                            inlineTensorLiterals);
}

namespace {
class ConvertTorchToNNPass
    : public nxs::impl::ConvertTorchToNNBase<ConvertTorchToNNPass> {
public:
  using nxs::impl::ConvertTorchToNNBase<
      ConvertTorchToNNPass>::ConvertTorchToNNBase;

protected:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    TypeConverter typeConverter;
    populateTorchToNNTypeConverter(typeConverter);
    // Bidirectional unrealized casts keep the conversion framework happy while
    // `!torch.list<vtensor>` operands (built by `prim.ListConstruct`) are
    // being unwrapped by the cat/slice patterns; any cast that survives only
    // feeds dead list ops and is dropped below.
    typeConverter.addSourceMaterialization(
        [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
          return UnrealizedConversionCastOp::create(builder, loc,
                                                    TypeRange(type), inputs)
              .getResult(0);
        });
    typeConverter.addTargetMaterialization(
        [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
          return UnrealizedConversionCastOp::create(builder, loc,
                                                    TypeRange(type), inputs)
              .getResult(0);
        });

    RewritePatternSet patterns(context);
    populateTorchToNNConversionPatterns(typeConverter, patterns,
                                        this->inlineTensorLiterals);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(*context);
    target.addLegalDialect<nxs::NNDialect, arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    // Convert the function signature when it still carries Torch tensor types.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });
    // Scalar constants and lists are consumed in-place by the patterns above
    // and are cleaned up below once dead.
    target.addLegalOp<ConstantIntOp, ConstantFloatOp, ConstantBoolOp,
                      ConstantNoneOp, PrimListConstructOp>();
    target.addIllegalDialect<Torch::TorchDialect>();

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();

    // Drop any scalar constants/lists/casts that are now dead (their values
    // were inlined into the NN ops) so no Torch ops leak into the output.
    // Iterate to a fixed point: a constant only becomes dead once its
    // ListConstruct user has been erased.
    SmallVector<Operation *> deadOps;
    bool changed = true;
    while (changed) {
      changed = false;
      deadOps.clear();
      func.walk([&](Operation *op) {
        if (isa<ConstantIntOp, ConstantFloatOp, ConstantBoolOp, ConstantNoneOp,
                PrimListConstructOp, UnrealizedConversionCastOp>(op) &&
            op->use_empty())
          deadOps.push_back(op);
      });
      for (Operation *op : deadOps) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace
