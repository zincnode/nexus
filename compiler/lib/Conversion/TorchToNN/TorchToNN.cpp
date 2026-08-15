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

// Helper to extract an int64 list from a torch.prim.ListConstruct of
// torch.constant.int values.
static FailureOr<DenseI64ArrayAttr>
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

// Helper to extract a constant int from a torch.constant.int value.
static LogicalResult getConstantInt(ConversionPatternRewriter &rewriter, Location loc,
                             Value v, int64_t &result) {
  auto cst = v.getDefiningOp<ConstantIntOp>();
  if (!cst)
    return rewriter.notifyMatchFailure(loc, "expected constant int");
  result = static_cast<int64_t>(cst.getValue());
  return success();
}

// Returns the value if it is a real tensor, or a null Value when it is a
// torch.constant.none.
static Value optionalTensorOrNone(Value v) {
  if (!v)
    return Value();
  if (isa_and_nonnull<ConstantNoneOp>(v.getDefiningOp()))
    return Value();
  return v;
}

// Materializes a scalar constant (float or int) as a rank-1 tensor so it can
// feed the tensor-tensor mul op. The scalar is converted to the tensor's
// element type, mirroring torch's scalar-to-tensor dtype promotion.
static Value materializeScalarTensor(ConversionPatternRewriter &rewriter,
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

static void populateTorchToNNConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                bool inlineTensorLiterals) {
  patterns.add<ConvertAtenConv2dOp, ConvertAtenBatchNormOp, ConvertAtenReluOp,
               ConvertAtenAddTensorOp, ConvertAtenMaxPool2dOp,
               ConvertAtenAdaptiveAvgPool2dOp, ConvertAtenFlattenUsingIntsOp,
               ConvertAtenLinearOp,
               ConvertAtenSiluOp, ConvertAtenMulScalarOp, ConvertAtenMulTensorOp,
               ConvertAtenCatOp, ConvertAtenSliceTensorOp,
               ConvertAtenUpsampleNearest2dOp, ConvertAtenMatmulOp,
               ConvertAtenSoftmaxIntOp, ConvertAtenTransposeIntOp,
               ConvertAtenReshapeOp, ConvertAtenViewOp>(
      typeConverter, patterns.getContext());
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
