#ifndef CONVERSION_TORCHTONN_TORCHTONN_H
#define CONVERSION_TORCHTONN_TORCHTONN_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
namespace nxs {

#define GEN_PASS_DECL_CONVERTTORCHTONN
#include "Conversion/Passes.h.inc"

// Helper to extract an int64 list from a torch.prim.ListConstruct of
// torch.constant.int values.
FailureOr<DenseI64ArrayAttr> getI64ListAttr(ConversionPatternRewriter &rewriter,
                                            Location loc, Value v);

// Helper to extract a constant int from a torch.constant.int value.
LogicalResult getConstantInt(ConversionPatternRewriter &rewriter, Location loc,
                             Value v, int64_t &result);

// Returns the value if it is a real tensor, or a null Value when it is a
// torch.constant.none.
Value optionalTensorOrNone(Value v);

// Materializes a scalar constant (float or int) as a rank-1 tensor so it can
// feed the tensor-tensor mul op. The scalar is converted to the tensor's
// element type, mirroring torch's scalar-to-tensor dtype promotion.
Value materializeScalarTensor(ConversionPatternRewriter &rewriter, Location loc,
                              Type elemType, Value scalar);

// Populates the conversion patterns for each semantic op category.
void populateTorchToNNConvolutionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
void populateTorchToNNNormalizationPatterns(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns);
void populateTorchToNNElementwisePatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
void populateTorchToNNPoolingPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns);
void populateTorchToNNShapeManipulationPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns);
void populateTorchToNNLinearAlgebraPatterns(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns);
void populateTorchToNNResamplingPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns);
void populateTorchToNNReductionPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns);

} // namespace nxs
} // namespace mlir

#endif // CONVERSION_TORCHTONN_TORCHTONN_H
