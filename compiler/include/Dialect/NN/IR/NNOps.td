#ifndef DIALECT_NN_IR_NNOPS_TD
#define DIALECT_NN_IR_NNOPS_TD

include "Dialect/NN/IR/NNBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Element-wise ops
//===----------------------------------------------------------------------===//

def NN_AddOp : NN_Op<"add", [Pure, NN_Elementwise]> {
  let summary = "Adds two tensors element-wise with broadcasting";
  let description = [{
    Follows torch/NumPy broadcasting semantics. Both operands and the result
    are tensors of the same element type.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

def NN_MulOp : NN_Op<"mul", [Pure, NN_Elementwise]> {
  let summary = "Multiplies two tensors element-wise with broadcasting";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

def NN_ReluOp : NN_Op<"relu", [Pure, NN_Elementwise]> {
  let summary = "Applies the rectified linear unit function";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

def NN_SiluOp : NN_Op<"silu", [Pure, NN_Elementwise]> {
  let summary = "Applies the SiLU (Swish) activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// Reduction ops
//===----------------------------------------------------------------------===//

def NN_SoftmaxOp : NN_Op<"softmax", [Pure, NN_Reduction]> {
  let summary = "Applies the softmax function over a dimension";
  let arguments = (ins AnyTensor:$input, I64Attr:$dim);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// Linear algebra ops
//===----------------------------------------------------------------------===//

def NN_MatmulOp : NN_Op<"matmul", [Pure, NN_LinearAlgebra]> {
  let summary = "Matrix multiplication with broadcasting";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

def NN_LinearOp : NN_Op<"linear", [Pure, NN_LinearAlgebra]> {
  let summary = "Applies an affine transformation to the last input dimension";
  let description = [{
    Computes `input @ weight^T + bias`. `bias` is optional.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    Optional<AnyTensor>:$bias
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input `,` $weight (`,` $bias^)? attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// Convolution ops
//===----------------------------------------------------------------------===//

def NN_Conv2dOp : NN_Op<"conv2d", [Pure, NN_Convolution]> {
  let summary = "2D convolution over a batched NCHW input";
  let description = [{
    Computes a 2D convolution with the given (constant) strides, paddings,
    dilations and groups. `bias` is optional; when present it is added to the
    convolution result.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    Optional<AnyTensor>:$bias,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$paddings,
    DenseI64ArrayAttr:$dilations,
    I64Attr:$groups
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input `,` $weight (`,` $bias^)? attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// Normalization ops
//===----------------------------------------------------------------------===//

def NN_BatchNormOp : NN_Op<"batch_norm", [Pure, NN_Normalization]> {
  let summary = "Batch normalization over a batched NCHW input";
  let description = [{
    Applies `(input - running_mean) / sqrt(running_var + eps) * weight +
    bias` in inference mode. All five tensor operands are required
    (affine, track_running_stats batch norms).
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$bias,
    AnyTensor:$running_mean,
    AnyTensor:$running_var,
    F64Attr:$eps
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input `,` $weight `,` $bias `,` $running_mean `,` $running_var attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// Pooling ops
//===----------------------------------------------------------------------===//

def NN_MaxPool2dOp : NN_Op<"max_pool2d", [Pure, NN_Pooling]> {
  let summary = "2D max pooling over a batched NCHW input";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$paddings,
    DenseI64ArrayAttr:$dilations,
    BoolAttr:$ceil_mode
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

def NN_AdaptiveAvgPool2dOp : NN_Op<"adaptive_avg_pool2d", [Pure, NN_Pooling]> {
  let summary = "Adaptive average pooling over a batched NCHW input";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$output_size
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// Resampling ops
//===----------------------------------------------------------------------===//

def NN_UpsampleOp : NN_Op<"upsample", [Pure, NN_Resampling]> {
  let summary = "Nearest-neighbor upsampling with scale factors";
  let arguments = (ins
    AnyTensor:$input,
    DenseF64ArrayAttr:$scale_factors
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// Shape manipulation ops
//===----------------------------------------------------------------------===//

def NN_FlattenOp : NN_Op<"flatten", [Pure, NN_ShapeManipulation]> {
  let summary = "Flattens a contiguous range of dimensions";
  let description = [{
    Flattens dims `[start_dim, end_dim]` into a single dimension, preserving
    the batch dimension. Negative dims are interpreted relative to the input
    rank, following torch semantics.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$start_dim,
    I64Attr:$end_dim
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

def NN_ReshapeOp : NN_Op<"reshape", [Pure, NN_ShapeManipulation]> {
  let summary = "Reshapes the input to a target shape";
  let arguments = (ins AnyTensor:$input, DenseI64ArrayAttr:$shape);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

def NN_TransposeOp : NN_Op<"transpose", [Pure, NN_ShapeManipulation]> {
  let summary = "Swaps two dimensions of the input";
  let arguments = (ins AnyTensor:$input, I64Attr:$dim0, I64Attr:$dim1);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

def NN_ConcatOp : NN_Op<"concat", [Pure, NN_ShapeManipulation]> {
  let summary = "Concatenates tensors along a dimension";
  let arguments = (ins Variadic<AnyTensor>:$inputs, I64Attr:$dim);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $inputs attr-dict `:` functional-type(operands, results) }];
}

def NN_SliceOp : NN_Op<"slice", [Pure, NN_ShapeManipulation]> {
  let summary = "Slices a tensor along a dimension";
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$dim,
    I64Attr:$start,
    I64Attr:$end,
    I64Attr:$step
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{ $input attr-dict `:` functional-type(operands, results) }];
}

#endif // DIALECT_NN_IR_NNOPS_TD
