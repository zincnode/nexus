#ifndef DIALECT_NN_IR_NNOPS_TD
#define DIALECT_NN_IR_NNOPS_TD

include "Dialect/NN/IR/NNBase.td"

def NN_AddOp : NN_Op<"add"> {
  let summary = "Adds two tensor operands element-wise";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

#endif // DIALECT_NN_IR_NNOPS_TD
