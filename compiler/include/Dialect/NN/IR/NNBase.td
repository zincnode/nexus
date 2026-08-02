#ifndef DIALECT_NN_IR_NNBASE_TD
#define DIALECT_NN_IR_NNBASE_TD

include "mlir/IR/OpBase.td"

def NN_Dialect : Dialect {
  let name = "nn";
  let description = [{

  }];
  let cppNamespace = "::mlir::nxs";
}

#endif // DIALECT_NN_IR_NNBASE_TD
