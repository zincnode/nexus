#ifndef DIALECT_NN_IR_NNBASE_TD
#define DIALECT_NN_IR_NNBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def NN_Dialect : Dialect {
  let name = "nn";
  let description = [{

  }];
  let cppNamespace = "::mlir::nxs";
}

class NN_Op<string mnemonic, list<Trait> traits = []>
    : Op<NN_Dialect, mnemonic, traits>;

#endif // DIALECT_NN_IR_NNBASE_TD
