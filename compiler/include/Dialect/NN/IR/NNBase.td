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

// Semantic op traits, used as documentation-only markers to group ops into
// categories. Backed by empty trait classes in `::mlir::nxs::OpTrait`.
class NN_OpTrait<string name> : NativeOpTrait<""> {
  let trait = name;
  let cppNamespace = "::mlir::nxs::OpTrait";
}

def NN_Elementwise : NN_OpTrait<"Elementwise">;
def NN_Reduction : NN_OpTrait<"Reduction">;
def NN_LinearAlgebra : NN_OpTrait<"LinearAlgebra">;
def NN_Convolution : NN_OpTrait<"Convolution">;
def NN_Normalization : NN_OpTrait<"Normalization">;
def NN_Pooling : NN_OpTrait<"Pooling">;
def NN_Resampling : NN_OpTrait<"Resampling">;
def NN_ShapeManipulation : NN_OpTrait<"ShapeManipulation">;

#endif // DIALECT_NN_IR_NNBASE_TD
