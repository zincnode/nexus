//===- Passes.td - Nexus conversion pass definitions --------*- tablegen -*-===//

#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TorchToNN
//===----------------------------------------------------------------------===//

def ConvertTorchToNN : Pass<"convert-torch-to-nn", "func::FuncOp"> {
  let summary = "Convert the Torch backend contract IR to the NN dialect";
  let description = [{
    Converts the closed set of Torch ops produced by the
    `torch-function-to-torch-backend-pipeline` (with
    `decompose-complex-ops=false`) into NN dialect ops. `!torch.vtensor` types
    are converted to standard `tensor<...>` types, and torch constants /
    `torch.prim.ListConstruct` are folded into op attributes.

    The conversion is strict: any remaining Torch dialect op (e.g. from a
    decomposed batch norm or an unconverted variant) fails the pass.
  }];
  let dependentDialects = ["mlir::nxs::NNDialect"];
  let options = [
    Option<"inlineTensorLiterals", "inline-tensor-literals", "bool", "false",
           "Materialize tensor-literal weights as inline dense attributes "
           "instead of dense_resource references">,
  ];
}

#endif // CONVERSION_PASSES_TD
