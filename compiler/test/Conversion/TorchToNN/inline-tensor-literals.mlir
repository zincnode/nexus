// RUN: nxs-opt %s --convert-torch-to-nn | FileCheck %s --check-prefix=RESOURCE
// RUN: nxs-opt %s --convert-torch-to-nn=inline-tensor-literals | FileCheck %s --check-prefix=INLINE

// tensor-literal weights can be kept as dense_resource references (default) or
// materialized as inline dense attributes (--inline-tensor-literals).

module {
  func.func @main(%arg0: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,4],f32> {
    %int1 = torch.constant.int 1
    %lit = torch.vtensor.literal(dense_resource<tiny_4xf32> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    %0 = torch.aten.add.Tensor %arg0, %lit, %int1 : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[1,4],f32>
    return %0 : !torch.vtensor<[1,4],f32>
  }
}
{-#
  dialect_resources: {
    builtin: {
      tiny_4xf32: "0x040000000000803F0000803F0000803F0000803F"
    }
  }
#-}

// RESOURCE-LABEL: func.func @main
// RESOURCE: arith.constant dense_resource<tiny_4xf32> : tensor<4xf32>
// RESOURCE-NOT: dense<

// INLINE-LABEL: func.func @main
// INLINE: arith.constant dense<1.000000e+00> : tensor<4xf32>
// INLINE-NOT: dense_resource
