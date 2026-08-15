// RUN: nxs-opt %s --convert-torch-to-nn --canonicalize | FileCheck %s

// Backend-contract YOLO26-shaped slice covering the ops added for the YOLO26
// feature extractor (no backend pipeline: reshape/view would be rewritten to
// dynamic torch shape functions).

module {
  func.func @yolo_ops(%arg0: !torch.vtensor<[1,32,80,80],f32>, %arg1: !torch.vtensor<[1,32,80,80],f32>) -> (!torch.vtensor<[1,48,80,80],f32>, !torch.vtensor<[1,16,160,160],f32>, !torch.vtensor<[1,16,80,80],f32>, !torch.vtensor<[1,80,16,80],f32>, !torch.vtensor<[1,80,80,80],f32>, !torch.vtensor<[1,102400],f32>, !torch.vtensor<[1,32,6400],f32>) {
    %float2 = torch.constant.float 2.000000e+00
    %float01 = torch.constant.float 1.000000e-01
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int16 = torch.constant.int 16
    %int32 = torch.constant.int 32
    %int6400 = torch.constant.int 6400
    %int102400 = torch.constant.int 102400
    %intm1 = torch.constant.int -1
    %scale = torch.prim.ListConstruct %float2, %float2 : (!torch.float, !torch.float) -> !torch.list<float>
    %shape1 = torch.prim.ListConstruct %int1, %int102400 : (!torch.int, !torch.int) -> !torch.list<int>
    %shape2 = torch.prim.ListConstruct %int1, %int32, %int6400 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %0 = torch.aten.silu %arg0 : !torch.vtensor<[1,32,80,80],f32> -> !torch.vtensor<[1,32,80,80],f32>
    %1 = torch.aten.slice.Tensor %arg1, %int1, %int0, %int16, %int1 : !torch.vtensor<[1,32,80,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,16,80,80],f32>
    %list = torch.prim.ListConstruct %1, %0 : (!torch.vtensor<[1,16,80,80],f32>, !torch.vtensor<[1,32,80,80],f32>) -> !torch.list<vtensor>
    %2 = torch.aten.mul.Scalar %1, %float01 : !torch.vtensor<[1,16,80,80],f32>, !torch.float -> !torch.vtensor<[1,16,80,80],f32>
    %3 = torch.aten.transpose.int %2, %int1, %int2 : !torch.vtensor<[1,16,80,80],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,80,16,80],f32>
    %4 = torch.aten.matmul %3, %2 : !torch.vtensor<[1,80,16,80],f32>, !torch.vtensor<[1,16,80,80],f32> -> !torch.vtensor<[1,80,80,80],f32>
    %5 = torch.aten.softmax.int %4, %intm1, %none : !torch.vtensor<[1,80,80,80],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,80,80,80],f32>
    %6 = torch.aten.cat %list, %int1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,48,80,80],f32>
    %7 = torch.aten.upsample_nearest2d.vec %1, %none, %scale : !torch.vtensor<[1,16,80,80],f32>, !torch.none, !torch.list<float> -> !torch.vtensor<[1,16,160,160],f32>
    %8 = torch.aten.reshape %1, %shape1 : !torch.vtensor<[1,16,80,80],f32>, !torch.list<int> -> !torch.vtensor<[1,102400],f32>
    %9 = torch.aten.view %0, %shape2 : !torch.vtensor<[1,32,80,80],f32>, !torch.list<int> -> !torch.vtensor<[1,32,6400],f32>
    return %6, %7, %2, %3, %5, %8, %9 : !torch.vtensor<[1,48,80,80],f32>, !torch.vtensor<[1,16,160,160],f32>, !torch.vtensor<[1,16,80,80],f32>, !torch.vtensor<[1,80,16,80],f32>, !torch.vtensor<[1,80,80,80],f32>, !torch.vtensor<[1,102400],f32>, !torch.vtensor<[1,32,6400],f32>
  }
}

// CHECK-LABEL: func.func @yolo_ops
// CHECK: arith.constant dense<1.000000e-01> : tensor<1xf32>
// CHECK: nn.silu
// CHECK: nn.slice {{.*}} {dim = 1 : i64, end = 16 : i64, start = 0 : i64, step = 1 : i64}
// CHECK: nn.mul
// CHECK: nn.transpose {{.*}} {dim0 = 1 : i64, dim1 = 2 : i64}
// CHECK: nn.matmul
// CHECK: nn.softmax {{.*}} {dim = -1 : i64}
// CHECK: nn.concat {{.*}} {dim = 1 : i64}
// CHECK: nn.upsample {{.*}} {scale_factors = array<f64: 2.000000e+00, 2.000000e+00>}
// CHECK: nn.reshape {{.*}} {shape = array<i64: 1, 102400>}
// CHECK: nn.reshape {{.*}} {shape = array<i64: 1, 32, 6400>}
// CHECK-NOT: torch.
