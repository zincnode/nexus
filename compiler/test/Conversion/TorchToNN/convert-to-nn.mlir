// RUN: nxs-opt %s "--torch-function-to-torch-backend-pipeline=decompose-complex-ops=false" --convert-torch-to-nn | FileCheck %s

// A hand-written backend-contract resnet50-shaped slice covering all NN ops.

module {
  func.func @maxpool_none_stride(%arg0: !torch.vtensor<[1,3,8,8],f32>) -> !torch.vtensor<[1,3,3,3],f32> {
    %int3 = torch.constant.int 3
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %c3 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %c1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %empty = torch.prim.ListConstruct  : () -> !torch.list<int>
    %0 = torch.aten.max_pool2d %arg0, %c3, %empty, %c1, %c1, %false : !torch.vtensor<[1,3,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,3,3],f32>
    return %0 : !torch.vtensor<[1,3,3,3],f32>
  }

  func.func @main(%arg0: !torch.vtensor<[1,3,224,224],f32>, %w1: !torch.vtensor<[64,3,7,7],f32>, %b1: !torch.vtensor<[64],f32>, %rm1: !torch.vtensor<[64],f32>, %rv1: !torch.vtensor<[64],f32>, %g1: !torch.vtensor<[64],f32>, %bb1: !torch.vtensor<[64],f32>, %w2: !torch.vtensor<[64,64,1,1],f32>, %b2: !torch.vtensor<[64],f32>, %rm2: !torch.vtensor<[64],f32>, %rv2: !torch.vtensor<[64],f32>, %g2: !torch.vtensor<[64],f32>, %bb2: !torch.vtensor<[64],f32>, %wf: !torch.vtensor<[1000,64],f32>, %bf: !torch.vtensor<[1000],f32>) -> !torch.vtensor<[1,1000],f32> {
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %false = torch.constant.bool false
    %none = torch.constant.none
    %lit = torch.vtensor.literal(dense<1.000000e+00> : tensor<64xf32>) : !torch.vtensor<[64],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %intm1 = torch.constant.int -1
    %c2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %c3 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %c1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %c0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %0 = torch.aten.conv2d %arg0, %w1, %b1, %c2, %c3, %c1, %int1 : !torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,112,112],f32>
    %1 = torch.aten.batch_norm %0, %g1, %bb1, %rm1, %rv1, %false, %float1.000000e-05, %float1.000000e-05, %false : !torch.vtensor<[1,64,112,112],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[1,64,112,112],f32>
    %2 = torch.aten.relu %1 : !torch.vtensor<[1,64,112,112],f32> -> !torch.vtensor<[1,64,112,112],f32>
    %3 = torch.aten.max_pool2d %2, %c3, %c2, %c1, %c1, %false : !torch.vtensor<[1,64,112,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,64,56,56],f32>
    %4 = torch.aten.conv2d %3, %w2, %none, %c1, %c0, %c1, %int1 : !torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64,64,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,56,56],f32>
    %5 = torch.aten.batch_norm %4, %lit, %bb2, %rm2, %rv2, %false, %float1.000000e-05, %float1.000000e-05, %false : !torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[1,64,56,56],f32>
    %6 = torch.aten.relu %5 : !torch.vtensor<[1,64,56,56],f32> -> !torch.vtensor<[1,64,56,56],f32>
    %7 = torch.aten.add.Tensor %6, %3, %int1 : !torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[1,64,56,56],f32>, !torch.int -> !torch.vtensor<[1,64,56,56],f32>
    %8 = torch.aten.adaptive_avg_pool2d %7, %c1 : !torch.vtensor<[1,64,56,56],f32>, !torch.list<int> -> !torch.vtensor<[1,64,1,1],f32>
    %9 = torch.aten.flatten.using_ints %8, %int1, %intm1 : !torch.vtensor<[1,64,1,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,64],f32>
    %10 = torch.aten.linear %9, %wf, %bf : !torch.vtensor<[1,64],f32>, !torch.vtensor<[1000,64],f32>, !torch.vtensor<[1000],f32> -> !torch.vtensor<[1,1000],f32>
    return %10 : !torch.vtensor<[1,1000],f32>
  }
}

// CHECK-LABEL: func.func @maxpool_none_stride
// CHECK-SAME: (%arg0: tensor<1x3x8x8xf32>
// CHECK-SAME: -> tensor<1x3x3x3xf32>
// CHECK: nn.max_pool2d {{.*}} {ceil_mode = false, dilations = array<i64: 1, 1>, kernel_size = array<i64: 3, 3>, paddings = array<i64: 1, 1>, strides = array<i64: 3, 3>}
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<1x3x224x224xf32>
// CHECK-SAME: -> tensor<1x1000xf32>
// CHECK: arith.constant dense<1.000000e+00> : tensor<64xf32>
// CHECK: nn.conv2d {{.*}} {dilations = array<i64: 1, 1>, groups = 1 : i64, paddings = array<i64: 3, 3>, strides = array<i64: 2, 2>}
// CHECK: nn.batch_norm {{.*}} {eps = 1.000000e-05 : f64}
// CHECK: nn.relu
// CHECK: nn.max_pool2d {{.*}} {ceil_mode = false, dilations = array<i64: 1, 1>, kernel_size = array<i64: 3, 3>, paddings = array<i64: 1, 1>, strides = array<i64: 2, 2>}
// CHECK: nn.conv2d {{.*}} {dilations = array<i64: 1, 1>, groups = 1 : i64, paddings = array<i64: 0, 0>, strides = array<i64: 1, 1>}
// CHECK: nn.add
// CHECK: nn.adaptive_avg_pool2d {{.*}} {output_size = array<i64: 1, 1>}
// CHECK: nn.flatten {{.*}} {end_dim = -1 : i64, start_dim = 1 : i64}
// CHECK: nn.linear
// CHECK-NOT: torch.
