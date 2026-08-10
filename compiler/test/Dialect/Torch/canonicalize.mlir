// RUN: nxs-opt %s --canonicalize | FileCheck %s
// Ensure the torch-mlir frontend dialects and passes are usable from nxs-opt.

// CHECK-LABEL: func.func @torch.aten.__range_length$fold
// CHECK-DAG: %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG: %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG: %[[INT3:.*]] = torch.constant.int 3
// CHECK-DAG: %[[INTM1:.*]] = torch.constant.int -1
// CHECK: %[[NEG_STEP:.*]] = torch.aten.__range_length %[[INT1]], %[[INT3]], %[[INTM1]] : !torch.int, !torch.int, !torch.int -> !torch.int
// CHECK: return %[[INT2]], %[[INT2]], %[[INT1]], %[[NEG_STEP]] : !torch.int, !torch.int, !torch.int, !torch.int
func.func @torch.aten.__range_length$fold() -> (!torch.int, !torch.int, !torch.int, !torch.int) {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %int-1 = torch.constant.int -1
  %0 = torch.aten.__range_length %int0, %int4, %int2  : !torch.int, !torch.int, !torch.int -> !torch.int
  %1 = torch.aten.__range_length %int1, %int4, %int2  : !torch.int, !torch.int, !torch.int -> !torch.int
  %2 = torch.aten.__range_length %int1, %int3, %int2  : !torch.int, !torch.int, !torch.int -> !torch.int
  %3 = torch.aten.__range_length %int1, %int3, %int-1  : !torch.int, !torch.int, !torch.int -> !torch.int
  return %0, %1, %2, %3 : !torch.int, !torch.int, !torch.int, !torch.int
}
