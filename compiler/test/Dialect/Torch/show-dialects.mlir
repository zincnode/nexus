// RUN: nxs-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-DAG: nn
// CHECK-DAG: tm_tensor
// CHECK-DAG: torch
// CHECK-DAG: torch_c
