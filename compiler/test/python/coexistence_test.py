# RUN: %PYTHON% %s | FileCheck %s

# Verifies the nexus and torch_mlir packages can be loaded and used in the same
# process. Both vendor the same MLIR/LLVM into separate Python C-extensions;
# this guards against symbol/type-registration clashes when a real frontend
# imports both at once.

import torch
from nexus.dialects import arith
from nexus.dialects import nn as nnx
from nexus.ir import (
    Context,
    DenseElementsAttr,
    F32Type,
    FloatAttr,
    InsertionPoint,
    Location,
    Module,
    RankedTensorType,
)
from torch import nn
from torch_mlir import fx

# CHECK: coexistence OK
print("coexistence OK")


# A torch-mlir module (exercises the torch_mlir C-extension).
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 4)

    def forward(self, x):
        return self.lin(x).relu()


torch_module = fx.export_and_import(M().eval(), torch.ones(1, 3))
# CHECK: func.func @main
# CHECK: torch.aten.linear
print(torch_module)

# A nexus module (exercises the nexus C-extension after torch_mlir was loaded).
with Context() as ctx, Location.unknown():
    module = Module.create()
    tensor_type = RankedTensorType.get([4], F32Type.get())

    def const(vals):
        return arith.ConstantOp(
            DenseElementsAttr.get(
                [FloatAttr.get(F32Type.get(), v) for v in vals], type=tensor_type
            ),
            None,
        )

    with InsertionPoint(module.body):
        lhs = const([1.0, 2.0, 3.0, 4.0])
        rhs = const([5.0, 6.0, 7.0, 8.0])
        nnx.AddOp(tensor_type, lhs, rhs)
    # CHECK: arith.constant
    # CHECK: nn.add
    print(module)
