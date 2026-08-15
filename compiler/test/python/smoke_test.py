# RUN: %PYTHON% %s | FileCheck %s

from nexus.dialects import arith, nn
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

# CHECK: module
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
        nn.AddOp(tensor_type, lhs, rhs)
    # CHECK: arith.constant
    # CHECK: nn.add
    print(module)
