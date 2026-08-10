# RUN: %PYTHON% %s | FileCheck %s

# Verifies the torch-mlir frontend package builds/imports alongside the nexus
# package, and that a PyTorch module imports to torch dialect IR.

import torch
from torch import nn
from torch_mlir import fx

# CHECK: torch_mlir frontend OK
print("torch_mlir frontend OK")


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 4)

    def forward(self, x):
        return self.lin(x).relu()


m = M().eval()
module = fx.export_and_import(m, torch.ones(1, 3))
text = str(module)

# CHECK-LABEL: module {
# CHECK: func.func @main(%arg0: !torch.vtensor<[1,3],f32>) -> !torch.vtensor<[1,4],f32>
# CHECK: torch.aten.linear
# CHECK-SAME: !torch.vtensor<[1,3],f32>
# CHECK: torch.aten.relu
# CHECK-SAME: !torch.vtensor<[1,4],f32>
print(text)
