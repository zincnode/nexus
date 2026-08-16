import argparse
import io

import nexus.ir as nir
import torch
from nexus.fx_utils import export_and_import_for_backend
from nexus.passmanager import PassManager
from torch import nn
from ultralytics import YOLO

NN_PIPELINE = (
    "builtin.module("
    " torch-function-to-torch-backend-pipeline{decompose-complex-ops=false},"
    " func.func(CONVERT_TORCH_TO_NN),"
    " canonicalize, cse)"
)


class YOLO26Features(nn.Module):
    """Backbone + PAN neck of the predefined ultralytics YOLO26n model.

    Reuses the module graph as-is (layers 0-22) and stops before the Detect
    head, whose NMS/DFL decode (arange/meshgrid/topk/gather/...) is outside the
    scope of the NN dialect. The output is the deepest P5/32 feature map.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model[0:23]

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            y.append(x)
        return y[-1]


def main():
    parser = argparse.ArgumentParser(description="YOLO26n feature extractor example")
    parser.add_argument(
        "--inline-literals",
        action="store_true",
        help="materialize tensor-literal weights as inline dense attrs instead "
        "of dense_resource references",
    )
    parser.add_argument(
        "--save-nn-ir",
        metavar="PATH",
        help="save the lowered NN IR (textual MLIR) to PATH",
    )
    args = parser.parse_args()

    convert = (
        "convert-torch-to-nn{inline-tensor-literals}"
        if args.inline_literals
        else "convert-torch-to-nn"
    )

    model = YOLO26Features(YOLO("yolo26n.yaml").model).eval()
    module = export_and_import_for_backend(model, torch.randn(1, 3, 640, 640))
    print("yolo26 import OK")
    print("verify OK" if module.operation.verify() else "verify FAILED")

    buf = io.BytesIO()
    module.operation.write_bytecode(buf)

    with nir.Context() as ctx:
        lowered = nir.Module.parse(buf.getvalue(), context=ctx)
        pm = PassManager.parse(NN_PIPELINE.replace("CONVERT_TORCH_TO_NN", convert))
        pm.run(lowered.operation)
        assert lowered.operation.verify()
        print("NN pipeline OK")
        if args.save_nn_ir:
            with open(args.save_nn_ir, "w") as f:
                f.write(lowered.operation.get_asm())
            print(f"NN IR saved to {args.save_nn_ir}")


if __name__ == "__main__":
    main()
