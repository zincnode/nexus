import io

import nexus.ir as nir
import torch
from nexus.fx_utils import export_and_import_for_backend
from nexus.passmanager import PassManager
from torchvision.models import resnet50

NN_PIPELINE = (
    "builtin.module("
    " torch-function-to-torch-backend-pipeline{decompose-complex-ops=false},"
    " func.func(convert-torch-to-nn),"
    " canonicalize, cse)"
)


def main():
    model = resnet50(weights=None).eval()
    module = export_and_import_for_backend(model, torch.randn(1, 3, 224, 224))
    print("resnet50 import OK")
    print("verify OK" if module.operation.verify() else "verify FAILED")

    buf = io.BytesIO()
    module.operation.write_bytecode(buf)

    with nir.Context() as ctx:
        lowered = nir.Module.parse(buf.getvalue(), context=ctx)
        pm = PassManager.parse(NN_PIPELINE)
        pm.run(lowered.operation)
        assert lowered.operation.verify()
        print("NN pipeline OK")


if __name__ == "__main__":
    main()
