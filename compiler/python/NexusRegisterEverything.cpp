#include "CAPI/RegisterEverything.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "torch-mlir-c/Registration.h"

NB_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "nexus and upstream dialect, translation and pass registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
    mlirRegisterNexusDialects(registry);
  });

  // Torch-mlir registers its dialects per-context (the torch dialect is not
  // part of the shared dialect registry, and torch-mlir's own
  // registerAllDialects would re-insert upstream dialects from a different
  // library copy).
  m.def("context_init_hook",
        [](MlirContext context) { torchMlirRegisterAllDialects(context); });

  m.def("register_llvm_translations",
        [](MlirContext context) { mlirRegisterAllLLVMTranslations(context); });

  // Register all passes on load.
  mlirRegisterAllPasses();
  mlirRegisterNexusPasses();
  torchMlirRegisterAllPasses();
}
