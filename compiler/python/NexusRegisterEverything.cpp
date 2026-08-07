#include "CAPI/RegisterEverything.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

NB_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "nexus and upstream dialect, translation and pass registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
    mlirRegisterNexusDialects(registry);
  });
  m.def("register_llvm_translations",
        [](MlirContext context) { mlirRegisterAllLLVMTranslations(context); });

  // Register all passes on load.
  mlirRegisterAllPasses();
}
