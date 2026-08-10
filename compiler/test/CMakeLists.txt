# Define the default arguments to use by `lit` when testing.
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

set(NEXUS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(NEXUS_BINARY_DIR ${PROJECT_BINARY_DIR})

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Python package dirs for the lit site config: the nexus package and the
# vendored torch_mlir frontend package (import root: .../python_packages/torch_mlir).
set(NEXUS_LIT_PYTHON_PACKAGES_DIR "${NEXUS_PYTHON_PACKAGES_DIR}")
set(TORCH_MLIR_LIT_PYTHON_PACKAGES_DIR "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir")

# `llvm-lit` lives in the (pre-built) LLVM/MLIR install, not in this build
# tree. LLVMConfig.cmake does not export LLVM_RUNTIME_OUTPUT_INTDIR, so point
# the lit launcher lookup at the LLVM tools dir explicitly.
set_property(GLOBAL PROPERTY LLVM_LIT_BASE_DIR "${LLVM_TOOLS_BINARY_DIR}")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

# Only nexus targets go here: the FileCheck/not/count CMake targets do not
# exist in this out-of-tree build; those tools are resolved at runtime from
# the LLVM tools dir via PATH.
set(NEXUS_TEST_DEPENDS
  nxs-opt
  )

if(NEXUS_ENABLE_PYTHON_BINDINGS)
  list(APPEND NEXUS_TEST_DEPENDS NexusPythonModules)
endif()
list(APPEND NEXUS_TEST_DEPENDS TorchMLIRPythonModules)

add_lit_testsuite(check-nexus "Running the nexus regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${NEXUS_TEST_DEPENDS}
  )
set_target_properties(check-nexus PROPERTIES FOLDER "Tests")
