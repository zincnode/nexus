cmake_minimum_required(VERSION 3.25.0)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(nexus VERSION 0.1.0 LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR} ${MLIR_CMAKE_DIR}
                             ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(AddLLVM)
include(AddMLIR)
include(NexusPolicies)
nexus_apply_forward_policies()
include(HandleLLVMOptions)

# Configure the nexus Python bindings (see cmake/NexusPython.cmake).
include(NexusPython)
nexus_configure_python_bindings()

set(MLIR_BINARY_DIR ${PROJECT_BINARY_DIR})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/lib)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/compiler/include)
include_directories(${PROJECT_BINARY_DIR}/compiler/include)

# Vendor torch-mlir as the PyTorch frontend (see cmake/NexusTorchMLIR.cmake).
include(NexusTorchMLIR)
nexus_vendor_torch_mlir()

add_subdirectory(compiler)
