cmake_minimum_required(VERSION 3.25.0)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

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

option(NEXUS_ENABLE_PYTHON_BINDINGS "Enable the nexus Python bindings" OFF)

if(NEXUS_ENABLE_PYTHON_BINDINGS)
  # Locate the Python interpreter (whatever is active at configure time) and
  # the nanobind package from it.
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()

  # Provides declare_mlir_python_sources / add_mlir_python_modules & friends.
  include(AddMLIRPython)

  # Re-root the vendored MLIR Python bindings under the `nexus` package.
  set(MLIR_PYTHON_PACKAGE_PREFIX "nexus" CACHE STRING
      "Top-level python package for the vendored MLIR bindings")
  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "nexus" CACHE STRING
      "Nanobind domain for the nexus python bindings")

  # Root of the MLIR source tree; used to vendor the MLIR Python sources.
  list(GET MLIR_INCLUDE_DIRS 0 MLIR_SOURCE_INCLUDE_DIR)
  get_filename_component(MLIR_SOURCE_DIR "${MLIR_SOURCE_INCLUDE_DIR}" DIRECTORY)

  if(NOT NEXUS_PYTHON_PACKAGES_DIR)
    set(NEXUS_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python_packages/nexus"
        CACHE PATH "Directory to assemble the nexus python package into")
  endif()
endif()

set(MLIR_BINARY_DIR ${PROJECT_BINARY_DIR})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/lib)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/compiler/include)
include_directories(${PROJECT_BINARY_DIR}/compiler/include)

add_subdirectory(compiler)
