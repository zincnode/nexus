# Vendor torch-mlir (third_party/torch-mlir) as the PyTorch frontend, keeping
# the top-level CMakeLists clean. Call after find_package(MLIR) and before
# add_subdirectory(compiler).
#
# A function() is used here rather than a macro: every effect of this module is
# either a cache variable write or an add_subdirectory() call, neither of which
# depends on ordinary variables in the caller's scope; cache writes remain
# visible to the whole build tree after the function returns (e.g.
# TORCH_MLIR_PYTHON_PACKAGES_DIR is read in compiler/test), and
# add_subdirectory still attaches to the directory that called it. So a
# function scope does not swallow any effects.

# torch-mlir is configured out-of-tree against the same prebuilt LLVM/MLIR used
# by nexus.
#   - StableHLO disabled: avoids the nested externals/stablehlo submodule.
#   - PyTorch C++ extensions disabled: no libtorch dependency at build time;
#     the fx_importer only needs `torch` at Python runtime.

function(nexus_vendor_torch_mlir)
  set(TORCH_MLIR_OUT_OF_TREE_BUILD ON CACHE BOOL "" FORCE)
  set(TORCH_MLIR_USE_INSTALLED_PYTORCH ON CACHE BOOL "" FORCE)
  set(TORCH_MLIR_ENABLE_STABLEHLO OFF CACHE BOOL "" FORCE)
  set(TORCH_MLIR_ENABLE_TOSA ON CACHE BOOL "" FORCE)
  set(TORCH_MLIR_ENABLE_REFBACKEND OFF CACHE BOOL "" FORCE)
  set(TORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS OFF CACHE BOOL "" FORCE)
  set(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER OFF CACHE BOOL "" FORCE)
  set(TORCH_MLIR_ENABLE_LTC OFF CACHE BOOL "" FORCE)
  set(TORCH_MLIR_ENABLE_ONNX_C_IMPORTER OFF CACHE BOOL "" FORCE)
  # torch-mlir's python/CMakeLists.txt assembles the package under
  # ${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir, so the base dir
  # must be the shared python_packages root (import root: .../torch_mlir).
  set(TORCH_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python_packages"
      CACHE PATH "Base directory for the torch_mlir python package")

  # Give torch-mlir its own nanobind domain so its Python bindings are isolated
  # from the `nexus` bindings; restore `nexus` afterwards for compiler/python.
  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "torch_mlir" CACHE STRING "" FORCE)

  # torch-mlir is a third-party subproject: with developer warnings enabled (see
  # CMakePresets.json) it would spew "Policy CMPXXXX is not set" and
  # deprecation noise, so suppress warnings for its subtree only. The
  # subdirectory inherits the diagnostic state current at add_subdirectory
  # time, and block() pops it automatically at endblock().
  # Note: do not wrap cmake_diagnostic's PUSH/POP in a macro — a macro call
  # carries its own diagnostic stack entry, which would report "PUSH without
  # matching POP"; a function does not automatically recycle diagnostic
  # settings either. block() is the safest option.
  if(COMMAND cmake_diagnostic)
    # CMake >= 4.4: unified diagnostics. CMD_AUTHOR/CMD_POLICY/CMD_DEPRECATED
    # are independent categories (author does not cover policy/deprecated).
    block(SCOPE_FOR DIAGNOSTICS)
      cmake_diagnostic(SET CMD_AUTHOR IGNORE)
      cmake_diagnostic(SET CMD_POLICY IGNORE)
      cmake_diagnostic(SET CMD_DEPRECATED IGNORE)
      add_subdirectory(third_party/torch-mlir)
    endblock()
  else()
    # CMake < 4.4: legacy suppression variables (-Wno-dev/-Wno-deprecated).
    # The VARIABLES scope auto-restores the backups; the cache writes persist
    # beyond the block, so restore them manually.
    block(SCOPE_FOR VARIABLES)
      set(NEXUS_DEV_WARNINGS_BAK "$CACHE{CMAKE_SUPPRESS_DEVELOPER_WARNINGS}")
      set(NEXUS_DEPRECATED_WARNINGS_BAK
          "$CACHE{CMAKE_SUPPRESS_DEPRECATED_WARNINGS}")
      set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)
      set(CMAKE_SUPPRESS_DEPRECATED_WARNINGS ON CACHE BOOL "" FORCE)
      add_subdirectory(third_party/torch-mlir)
      if(NEXUS_DEV_WARNINGS_BAK)
        set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS "${NEXUS_DEV_WARNINGS_BAK}"
            CACHE BOOL "" FORCE)
      else()
        unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS CACHE)
      endif()
      if(NEXUS_DEPRECATED_WARNINGS_BAK)
        set(CMAKE_SUPPRESS_DEPRECATED_WARNINGS "${NEXUS_DEPRECATED_WARNINGS_BAK}"
            CACHE BOOL "" FORCE)
      else()
        unset(CMAKE_SUPPRESS_DEPRECATED_WARNINGS CACHE)
      endif()
    endblock()
  endif()

  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "nexus" CACHE STRING "" FORCE)
endfunction()
