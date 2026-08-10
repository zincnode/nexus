# Configures the nexus Python bindings (vendored MLIR core API + nn dialect
# bindings), keeping the top-level CMakeLists clean. Must be called after
# find_package(MLIR) and before add_subdirectory(compiler).
#
# Note: this must be a macro, not a function. A function would create a new
# variable scope, whereas this macro needs to write the ordinary variable
# MLIR_SOURCE_DIR into the caller's directory scope for compiler/python to
# inherit (NEXUS_MLIR_PYTHON_ROOT / NEXUS_MLIR_PYTHON_LIB_ROOT depend on it); a
# macro creates no scope and is inlined directly, consistent with the
# convention in cmake/NexusPolicies.cmake.

macro(nexus_configure_python_bindings)
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
endmacro()

# Thin wrapper: compresses the boilerplate of vendoring upstream MLIR dialect
# Python bindings into a single call. Mirrors upstream
# `mlir/python/CMakeLists.txt`; adding/removing a dialect is a one-line change.
# Requires the caller to have set NEXUS_MLIR_PYTHON_ROOT /
# NEXUS_MLIR_PYTHON_LIB_ROOT.

# Wraps declare_mlir_dialect_python_bindings() for vendored upstream dialects.
# Usage:
#   nexus_vendored_dialect_binding(<DIALECT_NAME>
#     TD_FILE <f>                              # required
#     [GEN_ENUM_BINDINGS]
#     [GEN_ENUM_BINDINGS_TD_FILE <f>]
#     [SOURCES <...>] [SOURCES_GLOB <...>])
function(nexus_vendored_dialect_binding name)
  cmake_parse_arguments(ARG "GEN_ENUM_BINDINGS" ""
    "TD_FILE;GEN_ENUM_BINDINGS_TD_FILE;SOURCES;SOURCES_GLOB" ${ARGN})
  if(NOT ARG_TD_FILE)
    message(FATAL_ERROR "nexus_vendored_dialect_binding(${name}): TD_FILE is required")
  endif()

  # When neither SOURCES nor SOURCES_GLOB is given, default to the module whose
  # name matches the DIALECT_NAME (e.g. `arith` -> dialects/arith.py). Dialects
  # whose module name differs (openacc, openmp, async_dialect, transform, pdl,
  # quant) pass SOURCES/SOURCES_GLOB explicitly and are not affected.
  if(NOT ARG_SOURCES AND NOT ARG_SOURCES_GLOB)
    set(ARG_SOURCES "dialects/${name}.py")
  endif()

  set(_flags)
  if(ARG_GEN_ENUM_BINDINGS)
    list(APPEND _flags GEN_ENUM_BINDINGS)
  endif()
  set(_enum_args)
  if(ARG_GEN_ENUM_BINDINGS_TD_FILE)
    list(APPEND _enum_args GEN_ENUM_BINDINGS_TD_FILE ${ARG_GEN_ENUM_BINDINGS_TD_FILE})
  endif()

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT NexusPythonSources.Dialects
    ROOT_DIR "${NEXUS_MLIR_PYTHON_ROOT}"
    DIALECT_NAME "${name}"
    TD_FILE "${ARG_TD_FILE}"
    SOURCES ${ARG_SOURCES}
    SOURCES_GLOB ${ARG_SOURCES_GLOB}
    ${_flags}
    ${_enum_args}
  )
endfunction()

# Wraps declare_mlir_dialect_extension_python_bindings() for the transform
# dialect extensions (DIALECT_NAME is fixed to `transform`). Usage:
#   nexus_vendored_transform_extension(<EXTENSION_NAME> <TD_FILE>
#     [GEN_ENUM_BINDINGS_TD_FILE <f>] [SOURCES <...>])
function(nexus_vendored_transform_extension extension td_file)
  cmake_parse_arguments(ARG "" "" "GEN_ENUM_BINDINGS_TD_FILE;SOURCES" ${ARGN})

  set(_enum_args)
  if(ARG_GEN_ENUM_BINDINGS_TD_FILE)
    list(APPEND _enum_args GEN_ENUM_BINDINGS_TD_FILE ${ARG_GEN_ENUM_BINDINGS_TD_FILE})
  endif()

  declare_mlir_dialect_extension_python_bindings(
    ADD_TO_PARENT NexusPythonSources.Dialects
    ROOT_DIR "${NEXUS_MLIR_PYTHON_ROOT}"
    DIALECT_NAME transform
    EXTENSION_NAME "${extension}"
    TD_FILE "${td_file}"
    SOURCES ${ARG_SOURCES}
    ${_enum_args}
  )
endfunction()

# Wraps declare_mlir_python_extension() for dialect C extensions. Positional
# SOURCES come first; LINK/EMBED are optional keyword args. Usage:
#   nexus_vendored_dialect_cextension(<MODULE_NAME> <PARENT> <sources...>
#     [LINK <libs...>] [EMBED <libs...>])
function(nexus_vendored_dialect_cextension module parent)
  cmake_parse_arguments(ARG "" "" "LINK;EMBED" ${ARGN})

  declare_mlir_python_extension(NexusPythonExtension.${module}
    MODULE_NAME ${module}
    ADD_TO_PARENT NexusPythonSources.Dialects.${parent}
    ROOT_DIR "${NEXUS_MLIR_PYTHON_LIB_ROOT}"
    SOURCES ${ARG_UNPARSED_ARGUMENTS}
    PRIVATE_LINK_LIBS LLVMSupport ${ARG_LINK}
    EMBED_CAPI_LINK_LIBS ${ARG_EMBED}
  )
endfunction()
