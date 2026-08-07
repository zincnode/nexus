# 薄封装：把 vendoring 上游 MLIR dialect Python bindings 的样板压缩成一行调用。
# 镜像 upstream `mlir/python/CMakeLists.txt`；新增/删除 dialect 时只需增删一行。
# 依赖调用方已设置 NEXUS_MLIR_PYTHON_ROOT / NEXUS_MLIR_PYTHON_LIB_ROOT。

# Wraps declare_mlir_dialect_python_bindings() for vendored upstream dialects.
# Usage:
#   nexus_vendored_dialect_binding(<DIALECT_NAME>
#     TD_FILE <f>                              # 必填
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
