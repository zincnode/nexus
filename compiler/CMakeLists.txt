# -Werror applies to nexus targets only; vendored third_party sources (e.g.
# torch-mlir) are not forced under -Werror.
add_compile_options(-Werror)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

if(NEXUS_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
