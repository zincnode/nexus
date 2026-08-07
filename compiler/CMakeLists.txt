add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

if(NEXUS_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
