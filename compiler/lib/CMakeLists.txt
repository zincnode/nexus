add_subdirectory(Dialect)

add_mlir_library(MLIRRegisterNexusDialects
  RegisterNexusDialects.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRNNDialect
  )
