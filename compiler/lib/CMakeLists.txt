add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)

add_mlir_library(MLIRRegisterNexusDialects
  RegisterNexusDialects.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRNNDialect
  )

add_mlir_library(MLIRRegisterNexusPasses
  RegisterNexusPasses.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  NexusTorchToNN
  )
