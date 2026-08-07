add_mlir_public_c_api_library(MLIRCAPIRegisterNexusEverything
  RegisterEverything.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRRegisterNexusDialects
)
