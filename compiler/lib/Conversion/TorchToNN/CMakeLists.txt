add_mlir_conversion_library(NexusTorchToNN
  TorchToNN.cpp

  DEPENDS
  NexusConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRNNDialect
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(NexusTorchToNN)
