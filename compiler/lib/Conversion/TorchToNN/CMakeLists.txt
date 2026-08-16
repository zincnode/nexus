add_mlir_conversion_library(NexusTorchToNN
  TorchToNN.cpp
  Utils.cpp
  Convolution.cpp
  Elementwise.cpp
  LinearAlgebra.cpp
  Normalization.cpp
  Pooling.cpp
  Reduction.cpp
  Resampling.cpp
  ShapeManipulation.cpp

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
