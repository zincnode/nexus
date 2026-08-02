add_mlir_dialect_library(MLIRNNDialect
  NNOps.cpp

  DEPENDS
  MLIRNNOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  )
