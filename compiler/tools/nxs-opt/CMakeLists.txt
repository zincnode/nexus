add_llvm_executable(nxs-opt nxs-opt.cpp)

target_link_libraries(nxs-opt PRIVATE
  MLIROptLib
  MLIRTransforms
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRRegisterNexusDialects
  TorchMLIRInitAll
)
