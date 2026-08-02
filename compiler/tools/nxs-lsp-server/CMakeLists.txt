add_llvm_executable(nxs-lsp-server nxs-lsp-server.cpp)

target_link_libraries(nxs-lsp-server PRIVATE
  MLIRLspServerLib
  MLIRDialect
  MLIRRegisterAllDialects
  MLIRRegisterNexusDialects
)
