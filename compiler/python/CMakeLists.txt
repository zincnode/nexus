# Builds the `nexus` Python package: a vendored MLIR core API plus the nexus
# `nn` dialect bindings. Modeled on the MLIR `AddMLIRPython` infrastructure and
# torch-mlir's `python/CMakeLists.txt`.

# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# Re-root the vendored MLIR Python bindings under the `nexus` package.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=nexus.")

# Root of the nexus Python package sources.
set(NEXUS_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nexus")
# Roots of the vendored MLIR Python sources (source and generated).
set(NEXUS_MLIR_PYTHON_ROOT "${MLIR_SOURCE_DIR}/python/mlir")
set(NEXUS_MLIR_PYTHON_LIB_ROOT "${MLIR_SOURCE_DIR}/lib/Bindings/Python")

# Thin wrappers around the MLIR Python CMake API for the vendored dialect
# declarations below.
include(NexusPython)

################################################################################
# Pure Python sources.
################################################################################

declare_mlir_python_sources(NexusPythonSources)
declare_mlir_python_sources(NexusPythonSources.Core
  ADD_TO_PARENT NexusPythonSources
  ROOT_DIR "${NEXUS_MLIR_PYTHON_ROOT}"
  SOURCES
    _mlir_libs/__init__.py
    _mlir_libs/_mlir/py.typed
    ir.py
    passmanager.py
    rewrite.py
    dialects/_ods_common.py
    dialects/ext.py
)
declare_mlir_python_sources(NexusPythonSources.Extras
  ADD_TO_PARENT NexusPythonSources.Core
  ROOT_DIR "${NEXUS_MLIR_PYTHON_ROOT}"
  SOURCES
    extras/types.py
    extras/meta.py
)
declare_mlir_python_sources(NexusPythonSources.Dialects
  ADD_TO_PARENT NexusPythonSources
  ROOT_DIR "${NEXUS_MLIR_PYTHON_ROOT}"
)

################################################################################
# Vendored upstream MLIR dialect Python bindings, mirroring upstream
# `mlir/python/CMakeLists.txt` for full parity. Static sources come from the
# MLIR source tree; the _*_ops_gen.py files are re-generated into this build
# tree. `DEPENDS` arguments are dropped: they reference MLIR-internal tablegen
# targets not exported to downstream builds, and the generated .td fragments
# already live in the prebuilt MLIR include dirs.
################################################################################

nexus_vendored_dialect_binding(affine TD_FILE dialects/AffineOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(amdgpu TD_FILE dialects/AMDGPUOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(async TD_FILE dialects/AsyncOps.td
  SOURCES_GLOB dialects/async_dialect/*.py)
nexus_vendored_dialect_binding(bufferization TD_FILE dialects/BufferizationOps.td
  GEN_ENUM_BINDINGS_TD_FILE
  "../../include/mlir/Dialect/Bufferization/IR/BufferizationEnums.td")
nexus_vendored_dialect_binding(builtin TD_FILE dialects/BuiltinOps.td)
nexus_vendored_dialect_binding(complex TD_FILE dialects/ComplexOps.td)
nexus_vendored_dialect_binding(index TD_FILE dialects/IndexOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(cf TD_FILE dialects/ControlFlowOps.td)
nexus_vendored_dialect_binding(func TD_FILE dialects/FuncOps.td)
nexus_vendored_dialect_binding(acc TD_FILE dialects/OpenACCOps.td
  SOURCES dialects/openacc.py)
nexus_vendored_dialect_binding(gpu TD_FILE dialects/GPUOps.td
  SOURCES_GLOB dialects/gpu/*.py GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(linalg TD_FILE dialects/LinalgOps.td
  SOURCES_GLOB dialects/linalg/*.py GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(llvm TD_FILE dialects/LLVMOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(transform TD_FILE dialects/TransformOps.td
  SOURCES
    dialects/transform/__init__.py
    _mlir_libs/_mlir/dialects/transform/__init__.pyi
  GEN_ENUM_BINDINGS_TD_FILE
  "../../include/mlir/Dialect/Transform/IR/TransformAttrs.td")

declare_mlir_python_sources(
  NexusPythonSources.Dialects.transform.extras
  ADD_TO_PARENT NexusPythonSources.Dialects
  ROOT_DIR "${NEXUS_MLIR_PYTHON_ROOT}"
  SOURCES
    dialects/transform/extras/__init__.py
)

declare_mlir_python_sources(
  NexusPythonSources.Dialects.transform.interpreter
  ADD_TO_PARENT NexusPythonSources.Dialects
  ROOT_DIR "${NEXUS_MLIR_PYTHON_ROOT}"
  SOURCES
    dialects/transform/interpreter/__init__.py
)

nexus_vendored_transform_extension(transform_pdl_extension
  dialects/TransformPDLExtensionOps.td dialects/transform/pdl.py)
nexus_vendored_transform_extension(transform_smt_extension
  dialects/TransformSMTExtensionOps.td dialects/transform/smt.py)
nexus_vendored_transform_extension(transform_debug_extension
  dialects/TransformDebugExtensionOps.td dialects/transform/debug.py)
nexus_vendored_transform_extension(transform_tune_extension
  dialects/TransformTuneExtensionOps.td dialects/transform/tune.py)
nexus_vendored_transform_extension(bufferization_transform
  dialects/BufferizationTransformOps.td dialects/transform/bufferization.py)
nexus_vendored_transform_extension(gpu_transform
  dialects/GPUTransformOps.td dialects/transform/gpu.py)
nexus_vendored_transform_extension(loop_transform
  dialects/SCFLoopTransformOps.td dialects/transform/loop.py)
nexus_vendored_transform_extension(memref_transform
  dialects/MemRefTransformOps.td dialects/transform/memref.py)
nexus_vendored_transform_extension(nvgpu_transform
  dialects/NVGPUTransformOps.td dialects/transform/nvgpu.py)
nexus_vendored_transform_extension(structured_transform
  dialects/LinalgStructuredTransformOps.td dialects/transform/structured.py
  GEN_ENUM_BINDINGS_TD_FILE
  "../../include/mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td")
nexus_vendored_transform_extension(sparse_tensor_transform
  dialects/SparseTensorTransformOps.td dialects/transform/sparse_tensor.py)
nexus_vendored_transform_extension(tensor_transform
  dialects/TensorTransformOps.td dialects/transform/tensor.py)
nexus_vendored_transform_extension(vector_transform
  dialects/VectorTransformOps.td dialects/transform/vector.py
  GEN_ENUM_BINDINGS_TD_FILE
  "../../include/mlir/Dialect/Vector/Transforms/VectorTransformsBase.td")
nexus_vendored_transform_extension(x86_transform
  dialects/X86TransformOps.td dialects/transform/x86.py)
nexus_vendored_transform_extension(xegpu_transform
  dialects/XeGPUTransformOps.td dialects/transform/xegpu.py)

nexus_vendored_dialect_binding(math TD_FILE dialects/MathOps.td)
nexus_vendored_dialect_binding(arith TD_FILE dialects/ArithOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(memref TD_FILE dialects/MemRefOps.td)
nexus_vendored_dialect_binding(shard TD_FILE dialects/ShardOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(ml_program TD_FILE dialects/MLProgramOps.td)
nexus_vendored_dialect_binding(nvgpu TD_FILE dialects/NVGPUOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(nvvm TD_FILE dialects/NVVMOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(rocdl TD_FILE dialects/ROCDLOps.td)

declare_mlir_python_sources(
  NexusPythonSources.Dialects.quant
  ADD_TO_PARENT NexusPythonSources.Dialects
  ROOT_DIR "${NEXUS_MLIR_PYTHON_ROOT}"
  SOURCES
    dialects/quant.py
    _mlir_libs/_mlir/dialects/quant.pyi
)

nexus_vendored_dialect_binding(emitc TD_FILE dialects/EmitC.td)
nexus_vendored_dialect_binding(pdl TD_FILE dialects/PDLOps.td
  SOURCES
    dialects/pdl.py
    _mlir_libs/_mlir/dialects/pdl.pyi)
nexus_vendored_dialect_binding(omp TD_FILE dialects/OpenMPOps.td
  SOURCES dialects/openmp.py)
nexus_vendored_dialect_binding(scf TD_FILE dialects/SCFOps.td)
nexus_vendored_dialect_binding(shape TD_FILE dialects/ShapeOps.td)
nexus_vendored_dialect_binding(sparse_tensor TD_FILE dialects/SparseTensorOps.td
  GEN_ENUM_BINDINGS_TD_FILE
  "../../include/mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td")
nexus_vendored_dialect_binding(smt TD_FILE dialects/SMTOps.td GEN_ENUM_BINDINGS)
nexus_vendored_dialect_binding(spirv TD_FILE dialects/SPIRVOps.td)
nexus_vendored_dialect_binding(tensor TD_FILE dialects/TensorOps.td)
nexus_vendored_dialect_binding(tosa TD_FILE dialects/TosaOps.td)
nexus_vendored_dialect_binding(ub TD_FILE dialects/UBOps.td)
nexus_vendored_dialect_binding(vector TD_FILE dialects/VectorOps.td
  GEN_ENUM_BINDINGS_TD_FILE "dialects/VectorAttributes.td")
nexus_vendored_dialect_binding(x86 TD_FILE dialects/X86.td)
nexus_vendored_dialect_binding(irdl TD_FILE dialects/IRDLOps.td GEN_ENUM_BINDINGS)

# Nexus `nn` dialect bindings.
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NexusPythonSources.Dialects
  ROOT_DIR "${NEXUS_PYTHON_ROOT_DIR}"
  TD_FILE dialects/NNBinding.td
  SOURCES dialects/nn/__init__.py
  DIALECT_NAME nn
  DEPENDS MLIRNNOpsIncGen
)

# Nexus-specific Python sources.
declare_mlir_python_sources(NexusPythonSources.PublicAPI
  ADD_TO_PARENT NexusPythonSources
  ROOT_DIR "${NEXUS_PYTHON_ROOT_DIR}"
  SOURCES
    __init__.py
    dialects/__init__.py
)

################################################################################
# Python extensions.
################################################################################

# The MLIR core `_mlir` extension, rebuilt out-of-tree with the nexus domain.
declare_mlir_python_extension(NexusPythonExtension.Core
  MODULE_NAME _mlir
  ADD_TO_PARENT NexusPythonSources.Core
  ROOT_DIR "${NEXUS_MLIR_PYTHON_LIB_ROOT}"
  SOURCES
    MainModule.cpp
    IRAffine.cpp
    IRInterfaces.cpp
    Pass.cpp
    Rewrite.cpp

    # Headers must be included explicitly so they are installed.
    Pass.h
    Rewrite.h
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIDebug
    MLIRCAPIInterfaces
    MLIRCAPIFunc
    MLIRCAPITransforms
)

# The MLIR Python support library, rebuilt out-of-tree with the nexus domain.
declare_mlir_python_extension(NexusPythonExtension.Support
  _PRIVATE_SUPPORT_LIB
  MODULE_NAME MLIRPythonSupport
  ADD_TO_PARENT NexusPythonSources.Core
  ROOT_DIR "${NEXUS_MLIR_PYTHON_LIB_ROOT}"
  SOURCES
    Globals.cpp
    IRAttributes.cpp
    IRCore.cpp
    IRTypes.cpp
)

# Dialect C extensions required by the vendored dialect packages.
nexus_vendored_dialect_cextension(_mlirDialectsLinalg linalg DialectLinalg.cpp
  EMBED MLIRCAPIIR MLIRCAPILinalg)
nexus_vendored_dialect_cextension(_mlirDialectsGPU gpu DialectGPU.cpp
  EMBED MLIRCAPIIR MLIRCAPIGPU)
nexus_vendored_dialect_cextension(_mlirDialectsLLVM llvm DialectLLVM.cpp
  EMBED MLIRCAPIIR MLIRCAPILLVM MLIRCAPITarget)
nexus_vendored_dialect_cextension(_mlirDialectsQuant quant DialectQuant.cpp
  EMBED MLIRCAPIIR MLIRCAPIQuant)
nexus_vendored_dialect_cextension(_mlirDialectsNVGPU nvgpu DialectNVGPU.cpp
  EMBED MLIRCAPIIR MLIRCAPINVGPU)
nexus_vendored_dialect_cextension(_mlirDialectsPDL pdl DialectPDL.cpp
  EMBED MLIRCAPIIR MLIRCAPIPDL)
nexus_vendored_dialect_cextension(_mlirDialectsSparseTensor sparse_tensor DialectSparseTensor.cpp
  EMBED MLIRCAPIIR MLIRCAPISparseTensor)
nexus_vendored_dialect_cextension(_mlirDialectsTransform transform DialectTransform.cpp Rewrite.h
  LINK NexusPythonExtension.Core EMBED MLIRCAPIIR MLIRCAPITransformDialect)
nexus_vendored_dialect_cextension(_mlirDialectsIRDL irdl DialectIRDL.cpp
  EMBED MLIRCAPIIR MLIRCAPIIRDL)
nexus_vendored_dialect_cextension(_mlirDialectsSMT smt DialectSMT.cpp
  EMBED MLIRCAPIIR MLIRCAPISMT MLIRCAPIExportSMTLIB)
nexus_vendored_dialect_cextension(_mlirDialectsAMDGPU amdgpu DialectAMDGPU.cpp
  EMBED MLIRCAPIIR MLIRCAPIAMDGPU)
nexus_vendored_dialect_cextension(_mlirSparseTensorPasses sparse_tensor SparseTensorPasses.cpp
  EMBED MLIRCAPISparseTensor)
nexus_vendored_dialect_cextension(_mlirTransformInterpreter transform TransformInterpreter.cpp
  EMBED MLIRCAPITransformDialectTransforms)

# Nexus + upstream dialect, translation and pass registration extension.
declare_mlir_python_extension(NexusPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ADD_TO_PARENT NexusPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    NexusRegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIConversion
    MLIRCAPITransforms
    MLIRCAPIRegisterNexusEverything
    MLIRCAPIRegisterEverything
)

################################################################################
# Common CAPI dependency DSO.
################################################################################

add_mlir_python_common_capi_library(NexusPythonCAPI
  INSTALL_COMPONENT NexusPythonModules
  INSTALL_DESTINATION python_packages/nexus/nexus/_mlir_libs
  OUTPUT_DIRECTORY "${NEXUS_PYTHON_PACKAGES_DIR}/nexus/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES
    NexusPythonSources
)

################################################################################
# Assemble the Python package.
################################################################################

add_mlir_python_modules(NexusPythonModules
  ROOT_PREFIX "${NEXUS_PYTHON_PACKAGES_DIR}/nexus"
  INSTALL_PREFIX "python_packages/nexus/nexus"
  DECLARED_SOURCES
    NexusPythonSources
  COMMON_CAPI_LINK_LIBS
    NexusPythonCAPI
)
