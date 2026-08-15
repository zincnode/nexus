//===- Passes.h - Nexus conversion pass registration -----------*- C++ -*-===//

#ifndef CONVERSION_PASSES_H
#define CONVERSION_PASSES_H

#include "Conversion/TorchToNN/TorchToNN.h"

namespace mlir {
namespace nxs {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace nxs
} // namespace mlir

#endif // CONVERSION_PASSES_H
