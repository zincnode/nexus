#ifndef INITNEXUSDIALECTS_H
#define INITNEXUSDIALECTS_H

namespace mlir {
class DialectRegistry;
class MLIRContext;

namespace nxs {

/// Add all the Nexus dialects to the provided registry.
void registerNexusDialects(DialectRegistry &registry);

/// Append all the Nexus dialects to the registry contained in the given
/// context.
void registerNexusDialects(MLIRContext &context);

} // namespace nxs

} // namespace mlir

#endif // INITNEXUSDIALECTS_H
