#ifndef NKI_TRANSFORMS_PASSES_H
#define NKI_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::nki {

#define GEN_PASS_DECL
#include "nki/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "nki/Transforms/Passes.h.inc"

} // namespace mlir::nki

#endif // NKI_TRANSFORMS_PASSES_H
