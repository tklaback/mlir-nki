#ifndef NKI_TRANSFORMS_PASSES_H
#define NKI_TRANSFORMS_PASSES_H

#include <memory>
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace nki {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createNKIToPythonPass();

#define GEN_PASS_REGISTRATION
#include "nki/Transforms/Passes.h.inc"

} // namespace nki

#endif
