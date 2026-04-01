#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "nki/Transforms/Passes.h"

namespace mlir::nki {

#define GEN_PASS_DEF_NKITOPYTHONPASS
#include "nki/Transforms/Passes.h.inc"

struct NKIToPythonPass : public impl::NKIToPythonPassBase<NKIToPythonPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
  }
};

} // namespace mlir::nki
