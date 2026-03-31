#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "nki/IR/NKIOps.h" 
#include "nki/Transforms/Passes.h"

namespace nki {

class NKIToPythonPass : public mlir::PassWrapper<NKIToPythonPass, mlir::OperationPass<mlir::ModuleOp>> {
  // void runOnOperation() override {
  //   mlir::ModuleOp module = getOperation();

  //   module.walk([&](mlir::nki::LoadOp op) {
  //     // Convert nki.load to Python nl.load()
  //   });
  // }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createNKIToPythonPass() {
  return std::make_unique<NKIToPythonPass>();
}

} // namespace nki