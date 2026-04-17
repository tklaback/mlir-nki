#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "nki/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::nki {

#define GEN_PASS_DEF_NKITOPYTHONPASS
#include "nki/Transforms/Passes.h.inc"

struct NKIToPythonPass : public impl::NKIToPythonPassBase<NKIToPythonPass> {
  int indentLevel = 0;

  llvm::raw_ostream &indent() {
    return llvm::outs() << std::string(indentLevel * 4, ' ');
  }

  void runOnOperation() override {
    llvm::outs() << "import neuronxcc.nki as nki\n";
    llvm::outs() << "import neuronxcc.nki.language as nl\n";
    llvm::outs() << "import numpy as np\n\n";

    getOperation()->walk([this](func::FuncOp func, const WalkStage &stage) {
      if (stage.isBeforeAllRegions()) {
        indent() << "@nki.jit\n";
        indent() << "def " << func.getName() << "():\n";
        indentLevel++;
        indent() << "pass\n";
      } else if (stage.isAfterAllRegions()) {
        indentLevel--;
      }
    });
  }
};

} // namespace mlir::nki
