#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "nki/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::nki {

#define GEN_PASS_DEF_NKITOPYTHONPASS
#include "nki/Transforms/Passes.h.inc"

struct NKIToPythonPass : public impl::NKIToPythonPassBase<NKIToPythonPass> {
  void runOnOperation() override {
    llvm::outs() << "import neuronxcc.nki as nki\n";
    llvm::outs() << "import neuronxcc.nki.language as nl\n";
    llvm::outs() << "import numpy as np\n\n";

    getOperation()->walk([](func::FuncOp func) {
      llvm::outs() << "@nki.jit\n";
      llvm::outs() << "def " << func.getName() << "(";
      auto args = func.getArguments();
      for (unsigned i = 0; i < args.size(); ++i) {
        if (i > 0) llvm::outs() << ", ";
        llvm::outs() << "arg" << i;
      }
      llvm::outs() << "):\n";
      llvm::outs() << "    pass\n\n";
    });
  }
};

} // namespace mlir::nki
