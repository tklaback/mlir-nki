#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "nki/IR/NKIOps.h"
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

  void emitFunc(func::FuncOp op, const WalkStage &stage) {
    if (stage.isBeforeAllRegions()) {
      indent() << "@nki.jit\n";
      indent() << "def " << op.getName() << "(";
      auto args = op.getArguments();
      for (unsigned i = 0; i < args.size(); ++i) {
        if (i > 0) llvm::outs() << ", ";
        llvm::outs() << "arg" << i;
      }
      llvm::outs() << "):\n";
      indentLevel++;
      indent() << "pass";
    } else if (stage.isAfterAllRegions()) {
      indentLevel--;
      llvm::outs() << "\n";
    }
  }

  void emitConstant(arith::ConstantOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    // TODO: emit constant
  }

  void emitFor(scf::ForOp op, const WalkStage &stage) {
    if (stage.isBeforeAllRegions()) {
      // TODO: emit for loop header
      indentLevel++;
    } else if (stage.isAfterAllRegions()) {
      indentLevel--;
    }
  }

  void emitLoad(nki::LoadOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    // TODO: emit nl.load
  }

  void emitStore(nki::StoreOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    // TODO: emit nl.store
  }

  void runOnOperation() override {
    llvm::outs() << "import neuronxcc.nki as nki\n";
    llvm::outs() << "import neuronxcc.nki.language as nl\n";
    llvm::outs() << "import numpy as np\n\n";

    getOperation()->walk([this](Operation *op, const WalkStage &stage) {
      if (auto func = dyn_cast<func::FuncOp>(op))
        emitFunc(func, stage);
      else if (auto konst = dyn_cast<arith::ConstantOp>(op))
        emitConstant(konst, stage);
      else if (auto scfFor = dyn_cast<scf::ForOp>(op))
        emitFor(scfFor, stage);
      else if (auto load = dyn_cast<nki::LoadOp>(op))
        emitLoad(load, stage);
      else if (auto store = dyn_cast<nki::StoreOp>(op))
        emitStore(store, stage);
    });
  }
};

} // namespace mlir::nki
