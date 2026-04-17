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
  DenseMap<Value, std::string> valueNames;

  llvm::raw_ostream &indent() {
    return llvm::outs() << std::string(indentLevel * 4, ' ');
  }

  void emitFunc(func::FuncOp op, const WalkStage &stage) {
    if (stage.isBeforeAllRegions()) {
      indent() << "@nki.jit\n";
      indent() << "def " << op.getName() << "(";
      auto args = op.getArguments();
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string name = "arg" + std::to_string(i);
        valueNames[args[i]] = name;
        if (i > 0) llvm::outs() << ", ";
        llvm::outs() << name;
      }
      llvm::outs() << "):\n";
      indentLevel++;
    } else if (stage.isAfterAllRegions()) {
      indentLevel--;
      llvm::outs() << "\n";
    }
  }

  void emitConstant(arith::ConstantOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue()))
      valueNames[op.getResult()] = std::to_string(intAttr.getInt());
    else {
      std::string val;
      llvm::raw_string_ostream ss(val);
      op.getValue().print(ss);
      valueNames[op.getResult()] = val;
    }
  }

  void emitAlloc(memref::AllocOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    Value result = op.getResult();
    std::string name = "alloc_" + std::to_string(valueNames.size());
    valueNames[result] = name;
    indent() << name << " = nl.ndarray(...)\n";
  }

  void emitFor(scf::ForOp op, const WalkStage &stage) {
    if (stage.isBeforeAllRegions()) {
      std::string iv = "i" + std::to_string(indentLevel);
      valueNames[op.getInductionVar()] = iv;
      auto lb = valueNames.lookup(op.getLowerBound());
      auto ub = valueNames.lookup(op.getUpperBound());
      auto step = valueNames.lookup(op.getStep());
      indent() << "for " << iv << " in range(" << lb << ", " << ub << ", " << step << "):\n";
      indentLevel++;
    } else if (stage.isAfterAllRegions()) {
      indentLevel--;
    }
  }

  void emitLoad(nki::LoadOp op, const WalkStage &stage) {
    // TODO: emit so that offsets, sizes, and strides are included
    if (!stage.isBeforeAllRegions()) return;
    std::string name = "tmp_" + std::to_string(valueNames.size());
    valueNames[op.getResult()] = name;
    auto src = valueNames.lookup(op.getSrc());
    indent() << name << " = nl.load(" << src << ")\n";
  }

  void emitMemRefLoad(memref::LoadOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    auto buf = valueNames.lookup(op.getMemRef());
    std::string expr = buf + "[";
    auto indices = op.getIndices();
    for (unsigned i = 0; i < indices.size(); ++i) {
      if (i > 0) expr += ", ";
      expr += valueNames.lookup(indices[i]);
    }
    expr += "]";
    valueNames[op.getResult()] = expr;
  }

  void emitArith(Operation *op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    std::string lhs = valueNames.lookup(op->getOperand(0));
    std::string rhs = valueNames.lookup(op->getOperand(1));
    std::string pyOp;
    if (isa<arith::MulIOp, arith::MulFOp>(op)) pyOp = " * ";
    else if (isa<arith::AddIOp, arith::AddFOp>(op)) pyOp = " + ";
    else if (isa<arith::SubIOp, arith::SubFOp>(op)) pyOp = " - ";
    else pyOp = " ? ";
    valueNames[op->getResult(0)] = lhs + pyOp + rhs;
  }

  void emitMemRefStore(memref::StoreOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    auto buf = valueNames.lookup(op.getMemRef());
    auto val = valueNames.lookup(op.getValue());
    std::string idx = "[";
    auto indices = op.getIndices();
    for (unsigned i = 0; i < indices.size(); ++i) {
      if (i > 0) idx += ", ";
      idx += valueNames.lookup(indices[i]);
    }
    idx += "]";
    indent() << buf << idx << " = " << val << "\n";
  }

  void emitStore(nki::StoreOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    auto src = valueNames.lookup(op.getSrc());
    auto dst = valueNames.lookup(op.getDst());
    indent() << "nl.store(" << dst << ", " << src << ")\n";
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
      else if (auto alloc = dyn_cast<memref::AllocOp>(op))
        emitAlloc(alloc, stage);
      else if (auto load = dyn_cast<nki::LoadOp>(op))
        emitLoad(load, stage);
      else if (auto store = dyn_cast<nki::StoreOp>(op))
        emitStore(store, stage);
      else if (auto load = dyn_cast<memref::LoadOp>(op))
        emitMemRefLoad(load, stage);
      else if (auto store = dyn_cast<memref::StoreOp>(op))
        emitMemRefStore(store, stage);
      else if (isa<arith::MulIOp, arith::MulFOp, arith::AddIOp, arith::AddFOp,
                   arith::SubIOp, arith::SubFOp>(op))
        emitArith(op, stage);
    });
  }
};

} // namespace mlir::nki
