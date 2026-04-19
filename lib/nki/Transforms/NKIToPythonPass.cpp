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

  std::string mlirTypeToNKI(Type type) {
    if (type.isInteger(32)) return "int32";
    if (type.isInteger(16)) return "int16";
    if (type.isInteger(8))  return "int8";
    if (type.isF32())       return "float32";
    if (type.isF16())       return "float16";
    if (type.isBF16())      return "bfloat16";
    return "unknown";
  }

  llvm::raw_ostream &indent() {
    return llvm::outs() << std::string(indentLevel * 4, ' ');
  }

  void emitFunc(func::FuncOp op, const WalkStage &stage) {
    // TODO: Fix this emitter so that it is more generic.
    // This entails writing out all possible cases.
    if (stage.isBeforeAllRegions()) {
      // Classify args: output args are memrefs only used as nki.store dst.
      auto args = op.getArguments();
      SmallVector<Value> outputArgs;
      for (Value arg : args) {
        if (!isa<MemRefType>(arg.getType())) continue;
        // Arg is output if every use is as a nki.store destination.
        bool isOutput = !arg.use_empty() && llvm::all_of(arg.getUsers(), [&](Operation *user) {
          auto store = dyn_cast<nki::StoreOp>(user);
          return store && store.getDst() == arg;
        });
        if (isOutput)
          outputArgs.push_back(arg);
      }

      indent() << "@nki.jit\n";
      indent() << "def " << op.getName() << "(";
      bool first = true;
      for (Value arg : args) {
        if (llvm::is_contained(outputArgs, arg)) continue;
        std::string name = "arg" + std::to_string(cast<BlockArgument>(arg).getArgNumber());
        valueNames[arg] = name;
        if (!first) llvm::outs() << ", ";
        llvm::outs() << name;
        first = false;
      }
      llvm::outs() << "):\n";
      indentLevel++;

      // Emit output allocations inside the function body.
      for (Value arg : outputArgs) {
        valueNames[arg] = "out";
        auto memrefType = cast<MemRefType>(arg.getType());
        // Use arg0.shape / arg0.dtype to stay generic.
        indent() << "out = nl.ndarray(arg0.shape, dtype=arg0.dtype, buffer=nl.shared_hbm)\n";
        (void)memrefType;
      }
    } else if (stage.isAfterAllRegions()) {
      indent() << "return out\n";
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
    auto memrefType = cast<MemRefType>(op.getResult().getType());
    // SBUF temps (address space 2) are anonymous — their name is assigned
    // later by emitElementwise when the result is computed.
    bool isHBM = !memrefType.getMemorySpace() ||
                 (isa<IntegerAttr>(memrefType.getMemorySpace()) &&
                  cast<IntegerAttr>(memrefType.getMemorySpace()).getInt() == 0);
    if (!isHBM) return;

    Value result = op.getResult();
    std::string name = "alloc_" + std::to_string(valueNames.size());
    valueNames[result] = name;
    std::string shape = "(";
    for (unsigned i = 0; i < memrefType.getRank(); ++i) {
      if (i > 0) shape += ", ";
      shape += std::to_string(memrefType.getDimSize(i));
    }
    shape += ")";
    std::string dtype = "nl." + mlirTypeToNKI(memrefType.getElementType());
    indent() << name << " = nl.ndarray(" << shape << ", dtype=" << dtype << ")\n";
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

  void emitElementwise(nki::ElementwiseOp op, const WalkStage &stage) {
    if (!stage.isBeforeAllRegions()) return;
    std::string name = "tmp_" + std::to_string(valueNames.size());
    auto lhs = valueNames.lookup(op.getLhs());
    auto rhs = valueNames.lookup(op.getRhs());
    auto out = valueNames.lookup(op.getOutput());
    std::string nlFunc;
    switch (op.getKind()) {
      case 0: nlFunc = "nl.multiply"; break;
      case 1: nlFunc = "nl.add";      break;
      case 2: nlFunc = "nl.subtract"; break;
      default: nlFunc = "nl.unknown"; break;
    }
    indent() << name << " = " << nlFunc << "(" << lhs << ", " << rhs << ")\n";
    // Only store to HBM (address space 0); SBUF temps (address space 2) just
    // chain to the next op via the symbol table.
    auto outType = cast<MemRefType>(op.getOutput().getType());
    bool isHBM = !outType.getMemorySpace() ||
                 (isa<IntegerAttr>(outType.getMemorySpace()) &&
                  cast<IntegerAttr>(outType.getMemorySpace()).getInt() == 0);
    if (isHBM)
      indent() << "nl.store(" << out << ", " << name << ")\n";
    valueNames[op.getOutput()] = name;
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
      else if (auto ew = dyn_cast<nki::ElementwiseOp>(op))
        emitElementwise(ew, stage);
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
