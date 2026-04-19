#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nki/Transforms/Passes.h"

namespace mlir::nki {

#define GEN_PASS_DEF_NKILIFTLOOPSTOLINALGPASS
#include "nki/Transforms/Passes.h.inc"

namespace {

struct LiftLoopsToLinalg : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp outerFor,
                                PatternRewriter &rewriter) const override {
    // Phase 1: structural check
    // Outer body must contain exactly one inner scf.for and a scf.yield.
    Block *outerBlock = outerFor.getBody();

    scf::ForOp innerFor;
    for (Operation &op : *outerBlock) {
      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        if (innerFor)
          return rewriter.notifyMatchFailure(outerFor, "multiple inner for loops");
        innerFor = forOp;
      } else if (!isa<scf::YieldOp>(&op)) {
        return rewriter.notifyMatchFailure(outerFor, "unexpected op in outer loop body");
      }
    }
    if (!innerFor)
      return rewriter.notifyMatchFailure(outerFor, "no inner for loop found");

    // Inner body must contain: memref.load(s), scalar arith ops, exactly one
    // memref.store, and a scf.yield — in that order, with no other op types.
    Block *innerBlock = innerFor.getBody();
    memref::StoreOp storeOp;
    for (Operation &op : *innerBlock) {
      if (isa<memref::LoadOp>(&op)) continue;
      if (isa<arith::MulIOp, arith::MulFOp,
              arith::AddIOp, arith::AddFOp,
              arith::SubIOp, arith::SubFOp>(&op)) continue;
      if (auto s = dyn_cast<memref::StoreOp>(&op)) {
        if (storeOp)
          return rewriter.notifyMatchFailure(outerFor, "multiple stores in inner loop body");
        storeOp = s;
        continue;
      }
      if (isa<scf::YieldOp>(&op)) continue;
      return rewriter.notifyMatchFailure(outerFor, "unexpected op in inner loop body");
    }
    if (!storeOp)
      return rewriter.notifyMatchFailure(outerFor, "no store in inner loop body");

    // Phase 2: bounds check
    // Both loops must have lb=0, step=1, and ub equal to the corresponding
    // memref dimension so they cover the full tile.
    auto getConstVal = [](Value v) -> std::optional<int64_t> {
      if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
        return c.value();
      if (auto c = v.getDefiningOp<arith::ConstantOp>())
        if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
          return ia.getInt();
      return std::nullopt;
    };

    auto checkBounds = [&](scf::ForOp loop, int64_t expectedUb,
                           StringRef name) -> LogicalResult {
      auto lb = getConstVal(loop.getLowerBound());
      auto step = getConstVal(loop.getStep());
      auto ub = getConstVal(loop.getUpperBound());
      if (!lb || *lb != 0)
        return rewriter.notifyMatchFailure(outerFor, name + ": lb must be 0");
      if (!step || *step != 1)
        return rewriter.notifyMatchFailure(outerFor, name + ": step must be 1");
      if (!ub || *ub != expectedUb)
        return rewriter.notifyMatchFailure(outerFor, name + ": ub must match memref dim");
      return success();
    };

    auto memrefType = cast<MemRefType>(storeOp.getMemRef().getType());
    if (memrefType.getRank() != 2)
      return rewriter.notifyMatchFailure(outerFor, "expected 2-D memref");

    if (failed(checkBounds(outerFor, memrefType.getDimSize(0), "outer")))
      return failure();
    if (failed(checkBounds(innerFor, memrefType.getDimSize(1), "inner")))
      return failure();

    // Phase 3: body analysis + rewrite

    // Step 1: Collect memref.load ops → their source memrefs (inputs).
    SmallVector<memref::LoadOp> loadOps;
    SmallVector<Value> inputMemrefs;
    for (Operation &op : *innerBlock) {
      if (auto load = dyn_cast<memref::LoadOp>(&op)) {
        loadOps.push_back(load);
        inputMemrefs.push_back(load.getMemRef());
      }
    }
    if (loadOps.empty())
      return rewriter.notifyMatchFailure(outerFor, "no loads in inner body");

    // Step 2: Output memref comes from the store (already found in phase 1).
    Value outputMemref = storeOp.getMemRef();

    // Step 3: Verify the stored value comes from an arith op in this block.
    // Build a map from every Value defined in the inner block to its linalg
    // replacement: loads → block args, arith results → will be cloned.
    Value storedVal = storeOp.getValue();
    Operation *arithOp = storedVal.getDefiningOp();
    if (!arithOp || arithOp->getBlock() != innerBlock)
      return rewriter.notifyMatchFailure(outerFor,
                                         "store value must come from an arith op in the inner block");

    // Step 4: Build the linalg.generic op.
    unsigned rank = 2;
    unsigned numInputs = inputMemrefs.size();

    // Identity map for each input and output: (d0, d1) -> (d0, d1)
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps(numInputs + 1, identityMap);

    // All iterators are parallel.
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    rewriter.setInsertionPoint(outerFor);
    auto genericOp = linalg::GenericOp::create(
        rewriter,
        outerFor.getLoc(),
        /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/ValueRange(inputMemrefs),
        /*outputs=*/ValueRange{outputMemref},
        indexingMaps,
        iteratorTypes);

    // Step 5: Populate the body block.
    // Create the block with one scalar arg per input + one per output.
    SmallVector<Type> blockArgTypes;
    for (Value v : inputMemrefs)
      blockArgTypes.push_back(cast<MemRefType>(v.getType()).getElementType());
    blockArgTypes.push_back(cast<MemRefType>(outputMemref.getType()).getElementType());

    SmallVector<Location> blockArgLocs(blockArgTypes.size(), outerFor.getLoc());
    Block *body = rewriter.createBlock(&genericOp.getRegion(), {}, blockArgTypes, blockArgLocs);
    rewriter.setInsertionPointToStart(body);

    // Build a value map: inner block values → linalg body values.
    // Loads map to block args; arith ops will be cloned in order.
    DenseMap<Value, Value> valueMap;
    for (auto [idx, load] : llvm::enumerate(loadOps))
      valueMap[load.getResult()] = body->getArgument(idx);

    // Clone all arith ops in order, remapping operands via valueMap.
    Operation *newArith = nullptr;
    for (Operation &op : *innerBlock) {
      if (isa<memref::LoadOp, memref::StoreOp, scf::YieldOp>(&op)) continue;
      newArith = rewriter.clone(op);
      for (unsigned i = 0; i < newArith->getNumOperands(); ++i) {
        Value orig = newArith->getOperand(i);
        if (auto it = valueMap.find(orig); it != valueMap.end())
          newArith->setOperand(i, it->second);
      }

      // ensures chained operations work
      valueMap[op.getResult(0)] = newArith->getResult(0);
    }

    linalg::YieldOp::create(rewriter, outerFor.getLoc(), newArith->getResult(0));

    // Step 6: Erase the outer scf.for (inner is nested inside, erased with it).
    rewriter.eraseOp(outerFor);

    return success();
  }
};

} // namespace

struct NKILiftLoopsToLinalgPass
    : public impl::NKILiftLoopsToLinalgPassBase<NKILiftLoopsToLinalgPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LiftLoopsToLinalg>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::nki
