#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nki/Transforms/Passes.h"
#include <cassert>

namespace mlir::nki
{

#define GEN_PASS_DEF_NKILIFTLOOPSTOLINALGPASS
#include "nki/Transforms/Passes.h.inc"

  namespace
  {

    struct LiftLoopsToLinalg : public OpRewritePattern<scf::ForOp>
    {
      using OpRewritePattern::OpRewritePattern;

      LogicalResult matchAndRewrite(scf::ForOp outerFor,
                                    PatternRewriter &rewriter) const override
      {
        // Phase 1: Structural checks (well-nested and inner-body liftable)
        // Inference rule: OUTER-SHAPE
        // ops(outer-body) = [for_inner, scf.yield]
        Block *outerBlock = outerFor.getBody();

        scf::ForOp innerFor;
        int innerForCount = 0;
        bool hasYield = false;

        for (Operation &op : *outerBlock)
        {
          if (auto forOp = dyn_cast<scf::ForOp>(&op))
          {
            if (innerFor)
              return rewriter.notifyMatchFailure(outerFor, "multiple inner for loops");
            innerFor = forOp;
            innerForCount++;
          }
          else if (isa<scf::YieldOp>(&op))
          {
            hasYield = true;
          }
          else
          {
            return rewriter.notifyMatchFailure(outerFor, "unexpected op in outer loop body");
          }
        }

        // Assert: Multiple inner loops exist in outer loop (OUTER-SHAPE)
        assert(innerForCount <= 1 && "Multiple inner loops exist in outer loop");
        if (innerForCount != 1)
          return rewriter.notifyMatchFailure(outerFor, "no inner for loop found");

        // Assert: Outer body must have exactly one inner for and yield
        assert(hasYield && "Missing scf.yield in outer loop body");
        assert(innerFor && "Inner for loop not found after count check");

        // Inner body must contain: loads, scalar arith ops, exactly one store, yield
        // Inference rule: INNER-SHAPE
        // ops(inner-body) ⊆ {loads, arith, store, yield}
        // ∃! store
        Block *innerBlock = innerFor.getBody();
        memref::StoreOp storeOp;
        int storeCount = 0;
        SmallVector<memref::LoadOp> loads;
        SmallVector<Operation *> arithOps;

        for (Operation &op : *innerBlock)
        {
          if (auto load = dyn_cast<memref::LoadOp>(&op))
          {
            loads.push_back(load);
            continue;
          }
          // Assert: Illegal op in inner body (must be supported scalar arith)
          // Inference rule: ARITH Supported
          if (isa<arith::MulIOp, arith::MulFOp,
                  arith::AddIOp, arith::AddFOp,
                  arith::SubIOp, arith::SubFOp>(&op))
          {
            arithOps.push_back(&op);
            continue;
          }
          if (auto s = dyn_cast<memref::StoreOp>(&op))
          {
            storeCount++;
            storeOp = s;
            continue;
          }
          if (isa<scf::YieldOp>(&op))
            continue;
          return rewriter.notifyMatchFailure(outerFor, "unexpected op in inner loop body");
        }

        // Assert: Inner body has ≠ 1 store (INNER-SHAPE)
        assert(storeCount <= 1 && "Multiple stores in inner loop body");
        assert(storeOp && "Missing store in inner loop body");

        // Assert: Missing loads or store
        assert(!loads.empty() && "Missing loads in inner body");
        assert(storeOp && "Missing store in inner body");

        // Phase 2: Bounds check (full-tile judgment form)
        // Inference rule: BOUNDS-OK
        // C(lb)=0, C(step)=1, C(ub)=n => Γ ⊢ scf.for(lb, ub, step) : full-tile(n)
        auto getConstVal = [](Value v) -> std::optional<int64_t>
        {
          if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
            return c.value();
          if (auto c = v.getDefiningOp<arith::ConstantOp>())
            if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
              return ia.getInt();
          return std::nullopt;
        };

        auto checkBounds = [&](scf::ForOp loop, int64_t expectedUb,
                               StringRef name) -> LogicalResult
        {
          auto lb = getConstVal(loop.getLowerBound());
          auto step = getConstVal(loop.getStep());
          auto ub = getConstVal(loop.getUpperBound());

          // Assert: Non-constant bounds
          if (!lb)
            return rewriter.notifyMatchFailure(outerFor, name + ": lower bound is not constant");
          if (!step)
            return rewriter.notifyMatchFailure(outerFor, name + ": step is not constant");
          if (!ub)
            return rewriter.notifyMatchFailure(outerFor, name + ": upper bound is not constant");

          assert(lb && step && ub && "Bounds must be constant");

          // Assert: Loop bounds are not (0, 1, N) form
          // Assert: Bounds mismatch (lb, step, ub)
          if (*lb != 0)
            return rewriter.notifyMatchFailure(outerFor, name + ": lb must be 0");
          if (*step != 1)
            return rewriter.notifyMatchFailure(outerFor, name + ": step must be 1");
          if (*ub != expectedUb)
            return rewriter.notifyMatchFailure(outerFor, name + ": ub must match memref dim");

          assert(*lb == 0 && "Lower bound must be 0");
          assert(*step == 1 && "Step must be 1");
          assert(*ub == expectedUb && "Upper bound must match memref dimension");

          return success();
        };

        auto memrefType = cast<MemRefType>(storeOp.getMemRef().getType());

        // Assert: Memref not rank-2
        assert(memrefType.getRank() == 2 && "Store memref must be rank-2");
        if (memrefType.getRank() != 2)
          return rewriter.notifyMatchFailure(outerFor, "expected 2-D memref");

        // Inference rule: FULL-TILE-2D
        // M(store.memref) = memref<N_0 × N_1, τ>
        // Γ ⊢ for_outer : full-tile(N_0)
        // Γ ⊢ for_inner : full-tile(N_1)
        if (failed(checkBounds(outerFor, memrefType.getDimSize(0), "outer")))
          return failure();
        if (failed(checkBounds(innerFor, memrefType.getDimSize(1), "inner")))
          return failure();

        // Phase 3: body analysis + rewrite

        // Step 1: Collect memref.load ops → their source memrefs (inputs).
        SmallVector<memref::LoadOp> loadOps;
        SmallVector<Value> inputMemrefs;
        for (Operation &op : *innerBlock)
        {
          if (auto load = dyn_cast<memref::LoadOp>(&op))
          {
            loadOps.push_back(load);
            inputMemrefs.push_back(load.getMemRef());
          }
        }

        // Assert: Missing loads or store (double-checked)
        assert(!loadOps.empty() && "No loads found in inner body");
        if (loadOps.empty())
          return rewriter.notifyMatchFailure(outerFor, "no loads in inner body");

        // Step 2: Output memref comes from the store (already found in phase 1).
        Value outputMemref = storeOp.getMemRef();

        // Step 3: Verify the stored value comes from an arith op in this block.
        // Build a map from every Value defined in the inner block to its linalg
        // replacement: loads → block args, arith results → will be cloned.
        // Inference rule: STORE-SRC
        // v = result of a_j, a_j ∈ inner-body => Γ ⊢ store.value : defined-locally
        Value storedVal = storeOp.getValue();
        Operation *arithOp = storedVal.getDefiningOp();

        // Assert: Store value is not defined in inner block
        // Assert: Store not defined locally
        if (!arithOp || arithOp->getBlock() != innerBlock)
          return rewriter.notifyMatchFailure(outerFor,
                                             "store value must come from an arith op in the inner block");

        assert(arithOp && "Store value must have a defining operation");
        assert(arithOp->getBlock() == innerBlock && "Store value must be defined in inner block");

        // Verify computation constraint: value is result of pure arith chain
        // Inference rule: valid-reduction-tree
        // v = f(load_1, ..., load_k), f is pure chain of arith ops
        SmallVector<Value> arithChainValues;
        DenseSet<Value> allowedValues;

        // Collect all load results
        for (auto load : loadOps)
          allowedValues.insert(load.getResult());

        // Walk the arith chain back from stored value
        std::function<bool(Operation *)> validateArithChain = [&](Operation *op) -> bool
        {
          if (!op)
            return false;

          if (op->getBlock() != innerBlock)
            return false;

          // Assert: Computation depends on values outside load/arithmetic chain
          if (isa<arith::MulIOp, arith::MulFOp,
                  arith::AddIOp, arith::AddFOp,
                  arith::SubIOp, arith::SubFOp>(op))
          {
            for (operand_range::iterator it = op->operand_begin();
                 it != op->operand_end(); ++it)
            {
              Value operand = *it;
              if (allowedValues.count(operand))
                continue;

              if (auto defOp = operand.getDefiningOp())
              {
                if (!validateArithChain(defOp))
                {
                  llvm::errs() << "Invalid operand in arithmetic chain\n";
                  return false;
                }
                allowedValues.insert(operand);
              }
              else
              {
                llvm::errs() << "Operand has no defining operation in inner block\n";
                return false;
              }
            }
            return true;
          }
          return false;
        };

        if (!validateArithChain(arithOp))
          return rewriter.notifyMatchFailure(outerFor,
                                             "computation depends on values outside load/arithmetic chain");

        assert(validateArithChain(arithOp) && "Arithmetic chain must be self-contained");

        // Step 4: Build the linalg.generic op.
        unsigned rank = 2;
        unsigned numInputs = inputMemrefs.size();

        // Identity map for each input and output: (d0, d1) -> (d0, d1)
        AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
        SmallVector<AffineMap> indexingMaps(numInputs + 1, identityMap);

        // Assert: Load/store structure is not consistent across iterations
        // All iterators must be parallel (no reductions)
        for (const auto &map : indexingMaps)
        {
          assert(map.getNumDims() == rank && "IndexingMap dimension mismatch");
        }

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
        // Inference rule: MAP-LOAD
        // load_i ∈ loads => valueMap(load_i.result) = blockArg_i
        DenseMap<Value, Value> valueMap;
        for (auto [idx, load] : llvm::enumerate(loadOps))
          valueMap[load.getResult()] = body->getArgument(idx);

        // Clone all arith ops in order, remapping operands via valueMap.
        // Inference rule: MAP-ARITH
        // a_j' = clone(a_j) => valueMap(a_j.result) = a_j'.result
        Operation *newArith = nullptr;
        for (Operation &op : *innerBlock)
        {
          if (isa<memref::LoadOp, memref::StoreOp, scf::YieldOp>(&op))
            continue;
          newArith = rewriter.clone(&op);
          assert(newArith && "Cloned operation must not be null");

          for (unsigned i = 0; i < newArith->getNumOperands(); ++i)
          {
            Value orig = newArith->getOperand(i);
            if (auto it = valueMap.find(orig); it != valueMap.end())
              newArith->setOperand(i, it->second);
          }

          // ensures chained operations work
          valueMap[op.getResult(0)] = newArith->getResult(0);
        }

        assert(newArith && "Must have at least one arithmetic operation to yield");
        linalg::YieldOp::create(rewriter, outerFor.getLoc(), newArith->getResult(0));

        // Step 6: Erase the outer scf.for (inner is nested inside, erased with it).
        rewriter.eraseOp(outerFor);

        return success();
      }
    };

  } // namespace

  struct NKILiftLoopsToLinalgPass
      : public impl::NKILiftLoopsToLinalgPassBase<NKILiftLoopsToLinalgPass>
  {
    void runOnOperation() override
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<LiftLoopsToLinalg>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
    }
  };

} // namespace mlir::nki
