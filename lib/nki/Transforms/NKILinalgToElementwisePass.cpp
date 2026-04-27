#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nki/IR/NKIOps.h"
#include "nki/Transforms/Passes.h"
#include <cassert>

namespace mlir::nki
{

#define GEN_PASS_DEF_NKILINALGTOELEMENTWISEPASS
#include "nki/Transforms/Passes.h.inc"

  namespace
  {

    // Maps a linalg.generic body's single arith op to an nki.elementwise kind.
    // kind: 0=mul, 1=add, 2=sub
    static std::optional<int32_t> getElementwiseKind(Operation *op)
    {
      if (isa<arith::MulIOp, arith::MulFOp>(op))
        return 0;
      if (isa<arith::AddIOp, arith::AddFOp>(op))
        return 1;
      if (isa<arith::SubIOp, arith::SubFOp>(op))
        return 2;
      return std::nullopt;
    }

    struct LinalgGenericToElementwise : public OpRewritePattern<linalg::GenericOp>
    {
      using OpRewritePattern::OpRewritePattern;

      LogicalResult matchAndRewrite(linalg::GenericOp op,
                                    PatternRewriter &rewriter) const override
      {
        // Must have 1 or 2 inputs and exactly 1 output (memref-based, no results).
        if (op.getNumDpsInputs() < 1 || op.getNumDpsInputs() > 2 || op.getNumDpsInits() != 1)
          return rewriter.notifyMatchFailure(op, "expected 1 or 2 inputs and 1 output");
        if (op.getNumResults() != 0)
          return rewriter.notifyMatchFailure(op, "expected memref-based generic (no results)");

        assert(op.getNumDpsInputs() >= 1 && op.getNumDpsInputs() <= 2 &&
               "Generic op must have 1 or 2 inputs");
        assert(op.getNumDpsInits() == 1 && "Generic op must have exactly 1 init/output");
        assert(op.getNumResults() == 0 && "Generic op must be memref-based (no tensor results)");

        // Memref-based: all inputs and inits must be memrefs.
        for (Value v : op.getDpsInputs())
          assert(v.getType().isa<MemRefType>() && "All input operands must be memrefs");
        for (Value v : op.getDpsInits())
          assert(v.getType().isa<MemRefType>() && "Output init must be a memref");

        // All iterators must be parallel.
        for (auto iter : op.getIteratorTypesArray())
        {
          if (iter != utils::IteratorType::parallel)
            return rewriter.notifyMatchFailure(op, "non-parallel iterator");
          assert(iter == utils::IteratorType::parallel &&
                 "All iterator types must be parallel");
        }

        // All indexing maps must be identity.
        for (AffineMap map : op.getIndexingMapsArray())
        {
          if (!map.isIdentity())
            return rewriter.notifyMatchFailure(op, "non-identity indexing map");
          assert(map.isIdentity() && "All indexing maps must be identity");
        }

        // Body must contain exactly one binary arith op + linalg.yield.
        Block &body = op.getRegion().front();
        Operation *arithOp = nullptr;
        int nonYieldOps = 0;
        for (Operation &bodyOp : body)
        {
          if (isa<linalg::YieldOp>(&bodyOp))
            continue;
          nonYieldOps++;
          if (arithOp)
            return rewriter.notifyMatchFailure(op, "multiple ops in body");
          arithOp = &bodyOp;
        }
        if (!arithOp)
          return rewriter.notifyMatchFailure(op, "no arith op in body");
        assert(nonYieldOps == 1 && "Body must contain exactly one non-yield op");

        auto kind = getElementwiseKind(arithOp);
        if (!kind)
          return rewriter.notifyMatchFailure(op, "unrecognized arith op");
        assert(kind && "Arithmetic op kind must be defined for supported ops");

        // TODO: Fix this in case the first operand is a non-input
        // and the second operand is an input.
        auto inputs = op.getDpsInputs();
        if (inputs.size() > 2)
          return rewriter.notifyMatchFailure(op, "expected 1 or 2 inputs");

        assert(inputs.size() >= 1 && inputs.size() <= 2 &&
               "Generic op must have arity 1 or 2 inputs");

        Value lhs = inputs[0];
        Value out = op.getDpsInits()[0];
        assert(out && "Output init must exist");

        // rhs is either the second input memref, or a scalar operand of the arith
        // op that comes from outside the body (e.g. a constant).
        Value rhs;
        if (inputs.size() == 2)
        {
          rhs = inputs[1];
          assert(rhs && "RHS memref input must exist when 2 inputs are present");
        }
        else
        {
          Block &body = op.getRegion().front();
          // Find the arith operand that is not a block argument (i.e. external).
          for (Value operand : arithOp->getOperands())
          {
            // If operand comes from outside the body
            if (!llvm::is_contained(body.getArguments(), operand))
            {
              rhs = operand;
              break;
            }
          }
          if (!rhs)
            return rewriter.notifyMatchFailure(op, "could not find scalar rhs operand");
          assert(rhs && "Scalar RHS operand must be defined outside the body");
        }

        rewriter.replaceOpWithNewOp<nki::ElementwiseOp>(
            op, lhs, rhs, out,
            rewriter.getI32IntegerAttr(*kind));

        return success();
      }
    };

  } // namespace

  struct NKILinalgToElementwisePass
      : public impl::NKILinalgToElementwisePassBase<NKILinalgToElementwisePass>
  {
    void runOnOperation() override
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<LinalgGenericToElementwise>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
    }
  };

} // namespace mlir::nki
