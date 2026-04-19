#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nki/IR/NKIOps.h"
#include "nki/Transforms/Passes.h"

namespace mlir::nki {

#define GEN_PASS_DEF_NKILINALGTOELEMENTWISEPASS
#include "nki/Transforms/Passes.h.inc"

namespace {

// Maps a linalg.generic body's single arith op to an nki.elementwise kind.
// kind: 0=mul, 1=add, 2=sub
static std::optional<int32_t> getElementwiseKind(Operation *op) {
  if (isa<arith::MulIOp, arith::MulFOp>(op)) return 0;
  if (isa<arith::AddIOp, arith::AddFOp>(op)) return 1;
  if (isa<arith::SubIOp, arith::SubFOp>(op)) return 2;
  return std::nullopt;
}

struct LinalgGenericToElementwise : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Must have exactly 2 inputs and 1 output (memref-based, no results).
    if (op.getNumDpsInputs() < 1 || op.getNumDpsInits() != 1)
      return rewriter.notifyMatchFailure(op, "expected at least 1 input and 1 output");
    if (op.getNumResults() != 0)
      return rewriter.notifyMatchFailure(op, "expected memref-based generic (no results)");

    // All iterators must be parallel.
    for (auto iter : op.getIteratorTypesArray())
      if (iter != utils::IteratorType::parallel)
        return rewriter.notifyMatchFailure(op, "non-parallel iterator");

    // All indexing maps must be identity.
    for (AffineMap map : op.getIndexingMapsArray())
      if (!map.isIdentity())
        return rewriter.notifyMatchFailure(op, "non-identity indexing map");

    // Body must contain exactly one binary arith op + linalg.yield.
    Block &body = op.getRegion().front();
    Operation *arithOp = nullptr;
    for (Operation &bodyOp : body) {
      if (isa<linalg::YieldOp>(&bodyOp)) continue;
      if (arithOp)
        return rewriter.notifyMatchFailure(op, "multiple ops in body");
      arithOp = &bodyOp;
    }
    if (!arithOp)
      return rewriter.notifyMatchFailure(op, "no arith op in body");

    auto kind = getElementwiseKind(arithOp);
    if (!kind)
      return rewriter.notifyMatchFailure(op, "unrecognized arith op");

    // TODO: Fix this in case the first operand is a non-input
    // and the second operand is an input.
    auto inputs = op.getDpsInputs();
    if (inputs.size() > 2)
      return rewriter.notifyMatchFailure(op, "expected 1 or 2 inputs");

    Value lhs = inputs[0];
    Value out = op.getDpsInits()[0];

    // rhs is either the second input memref, or a scalar operand of the arith
    // op that comes from outside the body (e.g. a constant).
    Value rhs;
    if (inputs.size() == 2) {
      rhs = inputs[1];
    } else {
      Block &body = op.getRegion().front();
      // Find the arith operand that is not a block argument (i.e. external).
      for (Value operand : arithOp->getOperands()) {
        // If operand comes from outside the body
        if (!llvm::is_contained(body.getArguments(), operand)) {
          rhs = operand;
          break;
        }
      }
      if (!rhs)
        return rewriter.notifyMatchFailure(op, "could not find scalar rhs operand");
    }

    rewriter.replaceOpWithNewOp<nki::ElementwiseOp>(
        op, lhs, rhs, out,
        rewriter.getI32IntegerAttr(*kind));

    return success();
  }
};

} // namespace

struct NKILinalgToElementwisePass
    : public impl::NKILinalgToElementwisePassBase<NKILinalgToElementwisePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgGenericToElementwise>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::nki
