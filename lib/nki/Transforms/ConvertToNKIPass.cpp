#include "nki/IR/NKIOps.h"
#include "nki/Transforms/Passes.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::nki {

#define GEN_PASS_DEF_CONVERTAIRTONKIPASS
#include "nki/Transforms/Passes.h.inc"

namespace {

struct ConvertAIRLaunch : public OpRewritePattern<xilinx::air::LaunchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::LaunchOp launch,
                                PatternRewriter &rewriter) const override {
    OperandRange sizes = launch.getSizeOperands();
    if (sizes.size() != 2)
      return rewriter.notifyMatchFailure(launch, "expected 2D launch");

    auto nkiLaunch = nki::LaunchOp::create(rewriter, launch.getLoc(), sizes[0], sizes[1]);

    // Move the body block into nki.launch's region.
    Block *airBlock = &launch.getBody().front();
    Block *nkiBlock = &nkiLaunch.getBody().front();

    // Replace ID block args (induction variables) with the sizes themselves as
    // a placeholder — real program_id lowering comes later.
    ArrayRef<BlockArgument> ids = launch.getIds();
    ArrayRef<BlockArgument> sizeArgs = launch.getSize();
    for (auto [idArg, sizeArg, sizeVal] :
         llvm::zip(ids, sizeArgs, sizes)) {
      Value(idArg).replaceAllUsesWith(sizeVal); // placeholder
      Value(sizeArg).replaceAllUsesWith(sizeVal);
    }

    // Replace kernel block args with the original operands directly —
    // nki.launch is not IsolatedFromAbove so it can capture them.
    for (auto [kernelArg, operand] :
         llvm::zip(launch.getKernelArguments(), launch.getKernelOperands())) {
      Value(kernelArg).replaceAllUsesWith(operand);
    }

    // Erase all block arguments now that uses are replaced.
    airBlock->eraseArguments(
        [](BlockArgument) { return true; });

    // Move ops from the air body into the nki body, dropping the terminator.
    rewriter.mergeBlocks(airBlock, nkiBlock, /*argValues=*/{});
    // air.launch_terminator is the last op; remove it.
    nkiBlock->back().erase();

    rewriter.eraseOp(launch);
    return success();
  }
};

} // namespace

struct ConvertAIRToNKIPass
    : public impl::ConvertAIRToNKIPassBase<ConvertAIRToNKIPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertAIRLaunch>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::nki