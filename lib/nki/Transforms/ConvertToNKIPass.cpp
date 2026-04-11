#include "nki/IR/NKIOps.h"
#include "nki/Transforms/Passes.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>

namespace mlir::nki {

#define GEN_PASS_DEF_CONVERTAIRTONKIPASS
#include "nki/Transforms/Passes.h.inc"

namespace {

struct ConvertAIRLaunch : public OpRewritePattern<xilinx::air::LaunchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::LaunchOp launch,
                                PatternRewriter &rewriter) const override {
    OperandRange launchSizes = launch.getSizeOperands();
    if (launchSizes.size() != 2)
      return rewriter.notifyMatchFailure(launch, "expected 2D launch");

    // Inline each segment: replace its block args, then splice its body in place.
    launch.walk([](xilinx::air::SegmentOp seg) {
      for (auto [arg, operand] : llvm::zip(seg.getKernelArguments(), seg.getKernelOperands()))
        Value(arg).replaceAllUsesWith(operand);
      Block *segBlock = &seg.getBody().front();
      segBlock->eraseArguments([](BlockArgument) { return true; });
      segBlock->back().erase(); // erase segment_terminator
      seg->getBlock()->getOperations().splice(seg->getIterator(), segBlock->getOperations());
      seg.erase();
    });

    // getIds are the inductive variables
    // getSize are the names of the newly assigned bound variables
    // launchSizes these are the constants that hold the actual size info
    Block *launchBlock = &launch.getBody().front();
    for (auto [idArg, sizeArg, sizeVal] :
         llvm::zip(launch.getIds(), launch.getSize(), launchSizes))  {
      Value(idArg).replaceAllUsesWith(sizeVal);
      Value(sizeArg).replaceAllUsesWith(sizeVal);
    }

    // these are used to pass down the data structure from the outer scope into the launch.
    for (auto [kernelArg, operand] :
         llvm::zip(launch.getKernelArguments(), launch.getKernelOperands()))
      Value(kernelArg).replaceAllUsesWith(operand);

    launchBlock->eraseArguments([](BlockArgument) { return true; });

    // Value herdSizeX, herdSizeY;
    // launch.walk([&](xilinx::air::HerdOp herd) {

    // });

    Value one = arith::ConstantIndexOp::create(rewriter, launch.getLoc(), 1);

    auto nkiLaunch = nki::LaunchOp::create(rewriter, launch.getLoc(), one, one);

    Block *airBlock = &launch.getBody().front();
    nkiLaunch.getBody().emplaceBlock();
    Block *nkiBlock = &nkiLaunch.getBody().front();

    airBlock->eraseArguments(
        [](BlockArgument) { return true; });

    rewriter.mergeBlocks(airBlock, nkiBlock, /*argValues=*/{});
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

    // check for failure
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::nki