#include "nki/IR/NKIOps.h"
#include "nki/Transforms/Passes.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "nki/Analysis/ChannelDependencyAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/IRMapping.h"
#include <iostream>

namespace mlir::nki {

#define GEN_PASS_DEF_CONVERTAIRTONKIPASS
#include "nki/Transforms/Passes.h.inc"

namespace {

// Inlines an air.segment by splicing its body ops into the parent block in
// place of the segment op. Block arguments (bound to segment_operands) are
// replaced with the corresponding operand values before inlining.
struct InlineSegment : public OpRewritePattern<xilinx::air::SegmentOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::SegmentOp segment,
                                PatternRewriter &rewriter) const override {
    Block &body = segment.getBody().front();

    // The block args correspond to segment_operands (after async_dependencies
    // and sizes). Map each block arg -> its operand value in the parent scope.
    ValueRange segmentOperands = segment.getSegmentOperands();
    // Block args: [async_deps..., sizes..., segment_operands...]
    // Sizes and async token args precede segment_operands in the block arg list.
    unsigned numAsyncDeps = segment.getAsyncDependencies().size();
    unsigned numSizes = segment.getSizes().size();
    unsigned segArgOffset = numAsyncDeps + numSizes;

    for (auto [blockArg, operand] :
         llvm::zip(body.getArguments().drop_front(segArgOffset),
                   segmentOperands))
      rewriter.replaceAllUsesWith(blockArg, operand);

    // Drop the terminator, then splice remaining ops before the segment op.
    rewriter.eraseOp(body.getTerminator());
    rewriter.inlineBlockBefore(&body, segment);

    // The segment produces no results we need to forward (async token aside —
    // if present, callers should have waited on it before this pattern fires).
    rewriter.eraseOp(segment);
    return success();
  }
};

struct ConvertAIRChannel : public OpRewritePattern<xilinx::air::ChannelPutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::ChannelPutOp put,
                                PatternRewriter &rewriter) const override {
    // Find the matching get op by channel name.
    xilinx::air::ChannelGetOp matchedGet;
    put->getParentOfType<mlir::ModuleOp>().walk([&](xilinx::air::ChannelGetOp get) {
      if (get.getChanName() == put.getChanName())
        matchedGet = get;
    });

    if (!matchedGet) {
      return rewriter.notifyMatchFailure(put, "no matching channel get");
    }

    bool putInHerd = (bool)put->getParentOfType<xilinx::air::HerdOp>();
    bool getInHerd = (bool)matchedGet->getParentOfType<xilinx::air::HerdOp>();

    if (!putInHerd) {
      // The put is at launch level (placing a global buffer into the channel).
      // Convert to a nki.load inserted at the get's location inside the herd.
      //
      // Because HerdOp is IsIsolatedFromAbove, we can't reference the put's
      // src value directly from inside the herd. Instead, thread it in as a
      // new kernel operand via appendKernelOperands, which also adds the
      // corresponding block argument.

      Value src = put.getSrc();
      auto srcType = cast<MemRefType>(src.getType());

      auto herd = matchedGet->getParentOfType<xilinx::air::HerdOp>();
      rewriter.modifyOpInPlace(herd, [&]() {
        herd.appendKernelOperands(ValueRange{src});
      });
      // The new block arg is the last one appended.
      Value srcArg = herd.getBody().getArguments().back();

      rewriter.setInsertionPoint(matchedGet);
      auto loadOp = nki::LoadOp::create(
          rewriter, matchedGet.getLoc(),
          srcType,
          srcArg,
          put.getSrcOffsets(),
          put.getSrcSizes(),
          put.getSrcStrides());

      // Replace all uses of the get's destination buffer with the load result.
      matchedGet.getDst().replaceAllUsesWith(loadOp.getResult());

      rewriter.eraseOp(matchedGet);
      rewriter.eraseOp(put);
      return success();
    } else if (getInHerd) {
      // This is the case that the put is in the herd and the get is in the herd
      // For this case, we want the producer herd (the one with the put) to replace its air.channel.put with a nki.store into sbuf.
      // Then, we want the consumer herd to have a barrier that waits until that operation is finished.
      // This needs 
    } else {
      // Put is inside the herd; get is at launch level (herd -> global).
      // Thread the get's dst into the herd as a new kernel operand, then
      // replace the put with nki.store(src, dstArg, ...) and erase the get.
      Value dst = matchedGet.getDst();

      auto herd = put->getParentOfType<xilinx::air::HerdOp>();
      rewriter.modifyOpInPlace(herd, [&]() {
        herd.appendKernelOperands(ValueRange{dst});
      });
      Value dstArg = herd.getBody().getArguments().back();

      auto getOrZero = [&](ValueRange vals, unsigned idx) -> Value {
        if (idx < vals.size())
          return vals[idx];
        return arith::ConstantIndexOp::create(rewriter, put.getLoc(), 0);
      };

      rewriter.setInsertionPoint(put);
      nki::StoreOp::create(
          rewriter, put.getLoc(),
          put.getSrc(),
          dstArg,
          put.getSrcOffsets(),
          put.getSrcSizes(),
          put.getSrcStrides());

      rewriter.eraseOp(put);
      rewriter.eraseOp(matchedGet);
      return success();
    }
    return failure();
  }
};

struct ConvertAIRLaunch : public OpRewritePattern<xilinx::air::LaunchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
    xilinx::air::LaunchOp launch,
    PatternRewriter &rewriter
  ) const override {
    return success();
  }
};

} // namespace

struct ConvertAIRToNKIPass
    : public impl::ConvertAIRToNKIPassBase<ConvertAIRToNKIPass> {
    void runOnOperation() override {
      // Phase 1: lower channel ops before the launch structure is changed


      auto &analysis = getAnalysis<ChannelDependencyAnalysis>();
  
      constexpr StringRef names[] = {"LINEAR", "DAG", "CYCLIC", "FANOUT", "FANIN"};
      ChannelGraphType graphType = analysis.getGraphType();
      if (graphType == ChannelGraphType::LINEAR) {
        RewritePatternSet patterns(&getContext());
        patterns.add<InlineSegment>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
          signalPassFailure();
      } else if (graphType == ChannelGraphType::DAG) {
      } else {
      }
      llvm::errs() << names[static_cast<unsigned>(graphType)] << "\n";

      RewritePatternSet channelPatterns(&getContext());
      channelPatterns.add<ConvertAIRChannel>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(channelPatterns))))
          signalPassFailure();

    //   // Phase 2: lower launch/herd structure
    //   RewritePatternSet launchPatterns(&getContext());
    //   launchPatterns.add<ConvertAIRLaunch>(&getContext());
    //   if (failed(applyPatternsGreedily(getOperation(), std::move(launchPatterns))))
    //       signalPassFailure();
    }
};

} // namespace mlir::nki