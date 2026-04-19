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

// For a LINEAR channel graph, fuse all herds in topological order into a
// single merged herd, eliminating the herd-to-herd channels.
//
// For each consecutive pair (producer, consumer) in topological order:
//   - Find the channel.put in the producer and the channel.get in the consumer.
//   - Replace all uses of the get's dst buffer with the put's src buffer
//     (so the consumer operates directly on the producer's allocation).
//   - Erase the put, the get, and the channel declaration.
// Then splice all subsequent herd bodies into order[0], leaving one herd.
struct FuseLinearHerds : public OpRewritePattern<xilinx::air::SegmentOp> {
  ChannelDependencyAnalysis *analysis;

  FuseLinearHerds(MLIRContext *ctx, ChannelDependencyAnalysis *analysis)
      : OpRewritePattern(ctx), analysis(analysis) {}

  LogicalResult matchAndRewrite(xilinx::air::SegmentOp segment,
                                PatternRewriter &rewriter) const override {
    SmallVector<xilinx::air::HerdOp> order = analysis->getTopologicalOrder();
    if (order.size() < 2)
      return rewriter.notifyMatchFailure(segment, "nothing to fuse");

    // Verify all herds live inside this segment.
    for (auto herd : order)
      if (herd->getParentOfType<xilinx::air::SegmentOp>() != segment) {
        return rewriter.notifyMatchFailure(segment, "herds not in this segment");
      }

    // For each consecutive pair, eliminate the connecting channel.
    // Collect sharedBufs here (before erasing puts) so we can re-insert
    // deallocs after the merge.
    SmallVector<Value> sharedBufsToDealloc;
    for (unsigned i = 0; i + 1 < order.size(); ++i) {
      auto producer = order[i];
      auto consumer = order[i + 1];

      xilinx::air::ChannelOp channel =
          analysis->getChannelBetween(producer, consumer);
      if (!channel) {
        // llvm::errs() << "NO CHANNEL BETWEEN HERDS\n";
        return rewriter.notifyMatchFailure(segment, "no channel between herds");
      }

      StringAttr chanName = channel.getSymNameAttr();

      // Find the put in the producer herd body.
      xilinx::air::ChannelPutOp putOp;
      producer.walk([&](xilinx::air::ChannelPutOp op) {
        if (op.getChanName() == chanName)
          putOp = op;
      });
      // Find the get in the consumer herd body.
      xilinx::air::ChannelGetOp getOp;
      consumer.walk([&](xilinx::air::ChannelGetOp op) {
        if (op.getChanName() == chanName)
          getOp = op;
      });

      if (!putOp || !getOp)
        return rewriter.notifyMatchFailure(
            segment, "could not find put/get for herd-to-herd channel");

      Value sharedBuf = putOp.getSrc();
      Value consumedBuf = getOp.getDst();

      // Save sharedBuf before erasing putOp so we can dealloc it later.
      sharedBufsToDealloc.push_back(sharedBuf);

      // Replace consumedBuf uses with sharedBuf so the consumer ops refer to
      // the producer's allocation directly.
      rewriter.replaceAllUsesWith(consumedBuf, sharedBuf);
      rewriter.eraseOp(getOp);
      rewriter.eraseOp(putOp);

      // Remove all deallocs of sharedBuf from both herds — the producer's
      // dealloc fires too early (before consumer ops run), and after RAUW the
      // consumer may have a dealloc of consumedBuf that is now a double-free.
      // We re-insert a single dealloc just before the merged terminator below.
      SmallVector<memref::DeallocOp> toErase;
      producer.walk([&](memref::DeallocOp dealloc) {
        if (dealloc.getMemref() == sharedBuf)
          toErase.push_back(dealloc);
      });
      consumer.walk([&](memref::DeallocOp dealloc) {
        if (dealloc.getMemref() == sharedBuf)
          toErase.push_back(dealloc);
      });
      for (auto dealloc : toErase)
        rewriter.eraseOp(dealloc);

      // Erase the channel declaration from the module.
      rewriter.eraseOp(channel);
    }

    // Splice bodies of order[1..] into order[0] before its terminator,
    // so all ops run sequentially in producer-first order.
    //
    // HerdOp body blocks have block arguments:
    //   [tile_x, tile_y, size_x, size_y, kernel_operands...]
    // inlineBlockBefore requires the source block to have no block args, so we
    // replace them with concrete values before splicing.
    auto baseHerd = order[0];
    Block &baseBlock = baseHerd.getBody().front();
    Operation *baseTerminator = baseBlock.getTerminator();

    for (unsigned i = 1; i < order.size(); ++i) {
      auto herd = order[i];
      Block &srcBlock = herd.getBody().front();

      // Build replacement values for the source block's arguments.
      // Tile indices -> arith.constant 0 (sequential execution, one tile).
      // Size args -> the herd's corresponding size operands.
      // Kernel operands -> the herd's corresponding kernel operand values.
      SmallVector<Value> argReplacements;
      Location loc = herd.getLoc();
      unsigned numTiles = herd.getNumDims(); // number of tile index args
      rewriter.setInsertionPoint(baseTerminator);
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      for (unsigned t = 0; t < numTiles; ++t)
        argReplacements.push_back(zero);
      for (Value size : herd.getSizes())
        argReplacements.push_back(size);
      for (Value kop : herd.getKernelOperands())
        argReplacements.push_back(kop);

      rewriter.eraseOp(srcBlock.getTerminator());
      rewriter.inlineBlockBefore(&srcBlock, baseTerminator, argReplacements);
      rewriter.eraseOp(herd);
    }

    // Re-insert deallocs for sharedBufs just before the merged terminator.
    // Now that all consumer ops are in the same block, this is safe.
    rewriter.setInsertionPoint(baseTerminator);
    for (Value buf : sharedBufsToDealloc)
      memref::DeallocOp::create(rewriter, baseHerd.getLoc(), buf);

    return success();
  }
};

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
    } else if (putInHerd && getInHerd) {
      // Both put and get are inside herds. FuseLinearHerds should have already
      // eliminated all herd-to-herd channels before this pattern runs.
      // If we reach here, it means the graph was not LINEAR (e.g. DAG/CYCLIC)
      // and this case is not yet implemented.
      return rewriter.notifyMatchFailure(
          put, "herd-to-herd channel not eliminated by fusion (non-LINEAR?)");
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
  }
};

struct ConvertAIRLaunch : public OpRewritePattern<xilinx::air::LaunchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::LaunchOp launch,
                                PatternRewriter &rewriter) const override {
    // Expect exactly one herd inside the launch (segment already inlined).
    xilinx::air::HerdOp herd;
    launch.walk([&](xilinx::air::HerdOp h) { herd = h; });
    if (!herd)
      return rewriter.notifyMatchFailure(launch, "no herd in launch");

    Location loc = launch.getLoc();

    // Step 1: create nki.launch before the air.launch.
    rewriter.setInsertionPoint(launch);
    auto nkiLaunch = nki::LaunchOp::create(rewriter, loc);
    Block *nkiBlock = rewriter.createBlock(&nkiLaunch.getBody());

    // Step 2: replace herd block args with the outer values they represent.
    // Block arg layout: [tile_0..tile_N, size_0..size_N, kop_0..kop_M]
    // tile args are unused after fusion (sequential execution) — replace with
    // the herd's size operands[0] as a dummy index (they'll be DCE'd anyway).
    // size args -> herd.getSizes()[i]   (defined in the launch body)
    // kop args  -> herd.getKernelOperands()[i]  (defined in the launch body)
    Block &herdBody = herd.getBody().front();
    unsigned numTiles = herd.getNumDims();

    // printf("NUM TILES: %u\n", numTiles);

    rewriter.setInsertionPointToStart(nkiBlock);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // Build the full replacement list for mergeBlocks — must match
    // herdBody.getNumArguments() exactly: [tile..., size..., kop...]
    SmallVector<Value> argReplacements;
    for (unsigned t = 0; t < numTiles; ++t)
      argReplacements.push_back(zero);
    for (Value size : herd.getSizes())
      argReplacements.push_back(size);
    for (Value kop : herd.getKernelOperands())
      argReplacements.push_back(kop);

    // Step 3: merge herd body into nki.launch body and erase the herd.
    rewriter.eraseOp(herdBody.getTerminator());
    rewriter.mergeBlocks(&herdBody, nkiBlock, argReplacements);
    rewriter.eraseOp(herd);

    // Step 4: inline air.launch body before the launch op, then erase it.
    // Replace all launch body block args with their corresponding operands,
    // then inline with no remaining args.
    Block &launchBody = launch.getBody().front();
    unsigned numLaunchAsyncDeps = launch.getAsyncDependencies().size();
    unsigned numLaunchSizes = launch.getSizes().size();
    unsigned launchArgOffset = numLaunchAsyncDeps + numLaunchSizes;
    SmallVector<Value> launchArgReplacements;
    for (unsigned i = 0; i < launchArgOffset; ++i)
      launchArgReplacements.push_back(zero);
    for (Value operand : launch.getLaunchOperands())
      launchArgReplacements.push_back(operand);
    rewriter.eraseOp(launchBody.getTerminator());
    rewriter.inlineBlockBefore(&launchBody, launch, launchArgReplacements);
    rewriter.eraseOp(launch);

    return success();
  }
};

} // namespace

struct ConvertAIRToNKIPass
    : public impl::ConvertAIRToNKIPassBase<ConvertAIRToNKIPass> {
    void runOnOperation() override {
      auto &analysis = getAnalysis<ChannelDependencyAnalysis>();
      ChannelGraphType graphType = analysis.getGraphType();

      if (graphType == ChannelGraphType::LINEAR) {
        // Phase 1: fuse herds — erase herd-to-herd channels, merge bodies.
        RewritePatternSet fusePatterns(&getContext());
        fusePatterns.add<FuseLinearHerds>(&getContext(), &analysis);
        if (failed(applyPatternsGreedily(getOperation(), std::move(fusePatterns))))
          signalPassFailure();

        // Phase 2: inline air.segment into the launch body.
        RewritePatternSet inlinePatterns(&getContext());
        inlinePatterns.add<InlineSegment>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(inlinePatterns))))
          signalPassFailure();

        // Phase 3: lower launch↔herd boundary channels (load/store).
        RewritePatternSet channelPatterns(&getContext());
        channelPatterns.add<ConvertAIRChannel>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(channelPatterns))))
          signalPassFailure();

        // Erase air.channel declarations now that all put/get ops are gone.
        getOperation()->walk([](xilinx::air::ChannelOp ch) {
          ch.erase();
        });

        // // Phase 4: convert air.launch + air.herd -> nki.launch.
        RewritePatternSet launchPatterns(&getContext());
        launchPatterns.add<ConvertAIRLaunch>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(launchPatterns))))
          signalPassFailure();
      } else if (graphType == ChannelGraphType::DAG) {
        // TODO: parallel launch with barriers
      } else {
        // CYCLIC, FANOUT, FANIN: not yet implemented.
        signalPassFailure();
      }
    }
};

} // namespace mlir::nki
