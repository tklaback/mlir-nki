#include "nki/IR/NKIOps.h"
#include "nki/Transforms/Passes.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/IRMapping.h"
#include <iostream>

namespace mlir::nki {

#define GEN_PASS_DEF_CONVERTAIRTONKIPASS
#include "nki/Transforms/Passes.h.inc"

namespace {

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
      // This is the case where the put is not in a herd (i.e. a put operation at the launch level; placing values in the channel)
      // This should be converted into a nki.load inside of the herd (at its air.channel.get location)
      // The get in the herd gets into a destination tile. Nki.load is different because it doesn't load into that destination buffer, it returns the tile that is then used in place of the destination tile. So, we need to also replace the destination tile and all of its occurrences.
      // Need to make sure I am not violating dominance by using the return value of nki.load in a path not dominated by the herd.
    } else if (getInHerd) {
      // This is the case that the put is in the herd and the get is in the herd
      // For this case, we want the producer herd (the one with the put) to replace its air.channel.put with a nki.store into sbuf.
      // Then, we want the consumer herd to have a barrier that waits until that operation is finished.
    } else {
      // This is the case that the put is in the herd and the get is not in the herd.
      // For this case, we need to replace the air.channel.put inside the herd with nki.store.
    }

    return success();
  }
};

struct ConvertAIRLaunch : public OpRewritePattern<xilinx::air::LaunchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::LaunchOp launch,
                                PatternRewriter &rewriter) const override {
    OperandRange launchSizes = launch.getSizeOperands();

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

    SmallVector<xilinx::air::HerdOp> herds;
    launch.walk([&](xilinx::air::HerdOp herd) {
      herds.push_back(herd);
    });

    rewriter.setInsertionPoint(launch);
    if (herds.size() == 0) {
      auto nkiLaunch = nki::LaunchOp::create(rewriter, launch.getLoc(), launchSizes);
      nkiLaunch.getBody().emplaceBlock();
      Block *nkiBlock = &nkiLaunch.getBody().front();
      launchBlock->back().erase(); // erase launch terminator
      nkiBlock->getOperations().splice(nkiBlock->end(), launchBlock->getOperations());
    }
    else {
      for (int i = 0; i < herds.size(); i++) {
        xilinx::air::HerdOp curHerd = herds[i];
        OperandRange herdLaunchSizes = curHerd.getSizeOperands();
        // herdLaunchSizes are defined inside the launch body. Clone their defining
        // ops to before the launch so gridX/gridY live in the outer scope.
        IRMapping sizeMapping;
        Value herdSizeX = rewriter.clone(*herdLaunchSizes[0].getDefiningOp(), sizeMapping)->getResult(0);
        Value herdSizeY = rewriter.clone(*herdLaunchSizes[1].getDefiningOp(), sizeMapping)->getResult(0);
        Value gridX = launchSizes.empty() ? herdSizeX
                                          : arith::MulIOp::create(rewriter, launch.getLoc(), herdSizeX, launchSizes[0]);
        Value gridY = launchSizes.empty() ? herdSizeY
                                          : arith::MulIOp::create(rewriter, launch.getLoc(), herdSizeY, launchSizes[1]);

        auto nkiLaunch = nki::LaunchOp::create(rewriter, launch.getLoc(), ValueRange{gridX, gridY});

        // Each NKI launch should contain air.launches contents outside of the herds, but, then each nki launch's rest of its body will be the herd it is associated with.

        nkiLaunch.getBody().emplaceBlock();
        Block *nkiBlock = &nkiLaunch.getBody().front();

        // Clone non-herd ops from the launch body into nki.launch
        // IRMapping tracks value substitutions so cloned ops reference each other correctly.
        IRMapping mapping;
        rewriter.setInsertionPointToEnd(nkiBlock);
        for (Operation &op : *launchBlock) {
          if (isa<xilinx::air::HerdOp>(op) || op.hasTrait<OpTrait::IsTerminator>())
            continue;

          rewriter.clone(op, mapping);
        }

        for (auto [idArg, sizeArg, sizeVal] :
            llvm::zip(curHerd.getIds(), curHerd.getSize(), curHerd.getSizeOperands()))  {
          Value(idArg).replaceAllUsesWith(sizeVal);  // placeholder: use size as stand-in
          Value(sizeArg).replaceAllUsesWith(sizeVal);
        }

        for (auto [kernelArg, operand] :
          llvm::zip(curHerd.getKernelArguments(), curHerd.getKernelOperands()))
          Value(kernelArg).replaceAllUsesWith(operand);

        Block *herdBlock = &curHerd.getBody().front();

        herdBlock->eraseArguments([](BlockArgument) { return true; });

        // Remap herd body operands to point to cloned copies instead of originals.
        herdBlock->walk([&](Operation *op) {
          for (OpOperand &operand : op->getOpOperands())
            if (Value mapped = mapping.lookupOrNull(operand.get()))
              operand.set(mapped);
        });

        herdBlock->back().erase();

        nkiBlock->getOperations().splice(nkiBlock->end(), herdBlock->getOperations());
        curHerd->erase();

      }
    }

    rewriter.eraseOp(launch);
    return success();
  }
};

} // namespace

struct ConvertAIRToNKIPass
    : public impl::ConvertAIRToNKIPassBase<ConvertAIRToNKIPass> {
  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());

    patterns.add<ConvertAIRChannel>(&getContext());
    patterns.add<ConvertAIRLaunch>(&getContext());

    // check for failure
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::nki