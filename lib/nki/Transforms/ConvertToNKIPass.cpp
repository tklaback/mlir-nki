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
    if (!matchedGet)
      return rewriter.notifyMatchFailure(put, "no matching channel get");

    bool putInHerd = (bool)put->getParentOfType<xilinx::air::HerdOp>();
    bool getInHerd = (bool)matchedGet->getParentOfType<xilinx::air::HerdOp>();

    if (!putInHerd) {
      // Case 1: launch put + herd get → nki.load at the get's location.
      // The put describes the HBM source; the get's dst is the SBUF allocation.
      OperandRange offsets = put.getSrcOffsets();
      OperandRange sizes   = put.getSrcSizes();
      OperandRange strides = put.getSrcStrides();
      auto resultType = cast<MemRefType>(matchedGet.getDst().getType());
      rewriter.setInsertionPoint(matchedGet);
      auto load = nki::LoadOp::create(rewriter, put.getLoc(), resultType,
                                      put.getSrc(),
                                      offsets[0], offsets[1],
                                      sizes[0],   sizes[1],
                                      strides[0], strides[1]);
      rewriter.replaceAllUsesWith(matchedGet.getDst(), load.getResult());
      rewriter.eraseOp(matchedGet);
      rewriter.eraseOp(put);
    } else if (getInHerd) {
      // Case 3: herd put + herd get → route through a shared HBM buffer.
      // Allocate a flat HBM buffer at the segment level, store into it from
      // the put's herd, and load from it into the get's herd.
      auto srcType = cast<MemRefType>(put.getSrc().getType());
      // Strip the memory space (space 2 = SBUF) to get a plain HBM memref.
      auto hbmType = MemRefType::get(srcType.getShape(), srcType.getElementType());

      // Insert the HBM alloc before the segment that contains both herds.
      auto segment = put->getParentOfType<xilinx::air::SegmentOp>();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(segment ? (Operation*)segment : (Operation*)put->getParentOfType<xilinx::air::LaunchOp>());
      Value hbmBuf = memref::AllocOp::create(rewriter, put.getLoc(), hbmType, ValueRange{}).getResult();

      // Store from sender herd into HBM buffer at the put's location.
      rewriter.setInsertionPoint(put);
      nki::StoreOp::create(rewriter, put.getLoc(),
                           put.getSrc(), hbmBuf,
                           arith::ConstantIndexOp::create(rewriter, put.getLoc(), 0),
                           arith::ConstantIndexOp::create(rewriter, put.getLoc(), 0),
                           arith::ConstantIndexOp::create(rewriter, put.getLoc(), srcType.getDimSize(0)),
                           arith::ConstantIndexOp::create(rewriter, put.getLoc(), srcType.getDimSize(1)),
                           arith::ConstantIndexOp::create(rewriter, put.getLoc(), srcType.getDimSize(1)),
                           arith::ConstantIndexOp::create(rewriter, put.getLoc(), 1));

      // Load from HBM buffer into receiver herd at the get's location.
      auto getDstType = cast<MemRefType>(matchedGet.getDst().getType());
      rewriter.setInsertionPoint(matchedGet);
      auto load = nki::LoadOp::create(rewriter, matchedGet.getLoc(), getDstType,
                                      hbmBuf,
                                      arith::ConstantIndexOp::create(rewriter, matchedGet.getLoc(), 0),
                                      arith::ConstantIndexOp::create(rewriter, matchedGet.getLoc(), 0),
                                      arith::ConstantIndexOp::create(rewriter, matchedGet.getLoc(), getDstType.getDimSize(0)),
                                      arith::ConstantIndexOp::create(rewriter, matchedGet.getLoc(), getDstType.getDimSize(1)),
                                      arith::ConstantIndexOp::create(rewriter, matchedGet.getLoc(), getDstType.getDimSize(1)),
                                      arith::ConstantIndexOp::create(rewriter, matchedGet.getLoc(), 1));
      rewriter.replaceAllUsesWith(matchedGet.getDst(), load.getResult());
      rewriter.eraseOp(matchedGet);
      rewriter.eraseOp(put);
    } else {
      // Case 2: herd put + launch get → nki.store at the put's location.
      // The get describes the HBM destination; the put's src is the SBUF tile.
      OperandRange offsets = matchedGet.getDstOffsets();
      OperandRange sizes   = matchedGet.getDstSizes();
      OperandRange strides = matchedGet.getDstStrides();
      rewriter.setInsertionPoint(put);
      nki::StoreOp::create(rewriter, put.getLoc(),
                           put.getSrc(), matchedGet.getDst(),
                           offsets[0], offsets[1],
                           sizes[0],   sizes[1],
                           strides[0], strides[1]);
      rewriter.eraseOp(matchedGet);
      rewriter.eraseOp(put);
    }

    return success();
  }
};

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

    SmallVector<xilinx::air::HerdOp> herds;
    launch.walk([&](xilinx::air::HerdOp herd) {
      herds.push_back(herd);
    });

    rewriter.setInsertionPoint(launch);
    if (herds.size() == 0) {
      auto nkiLaunch = nki::LaunchOp::create(rewriter, launch.getLoc(), launchSizes[0], launchSizes[1]);
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
        Value gridX = arith::MulIOp::create(rewriter, launch.getLoc(), herdSizeX, launchSizes[0]);
        Value gridY = arith::MulIOp::create(rewriter, launch.getLoc(), herdSizeY, launchSizes[1]);

        auto nkiLaunch = nki::LaunchOp::create(rewriter, launch.getLoc(), gridX, gridY);

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