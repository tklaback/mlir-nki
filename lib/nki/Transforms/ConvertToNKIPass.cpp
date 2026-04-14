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

struct ConvertAIRChannel : public OpRewritePattern<xilinx::air::ChannelPutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
    xilinx::air::ChannelPutOp put, 
    PatternRewriter &rewriter
  ) const override {
    return success();
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
  
      if (analysis.getGraphType() == ChannelGraphType::LINEAR) {
        // fuse linearly
      } else if (analysis.getGraphType() == ChannelGraphType::DAG) {
        // topological fuse
      }

      RewritePatternSet channelPatterns(&getContext());
      channelPatterns.add<ConvertAIRChannel>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(channelPatterns))))
          signalPassFailure();

      // Phase 2: lower launch/herd structure
      RewritePatternSet launchPatterns(&getContext());
      launchPatterns.add<ConvertAIRLaunch>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(launchPatterns))))
          signalPassFailure();
    }
};

} // namespace mlir::nki