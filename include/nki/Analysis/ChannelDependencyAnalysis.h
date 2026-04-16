#ifndef NKI_ANALYSIS_CHANNEL_DEPENDENCY_ANALYSIS_H
#define NKI_ANALYSIS_CHANNEL_DEPENDENCY_ANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "air/Dialect/AIR/AIRDialect.h"

namespace mlir::nki {
  enum class ChannelGraphType {
    LINEAR,
    DAG,
    CYCLIC,
    FANOUT,
    FANIN,
  };
  class ChannelDependencyAnalysis {
  public:
    ChannelDependencyAnalysis(Operation *op);

    // query interface
    bool isFuseable();
    ChannelGraphType getGraphType(); // LINEAR, DAG, FAN
    SmallVector<xilinx::air::HerdOp> getTopologicalOrder();
    SmallVector<xilinx::air::HerdOp> getProducers(xilinx::air::ChannelOp channel);
    SmallVector<xilinx::air::HerdOp> getConsumers(xilinx::air::ChannelOp channel);
    // Returns the channel connecting producer -> consumer, or null if none.
    xilinx::air::ChannelOp getChannelBetween(xilinx::air::HerdOp producer,
                                              xilinx::air::HerdOp consumer);

  private:
    void buildGraph(Operation *op);
    
    // adjacency: herd -> channels it produces/consumes
    DenseMap<xilinx::air::HerdOp, SmallVector<xilinx::air::ChannelOp>> producerEdges;
    DenseMap<xilinx::air::HerdOp, SmallVector<xilinx::air::ChannelOp>> consumerEdges;
  };
} // namespace mlir::nki

#endif // NKI_ANALYSIS_CHANNEL_DEPENDENCY_ANALYSIS_H