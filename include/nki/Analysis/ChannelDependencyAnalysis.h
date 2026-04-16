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

struct Node {
  xilinx::air::HerdOp herd;
  // Edges this node produces into: (neighbor node, the channel connecting them)
  SmallVector<std::pair<Node *, xilinx::air::ChannelOp>> neighbors;
  unsigned inDegree = 0;
};

class ChannelDependencyAnalysis {
public:
  ChannelDependencyAnalysis(Operation *op);

  // query interface
  bool isFuseable();
  ChannelGraphType getGraphType();
  SmallVector<xilinx::air::HerdOp> getTopologicalOrder();
  SmallVector<xilinx::air::HerdOp> getProducers(xilinx::air::ChannelOp channel);
  SmallVector<xilinx::air::HerdOp> getConsumers(xilinx::air::ChannelOp channel);
  xilinx::air::ChannelOp getChannelBetween(xilinx::air::HerdOp producer,
                                            xilinx::air::HerdOp consumer);

private:
  void buildGraph(Operation *op);

  // Nodes in document (walk) order — stable, no pointer-hash non-determinism.
  SmallVector<std::unique_ptr<Node>> nodes;
  // Fast lookup: herd -> its node.
  DenseMap<xilinx::air::HerdOp, Node *> nodeMap;
};

} // namespace mlir::nki

#endif // NKI_ANALYSIS_CHANNEL_DEPENDENCY_ANALYSIS_H
