#include "nki/Analysis/ChannelDependencyAnalysis.h"

using namespace mlir;
using namespace mlir::nki;

ChannelDependencyAnalysis::ChannelDependencyAnalysis(Operation *op) {
  buildGraph(op);
}

void ChannelDependencyAnalysis::buildGraph(Operation *op) {
  auto getOrCreateNode = [&](xilinx::air::HerdOp herd) -> Node * {
    if (auto *existing = nodeMap.lookup(herd))
      return existing;
    auto node = std::make_unique<Node>();
    node->herd = herd;
    Node *ptr = node.get();
    nodeMap[herd] = ptr;
    nodes.push_back(std::move(node));
    return ptr;
  };

  // Steps 1-3: walk all herds; for each put inside a herd, find all other
  // herds that have a get on the same channel and add an edge.
  op->walk([&](xilinx::air::HerdOp herdA) {
    herdA.walk([&](xilinx::air::ChannelPutOp put) {
      StringAttr chanName = put.getChanNameAttr();

      op->walk([&](xilinx::air::HerdOp herdB) {
        if (herdB == herdA) return;
        herdB.walk([&](xilinx::air::ChannelGetOp get) {
          if (get.getChanNameAttr() != chanName) return;

          // herdA produces into herdB via this channel.
          Node *nodeA = getOrCreateNode(herdA);
          Node *nodeB = getOrCreateNode(herdB);

          // Look up the ChannelOp declaration by name.
          auto moduleOp = op->getParentOfType<ModuleOp>();
          if (!moduleOp) moduleOp = dyn_cast<ModuleOp>(op);
          auto *sym = SymbolTable::lookupSymbolIn(moduleOp, chanName);
          auto channel = sym ? dyn_cast<xilinx::air::ChannelOp>(sym)
                              : xilinx::air::ChannelOp{};

          nodeA->neighbors.push_back({nodeB, channel});
          nodeB->inDegree++;
        });
      });
    });
  });
}

bool ChannelDependencyAnalysis::isFuseable() {
  auto type = getGraphType();
  return type == ChannelGraphType::LINEAR || type == ChannelGraphType::DAG;
}

ChannelGraphType ChannelDependencyAnalysis::getGraphType() {
  
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getTopologicalOrder() {
  
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getProducers(xilinx::air::ChannelOp channel) {
  
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getConsumers(xilinx::air::ChannelOp channel) {
  
}

xilinx::air::ChannelOp ChannelDependencyAnalysis::getChannelBetween(
    xilinx::air::HerdOp producer, xilinx::air::HerdOp consumer) {
      
}