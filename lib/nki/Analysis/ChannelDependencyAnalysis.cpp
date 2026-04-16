#include "nki/Analysis/ChannelDependencyAnalysis.h"

using namespace mlir;
using namespace mlir::nki;

ChannelDependencyAnalysis::ChannelDependencyAnalysis(Operation *op) {
  buildGraph(op);
}

void ChannelDependencyAnalysis::buildGraph(Operation *op) {
  
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