#include "nki/Analysis/ChannelDependencyAnalysis.h"

using namespace mlir;
using namespace mlir::nki;

ChannelDependencyAnalysis::ChannelDependencyAnalysis(Operation *op) {
  buildGraph(op);
}

void ChannelDependencyAnalysis::buildGraph(Operation *op) {
  // Build a symbol table from the top-level module to look up ChannelOps by name.
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    moduleOp = dyn_cast<ModuleOp>(op);
  SymbolTable symbolTable(moduleOp ? moduleOp.getOperation() : op);

  auto lookupChannel = [&](StringRef name) -> xilinx::air::ChannelOp {
    auto *sym = symbolTable.lookup(name);
    if (!sym)
      return {};
    return dyn_cast<xilinx::air::ChannelOp>(sym);
  };

  // walk all puts and gets, populate edges
  op->walk([&](xilinx::air::ChannelPutOp put) {
    auto herd = put->getParentOfType<xilinx::air::HerdOp>();
    if (herd) {
      if (auto channel = lookupChannel(put.getChanName())) {
        producerEdges[herd].push_back(channel);
      }
    }
  });

  op->walk([&](xilinx::air::ChannelGetOp get) {
    auto herd = get->getParentOfType<xilinx::air::HerdOp>();
    if (herd) {
      if (auto channel = lookupChannel(get.getChanName())) {
        consumerEdges[herd].push_back(channel);
      }
    }
  });
}

bool ChannelDependencyAnalysis::isFuseable() {
  auto type = getGraphType();
  return type == ChannelGraphType::LINEAR || type == ChannelGraphType::DAG;
}

ChannelGraphType ChannelDependencyAnalysis::getGraphType() {
  // Collect all unique channels across producer and consumer edges.
  DenseSet<xilinx::air::ChannelOp> allChannels;
  for (auto &[herd, channels] : producerEdges)
    for (auto ch : channels)
      allChannels.insert(ch);
  for (auto &[herd, channels] : consumerEdges)
    for (auto ch : channels)
      allChannels.insert(ch);

  // Count how many herds produce/consume each channel.
  DenseMap<xilinx::air::ChannelOp, unsigned> producerCount, consumerCount;
  for (auto &[herd, channels] : producerEdges)
    for (auto ch : channels)
      producerCount[ch]++;
  for (auto &[herd, channels] : consumerEdges)
    for (auto ch : channels)
      consumerCount[ch]++;

  bool hasFanout = false, hasFanin = false;
  for (auto ch : allChannels) {
    if (consumerCount[ch] > 1)
      hasFanout = true;
    if (producerCount[ch] > 1)
      hasFanin = true;
  }

  if (hasFanout && hasFanin)
    return ChannelGraphType::DAG;
  if (hasFanout)
    return ChannelGraphType::FANOUT;
  if (hasFanin)
    return ChannelGraphType::FANIN;

  // Check for cycles: if a herd both produces and consumes, it's cyclic.
  // TODO: Make this into a graph to reason through cycles easier
  // TODO: Fix the logic here, not sure if this is right
  for (auto &[herd, channels] : producerEdges)
    if (consumerEdges.count(herd))
      for (auto ch : channels)
        if (llvm::is_contained(consumerEdges[herd], ch))
          return ChannelGraphType::CYCLIC;

  return ChannelGraphType::LINEAR;
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getTopologicalOrder() {
  SmallVector<xilinx::air::HerdOp> order;
  // Herds that only produce (no incoming channels) come first.
  for (auto &[herd, _] : producerEdges)
    if (!consumerEdges.count(herd))
      order.push_back(herd);
  // Then herds that both consume and produce.
  for (auto &[herd, _] : consumerEdges)
    if (producerEdges.count(herd))
      order.push_back(herd);
  // Then herds that only consume.
  for (auto &[herd, _] : consumerEdges)
    if (!producerEdges.count(herd))
      order.push_back(herd);
  return order;
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getProducers(xilinx::air::ChannelOp channel) {
  SmallVector<xilinx::air::HerdOp> result;
  for (auto &[herd, channels] : producerEdges)
    for (auto ch : channels)
      if (ch == channel)
        result.push_back(herd);
  return result;
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getConsumers(xilinx::air::ChannelOp channel) {
  SmallVector<xilinx::air::HerdOp> result;
  for (auto &[herd, channels] : consumerEdges)
    for (auto ch : channels)
      if (ch == channel)
        result.push_back(herd);
  return result;
}

xilinx::air::ChannelOp ChannelDependencyAnalysis::getChannelBetween(
    xilinx::air::HerdOp producer, xilinx::air::HerdOp consumer) {
  auto prodIt = producerEdges.find(producer);
  auto consIt = consumerEdges.find(consumer);
  if (prodIt == producerEdges.end() || consIt == consumerEdges.end())
    return {};
  // Return the first channel that appears in both edge lists.
  for (auto ch : prodIt->second)
    if (llvm::is_contained(consIt->second, ch))
      return ch;
  return {};
}