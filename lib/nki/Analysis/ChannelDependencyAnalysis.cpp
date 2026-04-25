#include "nki/Analysis/ChannelDependencyAnalysis.h"

using namespace mlir;
using namespace mlir::nki;

ChannelDependencyAnalysis::ChannelDependencyAnalysis(Operation *op)
{
  buildGraph(op);
}

void ChannelDependencyAnalysis::buildGraph(Operation *op)
{
  auto getOrCreateNode = [&](xilinx::air::HerdOp herd) -> Node *
  {
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
  op->walk([&](xilinx::air::HerdOp herdA)
           { herdA.walk([&](xilinx::air::ChannelPutOp put)
                        {
      StringAttr chanName = put.getChanNameAttr().getAttr();

      op->walk([&](xilinx::air::HerdOp herdB) {
        if (herdB == herdA) return;
        herdB.walk([&](xilinx::air::ChannelGetOp get) {
          if (get.getChanNameAttr().getAttr() != chanName) return;

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
      }); }); });
}

bool ChannelDependencyAnalysis::isFuseable()
{
  auto type = getGraphType();
  return type == ChannelGraphType::LINEAR || type == ChannelGraphType::DAG;
}

ChannelGraphType ChannelDependencyAnalysis::getGraphType()
{
  // TODO: implement
  return ChannelGraphType::LINEAR;
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getTopologicalOrder()
{
  DenseMap<Node *, unsigned> inDeg;
  for (auto &n : nodes)
    inDeg[n.get()] = n->inDegree;

  SmallVector<Node *> queue;
  for (auto &n : nodes)
    if (inDeg[n.get()] == 0)
      queue.push_back(n.get());

  SmallVector<xilinx::air::HerdOp> result;
  while (!queue.empty())
  {
    Node *cur = queue.front();
    queue.erase(queue.begin());
    result.push_back(cur->herd);
    for (auto [nbr, ch] : cur->neighbors)
    {
      assert(inDeg[nbr] > 0 && "ERROR, unreachable");
      if (--inDeg[nbr] == 0)
        queue.push_back(nbr);
    }
  }
  return result;
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getProducers(xilinx::air::ChannelOp channel)
{
  SmallVector<xilinx::air::HerdOp> result;
  for (auto &node : nodes)
    for (auto [nbr, ch] : node->neighbors)
      if (ch == channel)
      {
        result.push_back(node->herd);
        break;
      }
  return result;
}

SmallVector<xilinx::air::HerdOp>
ChannelDependencyAnalysis::getConsumers(xilinx::air::ChannelOp channel)
{
  SmallVector<xilinx::air::HerdOp> result;
  for (auto &node : nodes)
    for (auto [nbr, ch] : node->neighbors)
      if (ch == channel)
      {
        result.push_back(nbr->herd);
        break;
      }
  return result;
}

xilinx::air::ChannelOp ChannelDependencyAnalysis::getChannelBetween(
    xilinx::air::HerdOp producer, xilinx::air::HerdOp consumer)
{
  Node *prodNode = nodeMap.lookup(producer);
  if (!prodNode)
    return {};
  for (auto [nbr, ch] : prodNode->neighbors)
    if (nbr->herd == consumer)
      return ch;
  return {};
}

SmallVector<xilinx::air::ChannelOp> ChannelDependencyAnalysis::getChannelsBetween(
    xilinx::air::HerdOp a, xilinx::air::HerdOp b)
{
  SmallVector<xilinx::air::ChannelOp> result;
  Node *nodeA = nodeMap.lookup(a);
  if (!nodeA)
    return result;
  for (auto [nbr, ch] : nodeA->neighbors)
    if (nbr->herd == b)
      result.push_back(ch);
  return result;
}
bool ChannelDependencyAnalysis::hasMultipleChannelsBetween(
    HerdOp a, HerdOp b)
{
  return getChannelsBetween(a, b).size() > 1;
}