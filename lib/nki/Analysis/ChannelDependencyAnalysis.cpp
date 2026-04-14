ChannelDependencyAnalysis::ChannelDependencyAnalysis(Operation *op) {
  buildGraph(op);
}

void ChannelDependencyAnalysis::buildGraph(Operation *op) {
  // walk all puts and gets, populate edges
  op->walk([&](air::ChannelPutOp put) {
    auto herd = put->getParentOfType<air::HerdOp>();
    producerEdges[herd].push_back(put.getChannel());
  });

  op->walk([&](air::ChannelGetOp get) {
    auto herd = get->getParentOfType<air::HerdOp>();
    consumerEdges[herd].push_back(get.getChannel());
  });
}