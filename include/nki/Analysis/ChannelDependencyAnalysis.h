class ChannelDependencyAnalysis {
public:
  ChannelDependencyAnalysis(Operation *op);

  // query interface
  bool isFuseable();
  GraphType getGraphType(); // LINEAR, DAG, FAN
  SmallVector<air::HerdOp> getTopologicalOrder();
  SmallVector<air::HerdOp> getProducers(air::ChannelOp channel);
  SmallVector<air::HerdOp> getConsumers(air::ChannelOp channel);

private:
  void buildGraph(Operation *op);
  
  // adjacency: herd -> channels it produces/consumes
  DenseMap<air::HerdOp, SmallVector<air::ChannelOp>> producerEdges;
  DenseMap<air::HerdOp, SmallVector<air::ChannelOp>> consumerEdges;
};