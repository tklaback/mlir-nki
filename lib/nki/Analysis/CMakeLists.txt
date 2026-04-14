add_mlir_library(NKIAnalysis
  ChannelDependencyAnalysis.cpp
  LINK_LIBS PUBLIC
  AIRDialect
  MLIRIR
)