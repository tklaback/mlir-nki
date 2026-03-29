add_mlir_conversion_library(MLIRAIRToNKI
  AIRToNKI.cpp
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRNKIDialect
  MLIRSCY
)