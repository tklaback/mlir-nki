add_mlir_dialect_library(NKITransforms
  NKIToPythonPass.cpp

  DEPENDS
  NKITransformsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)