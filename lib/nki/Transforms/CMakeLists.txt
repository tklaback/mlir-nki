add_mlir_dialect_library(NKITransforms
  NKIToPythonPass.cpp
  ConvertToNKIPass.cpp

  DEPENDS
  NKITransformsIncGen

  LINK_LIBS PUBLIC
  AIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRAffineDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)