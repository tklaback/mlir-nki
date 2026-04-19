add_mlir_dialect_library(NKITransforms
  NKIToPythonPass.cpp
  ConvertToNKIPass.cpp
  NKILiftLoopsToLinalgPass.cpp

  DEPENDS
  NKITransformsIncGen

  LINK_LIBS PUBLIC
  AIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)