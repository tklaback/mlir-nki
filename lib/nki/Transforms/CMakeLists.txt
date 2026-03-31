add_mlir_library(NKITransforms
  NKIToPythonPass.cpp

  DEPENDS
    NKIPassesIncGen

  LINK_LIBS PUBLIC
    NKIIR
    MLIRPass
    MLIRTransformUtils
)