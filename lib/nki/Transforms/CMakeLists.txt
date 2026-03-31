add_mlir_library(NKITransforms
  NKIToPython.cpp

  DEPENDS
    NKIPassesIncGen

  LINK_LIBS PUBLIC
    NKIIR
    MLIRPass
    MLIRTransformUtils
)