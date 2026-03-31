add_mlir_library(NKIIR
  NKIOps.cpp

  DEPENDS
    NKIIRIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)