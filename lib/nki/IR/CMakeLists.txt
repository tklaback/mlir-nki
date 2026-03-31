add_mlir_library(NKIIR
  NKIOps.cpp

  DEPENDS
    NKIOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)