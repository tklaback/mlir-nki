add_mlir_library(NKIIR
  NKIDialect.cpp

  DEPENDS
    NKIIRIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)