set(LLVM_TARGET_DEFINITIONS NKIDialect.td)
mlir_tablegen(NKIDialect.h.inc -gen-dialect-decls)
mlir_tablegen(NKIDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(NKIOps.h.inc -gen-op-decls)
mlir_tablegen(NKIOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRNKIIncGen)

add_mlir_dialect_library(MLIRNKIDialect
  NKIDialect.cpp
  NKIOps.cpp
  
  DEPENDS
  MLIRNKIIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
)