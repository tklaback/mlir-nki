set(LLVM_TARGET_DEFINITIONS NKIOps.td)
mlir_tablegen(NKIOps.h.inc -gen-op-decls -I${MLIR_INCLUDE_DIR})
mlir_tablegen(NKIOps.cpp.inc -gen-op-defs -I${MLIR_INCLUDE_DIR})
add_public_tablegen_target(NKIOpsIncGen)