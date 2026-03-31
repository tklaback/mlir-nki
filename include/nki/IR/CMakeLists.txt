set(LLVM_TARGET_DEFINITIONS NKIOps.td)
mlir_tablegen(NKIOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(NKIOpsDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(NKIOps.h.inc -gen-op-decls)
mlir_tablegen(NKIOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(NKIIRIncGen)