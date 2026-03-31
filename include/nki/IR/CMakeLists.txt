# Generate NKI Dialect
set(LLVM_TARGET_DEFINITIONS NKIOps.td)
mlir_tablegen(NKIOps.h.inc -gen-dialect-decls)
mlir_tablegen(NKIOps.cpp.inc -gen-dialect-defs)

# Create public tablegen targets
add_public_tablegen_target(NKIIRIncGen)