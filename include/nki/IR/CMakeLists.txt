# Generate NKI Dialect
set(LLVM_TARGET_DEFINITIONS NKIDialect.td)
mlir_tablegen(NKIDialect.h.inc -gen-dialect-decls)
mlir_tablegen(NKIDialect.cpp.inc -gen-dialect-defs)

# Generate NKI Operations
set(LLVM_TARGET_DEFINITIONS NKIOps.td)
mlir_tablegen(NKIOps.h.inc -gen-op-decls)
mlir_tablegen(NKIOps.cpp.inc -gen-op-defs)

# Create public tablegen targets
add_public_tablegen_target(NKIIRIncGen)