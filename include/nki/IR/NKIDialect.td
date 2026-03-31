#ifndef NKI_DIALECT
#define NKI_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

def NKI_Dialect : Dialect {
  let name = "nki";
  let cppNamespace = "::mlir::nki";
  let summary = "NKI dialect for learning MLIR";
  let description = [{
    The NKI dialect provides a simple set of operations
    for learning and understanding MLIR fundamentals.
  }];
}

#endif
