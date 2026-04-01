#ifndef NKI_PASSES
#define NKI_PASSES

include "mlir/Pass/PassBase.td"

def NKIToPythonPass : Pass<"nki-to-python", "mlir::ModuleOp"> {
  let summary = "Convert NKI operations to Python";
  let description = [{
    Converts NKI dialect operations to Python representations.
  }];
}

#endif // NKI_PASSES
