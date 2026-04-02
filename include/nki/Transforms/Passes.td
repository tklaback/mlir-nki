#ifndef NKI_PASSES
#define NKI_PASSES

include "mlir/Pass/PassBase.td"

def NKIToPythonPass : Pass<"nki-to-python", "mlir::ModuleOp"> {
  let summary = "Convert NKI operations to Python";
  let description = [{
    Converts NKI dialect operations to Python representations.
  }];
}

def ConvertAIRToNKIPass : Pass<"convert-air-to-nki", "mlir::ModuleOp"> {
  let summary = "Convert AIR dialect operations to NKI";
  let description = [{
    Converts air.launch and air.herd operations to nki.launch.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect"
  ];
}

#endif // NKI_PASSES
