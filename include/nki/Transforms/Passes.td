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
    "mlir::affine::AffineDialect",
    "mlir::nki::NKIDialect"
  ];
}

def NKILinalgToElementwisePass : Pass<"nki-linalg-to-elementwise", "mlir::ModuleOp"> {
  let summary = "Lower linalg.generic elementwise ops to nki.elementwise";
  let description = [{
    Pattern-matches linalg.generic ops with all-parallel iterators and identity
    indexing maps whose bodies contain a single binary arith op, and replaces
    them with nki.elementwise<kind>.
  }];
  let dependentDialects = [
    "mlir::nki::NKIDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def NKILiftLoopsToLinalgPass : Pass<"nki-lift-loops-to-linalg", "mlir::ModuleOp"> {
  let summary = "Lift full-tile scf.for loop nests to linalg.generic";
  let description = [{
    Detects nested scf.for loops whose bounds cover the full memref shape
    and whose bodies perform elementwise scalar operations, then reconstructs
    them as linalg.generic ops with parallel iterators and identity indexing maps.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // NKI_PASSES
