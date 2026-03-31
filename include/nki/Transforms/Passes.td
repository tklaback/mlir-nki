include "mlir/Pass/PassBase.td"

def NKIToPython : Pass<"nki-to-python", "ModuleOp"> {
  let summary = "Convert NKI dialect to Python";
  let constructor = "nki::createNKIToPythonPass()";
}