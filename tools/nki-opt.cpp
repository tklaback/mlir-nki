#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "nki/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::nki::NKIDialect, mlir::func::FuncDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "NKI Optimizer", registry));
}