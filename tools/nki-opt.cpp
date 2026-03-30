#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "nki/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::nki::NKIDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "NKI Optimizer", registry));
}