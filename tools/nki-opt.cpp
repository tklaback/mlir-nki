#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "nki/Transforms/Passes.h"
#include "nki/IR/NKIOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<
    mlir::arith::ArithDialect,
    xilinx::air::airDialect,
    mlir::affine::AffineDialect,
    mlir::nki::NKIDialect, 
    mlir::linalg::LinalgDialect, 
    mlir::scf::SCFDialect, 
    mlir::func::FuncDialect,
    mlir::memref::MemRefDialect
  >();

  mlir::nki::registerNKIPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "NKI Optimizer", registry));
}