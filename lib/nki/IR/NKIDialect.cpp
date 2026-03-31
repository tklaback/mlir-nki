#include "nki/IR/NKIDialect.h"
#include "nki/IR/NKIOps.h"

using namespace mlir;
using namespace mlir::nki;

#include "nki/IR/NKIDialect.cpp.inc"

void mlir::nki::NKIDialect::initialize() {
  addOperations<
  #define GET_OP_LIST
  #include "nki/IR/NKIOps.cpp.inc"
  >();
}