#include "nki/IR/NKIOps.h"
using namespace mlir;
using namespace mlir::nki;

#define GET_DIALECT_DEFS
#include "nki/IR/NKIOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "nki/IR/NKIOps.cpp.inc"

void mlir::nki::NKIDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "nki/IR/NKIOps.cpp.inc"
  >();
}