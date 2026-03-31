#include "nki/IR/NKIOps.h"

using namespace mlir;
using namespace mlir::nki;

#define GET_DIALECT_DEFS
#include "nki/IR/NKIOps.cpp.inc"

void mlir::nki::NKIDialect::initialize() {
  // Operations will be registered here once defined
}