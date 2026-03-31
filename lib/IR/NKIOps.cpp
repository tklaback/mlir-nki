#include "nki/IR/NKIOps.h"

using namespace mlir;
using namespace mlir::nki;

#define GEN_DIALECT_DEFS
#include "nki/IR/NKIOps.h.inc"

#define GEN_OP_DEFS
#include "nki/IR/NKIOps.h.inc"