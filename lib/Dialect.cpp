
#include "nki/Transforms/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::nki;

#include "nki/Transforms/Dialect.cpp.inc"

void NKIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nki/Transforms/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "nki/Transforms/Ops.cpp.inc"
