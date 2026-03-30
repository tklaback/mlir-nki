#ifndef DIALECT_H_
#define DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "nki/Transforms/Dialect.h.inc"

#define GET_OP_CLASSES
#include "nki/Transforms/Ops.h.inc"

#endif
