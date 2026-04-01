#ifndef NKI_DIALECT_H
#define NKI_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "nki/IR/NKIOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "nki/IR/NKIOps.h.inc"

#endif // NKI_DIALECT_H
