#ifndef NKI_OPS_H
#define NKI_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::nki {

#define GEN_OP_DECLS
#include "nki/IR/NKIOps.h.inc"

} // namespace mlir::nki

#endif