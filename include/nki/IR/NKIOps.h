#ifndef NKI_OPS_H
#define NKI_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "nki/IR/NKIDialect.h"

#define GET_OP_CLASSES
#include "nki/IR/NKIOps.h.inc"

#endif // NKI_OPS_H