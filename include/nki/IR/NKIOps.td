#ifndef NKI_OPS
#define NKI_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NKI_Dialect : Dialect {
  let name = "nki";
  let cppNamespace = "::mlir::nki";
  let summary = "NKI dialect";
}

class NKI_Op<string mnemonic, list<Trait> traits = []> :
    Op<NKI_Dialect, mnemonic, traits>;

def NKI_LoadOp : NKI_Op<"load"> {
  let arguments = (ins AnyMemRef:$src);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$src `:` type($src) `->` type($result) attr-dict";
}

def NKI_StoreOp : NKI_Op<"store"> {
  let arguments = (ins AnyTensor:$src, AnyMemRef:$dst);
  let assemblyFormat = "$src `,` $dst `:` type($src) `,` type($dst) attr-dict";
}

#endif
