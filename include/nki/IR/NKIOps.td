#ifndef NKI_DIALECT
#define NKI_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

def NKI_Dialect : Dialect {
  let name = "nki";
  let cppNamespace = "::mlir::nki";
  let summary = "NKI Dialect for lowering to nki python";
  let description = [{

  }];
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

def NKI_LaunchOp : NKI_Op<"launch", [SingleBlock, NoTerminator]> {
  let arguments = (ins
    Index:$grid_x,
    Index:$grid_y
  );
  let results = (outs);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `grid` `[` $grid_x `,` $grid_y `]`
    $body attr-dict
  }];
}

#endif
