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

def NKI_LoadOp : NKI_Op<"load",
    [AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$src,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $src `[` $offsets `]`
         `[` $sizes `]`
         `[` $strides `]`
    `:` type($src) `->` type($result) attr-dict
  }];
}


def NKI_StoreOp : NKI_Op<"store", [AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );
  let assemblyFormat = [{
    $src `,` $dst
         `[` $offsets `]`
         `[` $sizes `]`
         `[` $strides `]`
    `:` type($src) `,` type($dst) attr-dict
  }];
}


// Returns the program (core) index along the given axis.
// Lowers to nl.program_id(axis).
def NKI_ProgramIdOp : NKI_Op<"program_id"> {
  let arguments = (ins I32Attr:$axis);
  let results = (outs Index:$result);
  let assemblyFormat = [{ `(` $axis `)` attr-dict }];
}

// Copies src into dst via DMA (both must be memrefs).
// Lowers to nisa.dma_copy(dst=dst, src=src).
def NKI_DmaCopyOp : NKI_Op<"dma_copy"> {
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = [{
    $src `->` $dst `:` type($src) `,` type($dst) attr-dict
  }];
}

// Inter-core send+receive. Both cores in the grid must call this.
// Lowers to nisa.sendrecv(...).
def NKI_SendRecvOp : NKI_Op<"sendrecv"> {
  let arguments = (ins
    AnyMemRef:$send_buf,
    AnyMemRef:$recv_buf,
    Index:$send_to_rank,
    Index:$recv_from_rank,
    I32Attr:$pipe_id
  );
  let assemblyFormat = [{
    $send_buf `->` $recv_buf
    `send_to` `(` $send_to_rank `)`
    `recv_from` `(` $recv_from_rank `)`
    `pipe` `(` $pipe_id `)`
    `:` type($send_buf) `,` type($recv_buf) attr-dict
  }];
}

// Elementwise op over full memref tiles. kind encodes the arith op:
// 0=mul, 1=add, 2=sub. Lowers to nl.multiply / nl.add / nl.subtract.
def NKI_ElementwiseOp : NKI_Op<"elementwise"> {
  let arguments = (ins
    AnyMemRef:$lhs,
    AnyType:$rhs,
    AnyMemRef:$output,
    I32Attr:$kind
  );
  let assemblyFormat = [{
    `kind` `(` $kind `)` $lhs `,` $rhs `->` $output
    `:` type($lhs) `,` type($rhs) `->` type($output) attr-dict
  }];
}

def NKI_LaunchOp : NKI_Op<"launch", [SingleBlock, NoTerminator]> {
  let results = (outs);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $body attr-dict
  }];
}

#endif
