# MLIR NKI Launch Lowering Notes

A launch should be flattened with a herd to create the dimensions of the SPMD grid launch.

**Example:** `launch 4x4` + `herd 2x2` → `nki.launch [8, 8]`

---

## Multiple Herds

If there is more than one herd, each should be lowered to a separate kernel with its own launch configuration.

**Example:**

```
launch 4x4
  herd 2x2
  herd 3x3
```

Would lower to:

```python
@nki_jit
def herd1_kernel(args):
    launch_i = nl.program_id(0) // 2  # 0-3
    launch_j = nl.program_id(1) // 2  # 0-3
    herd_x = nl.program_id(0) % 2     # 0-1
    herd_y = nl.program_id(1) % 2     # 0-1
    # Herd 1 body
    ...

@nki_jit
def herd2_kernel(args):
    launch_i = nl.program_id(0) // 3  # 0-3
    launch_j = nl.program_id(1) // 3  # 0-3
    herd_x = nl.program_id(0) % 3     # 0-2
    herd_y = nl.program_id(1) % 3     # 0-2
    # Herd 2 body
    ...

# Sequential execution
herd1_kernel[8, 8](args)    # 4×2, 4×2
herd2_kernel[12, 12](args)  # 4×3, 4×3
```

## Additional Notes

```sh
cmake .. \
  -DAIR_DIR=/home/ty/code/mlir-air/my_install/lib/cmake/air \
  -DAIR_BUILD_DIR=/home/ty/code/mlir-air/build \
  -DMLIR_DIR=/home/ty/code/mlir-air/my_install/mlir/lib/cmake/mlir \
  -DLLVM_DIR=/home/ty/code/mlir-air/my_install/mlir/lib/cmake/llvm \
  -G Ninja
```

`-DMLIR_DIR` and `-DLLVM_DIR` are required because `libAIRDialect.a` was built against LLVM 22 (bundled with mlir-air), while the system MLIR at `/usr/local` is LLVM 23. Pointing cmake at the mlir-air bundled install ensures all components share the same ABI.

Then, run `ninja` while in the build dir.

Then, to run a test from the build dir:

./tools/nki-opt --convert-air-to-nki (or whatever lowering(s)) ../test/nki/simple_add.mlir (or whichever test)