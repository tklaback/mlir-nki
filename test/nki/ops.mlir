// Test nki.load
func.func @test_load(%arg0: memref<32x16xf16>) {
  %0 = nki.load %arg0 : memref<32x16xf16> -> tensor<32x16xf16>
  return
}

// Test nki.store
func.func @test_store(%arg0: tensor<32x16xf16>, %arg1: memref<32x16xf16>) {
  nki.store %arg0, %arg1 : tensor<32x16xf16>, memref<32x16xf16>
  return
}