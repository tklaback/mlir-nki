module {
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @single_put_get(%a0: memref<32x16xi32>, %a1: memref<32x16xi32>) {
    %c2_0 = arith.constant 2 : index
    air.launch (%arg2, %arg3) in (%arg4=%c2_0, %arg5=%c2_0) args(%arg0=%a0, %arg1=%a1) : memref<32x16xi32>, memref<32x16xi32> {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = air.channel.put async  @channel_0[%c0, %c0] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c1]) {id = 1 : i32} : (memref<32x16xi32>)
      %1 = air.channel.get async  @channel_1[%c0, %c0] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c1]) {id = 2 : i32} : (memref<32x16xi32>)
      air.segment @segment_0 {
        %c1_0 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) {
          %c0_4 = arith.constant 0 : index
          %c1_4 = arith.constant 1 : index
          %c32_5 = arith.constant 32 : index
          %c16_6 = arith.constant 16 : index
          %c8_7 = arith.constant 8 : index
          %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
          %alloc_8 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
          air.channel.get  @channel_0[%arg10, %arg11] (%alloc[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c1_4]) {id = 3 : i32} : (memref<16x8xi32, 2>)
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 16 {
              %2 = affine.load %alloc[%arg19, %arg18] : memref<16x8xi32, 2>
              affine.store %2, %alloc_8[%arg19, %arg18] : memref<16x8xi32, 2>
            }
          }
          air.channel.put  @channel_1[%arg10, %arg11] (%alloc_8[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c1_4]) {id = 4 : i32} : (memref<16x8xi32, 2>)
          memref.dealloc %alloc_8 : memref<16x8xi32, 2>
          memref.dealloc %alloc : memref<16x8xi32, 2>
        }
      }
    }
    return
  }
}