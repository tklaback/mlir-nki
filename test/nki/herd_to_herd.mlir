#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  air.channel @ChanIn []
  air.channel @ChanOut []
  air.channel @Herd2Herd []
  func.func @copy(%arg0: memref<16x32xi32>, %arg1: memref<16x32xi32>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<16x32xi32>, memref<16x32xi32> {
      air.channel.put  @ChanIn[] (%arg2[] [] []) : (memref<16x32xi32>)
      air.channel.get  @ChanOut[] (%arg3[] [] []) : (memref<16x32xi32>)
      air.segment @seg  {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        air.herd @producer_herd  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1_0) {
          %alloc = memref.alloc() : memref<16x32xi32, 2 : i32>
          %alloc_3 = memref.alloc() : memref<16x32xi32, 2 : i32>
          air.channel.get  @ChanIn[] (%alloc[] [] []) : (memref<16x32xi32, 2 : i32>)
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc : memref<16x32xi32, 2 : i32>, memref<16x32xi32, 2 : i32>) outs(%alloc_3 : memref<16x32xi32, 2 : i32>) {
          ^bb0(%in: i32, %in_4: i32, %out: i32):
            %0 = arith.muli %in, %in_4 : i32
            linalg.yield %0 : i32
          }
          air.channel.put  @Herd2Herd[] (%alloc_3[] [] []) : (memref<16x32xi32, 2 : i32>)
          memref.dealloc %alloc : memref<16x32xi32, 2 : i32>
          memref.dealloc %alloc_3 : memref<16x32xi32, 2 : i32>
        }
        %c1_1 = arith.constant 1 : index
        %c1_2 = arith.constant 1 : index
        air.herd @consumer_herd  tile (%arg4, %arg5) in (%arg6=%c1_1, %arg7=%c1_2) {
          %alloc = memref.alloc() : memref<16x32xi32, 2 : i32>
          %alloc_3 = memref.alloc() : memref<16x32xi32, 2 : i32>
          air.channel.get  @Herd2Herd[] (%alloc[] [] []) : (memref<16x32xi32, 2 : i32>)
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_4 = arith.constant 1 : index
          scf.for %arg8 = %c0 to %c16 step %c1_4 {
            %c0_5 = arith.constant 0 : index
            %c32 = arith.constant 32 : index
            %c1_6 = arith.constant 1 : index
            scf.for %arg9 = %c0_5 to %c32 step %c1_6 {
              %0 = memref.load %alloc[%arg8, %arg9] : memref<16x32xi32, 2 : i32>
              %c1_i32 = arith.constant 1 : i32
              %1 = arith.addi %0, %c1_i32 : i32
              memref.store %1, %alloc_3[%arg8, %arg9] : memref<16x32xi32, 2 : i32>
            }
          }
          air.channel.put  @ChanOut[] (%alloc_3[] [] []) : (memref<16x32xi32, 2 : i32>)
          memref.dealloc %alloc : memref<16x32xi32, 2 : i32>
          memref.dealloc %alloc_3 : memref<16x32xi32, 2 : i32>
        }
      }
    }
    return
  }
}