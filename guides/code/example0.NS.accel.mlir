#map0 = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func @matmul_m16_n8_k32_call(%arg0: memref<16x32xi32>, %arg1: memref<32x8xi32>, %arg2: memref<16x8xi32>) attributes {llvm.emit_c_interface} {
    %c1077936128_i32 = arith.constant 1077936128 : i32
    %c369098752_i32 = arith.constant 369098752 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c373293056_i32 = arith.constant 373293056 : i32
    accel.init_dma %c1077936128_i32, %c369098752_i32, %c65536_i32, %c373293056_i32, %c65536_i32 : (i32, i32, i32, i32, i32)
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c16 step %c4 {
      scf.for %arg4 = %c0 to %c8 step %c4 {
        scf.for %arg5 = %c0 to %c32 step %c4 {
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %0 = accel.sendLiteral %c1_i32, %c0_i32 : (i32, i32) -> i32
          %1 = memref.subview %arg0[%arg3, %arg5] [%c4, %c4] [1, 1] : memref<16x32xi32> to memref<?x?xi32, #map0>
          %2 = accel.send %1, %0 : (memref<?x?xi32, #map0>, i32) -> i32
          %c2_i32 = arith.constant 2 : i32
          %3 = accel.sendLiteral %c2_i32, %c0_i32 : (i32, i32) -> i32
          %4 = memref.subview %arg1[%arg5, %arg4] [%c4, %c4] [1, 1] : memref<32x8xi32> to memref<?x?xi32, #map1>
          %5 = accel.send %4, %3 : (memref<?x?xi32, #map1>, i32) -> i32
          %c4_i32 = arith.constant 4 : i32
          %6 = accel.sendLiteral %c4_i32, %c0_i32 : (i32, i32) -> i32
          %c8_i32 = arith.constant 8 : i32
          %7 = accel.sendLiteral %c8_i32, %c0_i32 : (i32, i32) -> i32
          %8 = memref.subview %arg2[%arg3, %arg4] [%c4, %c4] [1, 1] : memref<16x8xi32> to memref<?x?xi32, #map1>
          %9 = memref.alloca() : memref<4x4xi32>
          %10 = accel.recv %9, %7 : (memref<4x4xi32>, i32) -> i32
          linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%9 : memref<4x4xi32>) outs(%8 : memref<?x?xi32, #map1>) {
          ^bb0(%arg6: i32, %arg7: i32):
            %11 = arith.addi %arg6, %arg7 : i32
            linalg.yield %11 : i32
          }
        }
      }
    }
    return
  }
}

