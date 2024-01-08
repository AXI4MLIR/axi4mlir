#map0 = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func private @copy_from_outbuffer_i32(memref<*xi32>, i32) -> i32
  func private @dma_start_recv(i32, i32) -> i32
  func private @dma_wait_recv()
  func private @copy_to_inbuffer_i32(memref<*xi32>, i32) -> i32
  func private @dma_start_send(i32, i32) -> i32
  func private @dma_wait_send()
  func private @dma_init(i32, i32, i32, i32, i32)
  func private @dma_free()
  func @matmul_m16_n8_k32_call(%arg0: memref<16x32xi32>, %arg1: memref<32x8xi32>, %arg2: memref<16x8xi32>) attributes {llvm.emit_c_interface} {
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c373293056_i32 = arith.constant 373293056 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c369098752_i32 = arith.constant 369098752 : i32
    %c1077936128_i32 = arith.constant 1077936128 : i32
    call @dma_init(%c1077936128_i32, %c369098752_i32, %c65536_i32, %c373293056_i32, %c65536_i32) : (i32, i32, i32, i32, i32) -> ()
    scf.for %arg3 = %c0 to %c16 step %c4 {
      scf.for %arg4 = %c0 to %c8 step %c4 {
        scf.for %arg5 = %c0 to %c32 step %c4 {
          %0 = memref.alloc() : memref<i32>
          memref.store %c1_i32, %0[] : memref<i32>
          %1 = memref.cast %0 : memref<i32> to memref<*xi32>
          %2 = call @copy_to_inbuffer_i32(%1, %c0_i32) : (memref<*xi32>, i32) -> i32
          %3 = call @dma_start_send(%c1_i32, %c0_i32) : (i32, i32) -> i32
          call @dma_wait_send() : () -> ()
          memref.dealloc %0 : memref<i32>
          %4 = memref.subview %arg0[%arg3, %arg5] [4, 4] [1, 1] : memref<16x32xi32> to memref<4x4xi32, #map0>
          %5 = memref.cast %4 : memref<4x4xi32, #map0> to memref<*xi32>
          %6 = call @copy_to_inbuffer_i32(%5, %c1_i32) : (memref<*xi32>, i32) -> i32
          %7 = call @dma_start_send(%c16_i32, %c1_i32) : (i32, i32) -> i32
          call @dma_wait_send() : () -> ()
          %8 = memref.alloc() : memref<i32>
          memref.store %c2_i32, %8[] : memref<i32>
          %9 = memref.cast %8 : memref<i32> to memref<*xi32>
          %10 = call @copy_to_inbuffer_i32(%9, %c0_i32) : (memref<*xi32>, i32) -> i32
          %11 = call @dma_start_send(%c1_i32, %c0_i32) : (i32, i32) -> i32
          call @dma_wait_send() : () -> ()
          memref.dealloc %8 : memref<i32>
          %12 = memref.subview %arg1[%arg5, %arg4] [4, 4] [1, 1] : memref<32x8xi32> to memref<4x4xi32, #map1>
          %13 = memref.cast %12 : memref<4x4xi32, #map1> to memref<*xi32>
          %14 = call @copy_to_inbuffer_i32(%13, %c1_i32) : (memref<*xi32>, i32) -> i32
          %15 = call @dma_start_send(%c16_i32, %c1_i32) : (i32, i32) -> i32
          call @dma_wait_send() : () -> ()
          %16 = memref.alloc() : memref<i32>
          memref.store %c4_i32, %16[] : memref<i32>
          %17 = memref.cast %16 : memref<i32> to memref<*xi32>
          %18 = call @copy_to_inbuffer_i32(%17, %c0_i32) : (memref<*xi32>, i32) -> i32
          %19 = call @dma_start_send(%c1_i32, %c0_i32) : (i32, i32) -> i32
          call @dma_wait_send() : () -> ()
          memref.dealloc %16 : memref<i32>
          %20 = memref.alloc() : memref<i32>
          memref.store %c8_i32, %20[] : memref<i32>
          %21 = memref.cast %20 : memref<i32> to memref<*xi32>
          %22 = call @copy_to_inbuffer_i32(%21, %c0_i32) : (memref<*xi32>, i32) -> i32
          %23 = call @dma_start_send(%c1_i32, %c0_i32) : (i32, i32) -> i32
          call @dma_wait_send() : () -> ()
          memref.dealloc %20 : memref<i32>
          %24 = memref.subview %arg2[%arg3, %arg4] [4, 4] [1, 1] : memref<16x8xi32> to memref<4x4xi32, #map1>
          %25 = memref.alloca() : memref<4x4xi32>
          %26 = memref.cast %25 : memref<4x4xi32> to memref<*xi32>
          %27 = call @dma_start_recv(%c16_i32, %c1_i32) : (i32, i32) -> i32
          call @dma_wait_recv() : () -> ()
          %28 = call @copy_from_outbuffer_i32(%26, %c1_i32) : (memref<*xi32>, i32) -> i32
          linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%25 : memref<4x4xi32>) outs(%24 : memref<4x4xi32, #map1>) {
          ^bb0(%arg6: i32, %arg7: i32):
            %29 = arith.addi %arg6, %arg7 : i32
            linalg.yield %29 : i32
          }
        }
      }
    }
    call @dma_free() : () -> ()
    return
  }
}

