// Transforming the matrix multiplications into IR with the `accel` dialect:

// No reuse (Nothing Stationary) with accumulation on CPU:
// /working_dir/builds/llvm-project/build-x86/bin/mlir-opt example0.mlir -test-generic-to-accel="anchor-op=linalg.matmul loop-permutation=0,1,2 opcode-map=\"opcode_map<sendA=[op_send_literal(1),op_send(0)], sendB=[op_send_literal(2),op_send(1)],compute = [op_send_literal(4)], recvC=[op_send_literal(8),op_recv(2)]>\" opcode-flow=\"(sendA sendB compute recvC)\"  acc-on-cpu=2 accel-tile-size=4 anchor-filter=ACC_V3" --cse -o example0.NS.accel.mlir

// Reusing C (C Stationary), no accumulation on CPU necessary:
// /working_dir/builds/llvm-project/build-x86/bin/mlir-opt example0.mlir -test-generic-to-accel="anchor-op=linalg.matmul loop-permutation=0,1,2 opcode-map=\"opcode_map<sendA=[op_send_literal(1),op_send(0)], sendB=[op_send_literal(2),op_send(1)],compute = [op_send_literal(4)], recvC=[op_send_literal(8),op_recv(2)]>\" opcode-flow=\"((sendA sendB compute)  recvC)\"  accel-tile-size=4 anchor-filter=ACC_V3" --cse -o example0.CS.accel.mlir

// Transforming `accel` operations into calls to our DMA driver library.
// /working_dir/builds/llvm-project/build-x86/bin/mlir-opt example0-NS.accel.mlir -test-accel-to-axi4mlir -o example0-NS.withcalls.mlir

func @matmul_m16_n8_k32_call(
  %A: memref<16x32xi32>, 
  %B: memref<32x8xi32>, 
  %C: memref<16x8xi32>) 
  attributes {llvm.emit_c_interface} 
{
  linalg.matmul {__accel_transform__="ACC_V3"}
   ins(%A, %B: memref<16x32xi32>, memref<32x8xi32>)
   outs(%C: memref<16x8xi32>)
  return
}
