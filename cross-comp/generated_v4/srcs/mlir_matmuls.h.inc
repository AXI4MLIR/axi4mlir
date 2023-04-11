
// M=512 N=1024 K=64
extern "C" void matmul_m512_n1024_k64_ACC_v4_As_64_64_64_0_call(
  int *allocated_ptr0, int *aligned_ptr0,
  int64_t offset0, int64_t size0_d0, int64_t size0_d1,
  int64_t stride0_d0, int64_t stride0_d1,
  // Second Memref (%arg1)
  int *allocated_ptr1, int *aligned_ptr1,
  int64_t offset1, int64_t size1_d0, int64_t size1_d1,
  int64_t stride1_d0, int64_t stride1_d1,
  int *allocated_ptr2, int *aligned_ptr2,
  int64_t offset2, int64_t size2_d0, int64_t size2_d1,
  int64_t stride2_d0, int64_t stride2_d1);

extern "C" void _mlir_ciface_matmul_m512_n1024_k64_ACC_v4_As_64_64_64_0_call(
  memref_2d_descriptor *arg0,
  memref_2d_descriptor *arg1,
  memref_2d_descriptor *arg2);


// M=64 N=512 K=512
extern "C" void matmul_m64_n512_k512_ACC_v4_Cs_64_64_64_1_call(
  int *allocated_ptr0, int *aligned_ptr0,
  int64_t offset0, int64_t size0_d0, int64_t size0_d1,
  int64_t stride0_d0, int64_t stride0_d1,
  // Second Memref (%arg1)
  int *allocated_ptr1, int *aligned_ptr1,
  int64_t offset1, int64_t size1_d0, int64_t size1_d1,
  int64_t stride1_d0, int64_t stride1_d1,
  int *allocated_ptr2, int *aligned_ptr2,
  int64_t offset2, int64_t size2_d0, int64_t size2_d1,
  int64_t stride2_d0, int64_t stride2_d1);

extern "C" void _mlir_ciface_matmul_m64_n512_k512_ACC_v4_Cs_64_64_64_1_call(
  memref_2d_descriptor *arg0,
  memref_2d_descriptor *arg1,
  memref_2d_descriptor *arg2);


// M=512 N=256 K=16
extern "C" void matmul_m512_n256_k16_ACC_v4_Bs_16_256_16_2_call(
  int *allocated_ptr0, int *aligned_ptr0,
  int64_t offset0, int64_t size0_d0, int64_t size0_d1,
  int64_t stride0_d0, int64_t stride0_d1,
  // Second Memref (%arg1)
  int *allocated_ptr1, int *aligned_ptr1,
  int64_t offset1, int64_t size1_d0, int64_t size1_d1,
  int64_t stride1_d0, int64_t stride1_d1,
  int *allocated_ptr2, int *aligned_ptr2,
  int64_t offset2, int64_t size2_d0, int64_t size2_d1,
  int64_t stride2_d0, int64_t stride2_d1);

extern "C" void _mlir_ciface_matmul_m512_n256_k16_ACC_v4_Bs_16_256_16_2_call(
  memref_2d_descriptor *arg0,
  memref_2d_descriptor *arg1,
  memref_2d_descriptor *arg2);


// M=32 N=256 K=1024
extern "C" void matmul_m32_n256_k1024_ACC_v4_Cs_32_128_32_3_call(
  int *allocated_ptr0, int *aligned_ptr0,
  int64_t offset0, int64_t size0_d0, int64_t size0_d1,
  int64_t stride0_d0, int64_t stride0_d1,
  // Second Memref (%arg1)
  int *allocated_ptr1, int *aligned_ptr1,
  int64_t offset1, int64_t size1_d0, int64_t size1_d1,
  int64_t stride1_d0, int64_t stride1_d1,
  int *allocated_ptr2, int *aligned_ptr2,
  int64_t offset2, int64_t size2_d0, int64_t size2_d1,
  int64_t stride2_d0, int64_t stride2_d1);

extern "C" void _mlir_ciface_matmul_m32_n256_k1024_ACC_v4_Cs_32_128_32_3_call(
  memref_2d_descriptor *arg0,
  memref_2d_descriptor *arg1,
  memref_2d_descriptor *arg2);
