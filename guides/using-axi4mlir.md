# Using AXI4MLIR Extensions

This guide demonstrates how to use AXI4MLIR extensions to generate code for a matrix
multiplication linear algebra operation. The matrix multiplication in `mlir` looks like this:

```mlir
func @matmul_call(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {
  linalg.matmul
   ins(%A, %B: memref<16x8xi32>, memref<8x32xi32>)
   outs(%C: memref<16x32xi32>)
  return
```

Or, using its `linalg.generic`` form:

```mlir
matmul_trait = {
  // Original generic information
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>, // A
    affine_map<(m, n, k) -> (k, n)>, // B
    affine_map<(m, n, k) -> (m, n)>  // C
  ]
}

func @matmul_call(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  linalg.generic #matmul_trait
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>) {
    ^bb0(%a: i32, %b: i32, %c: i32):
      %0 = arith.muli %a, %b : i32
      %1 = arith.addi %c, %0 : i32
      linalg.yield %1 : i32
  }
  return
}
```

Which may take the following form during AXI4MLIR transformations:

```mlir
matmul_trait = {
  __internal_linalg_transform__="ACCELERATE",
  accel_dmaAddress = 41,
  accel_dmaInputBufferSize = 42,
  accel_dmaOutputAddress = 43,
  accel_loop_permutation = [0,2,1], // Changes loops (m, n, k) to (m, k, n)
  accel_accel_tile_size = 4,
  accel_acc_on_cpu = 2, // Accumulates received data on data structure 2: C
  accel_opcode_map_str = "opcode_map<s0=[op_send(0)], s1=[op_send(1)], s2=[op_send(2)], r2=[op_recv(2)], s0s1s2r2=[op_send(0), op_send(1), op_send(2), op_recv(2)]>",
  accel_init_flow_str = "reset",
  accel_opcode_flow_str = "(s0 (s1 r2))",

  // Original generic information
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>, // A
    affine_map<(m, n, k) -> (k, n)>, // B
    affine_map<(m, n, k) -> (m, n)>  // C
  ]
}

func @matmul_call(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {

  linalg.generic #matmul_trait
    ins (%A, %B : memref<16x8xi32>, memref<8x32xi32>)
    outs(%C : memref<16x32xi32>) {
    ^bb0(%a: i32, %b: i32, %c: i32):
      %0 = arith.muli %a, %b : i32
      %1 = arith.addi %c, %0 : i32
      linalg.yield %1 : i32
  }
  return
}
```

## Accelerator description

In this example, we use a matrix multiplication accelerator that takes two 4x4
matrices as input and produces a 4x4 matrix as output. The accelerator is
implemented in SystemC and exported as an IP to be used in Vivado or Vitis. See
[pynq-setup.md](../guides/pynq-setup.md) for more information on setting up
the accelerator.

The implementation for our accelerator is shown
[here](https://github.com/AXI4MLIR/llvm-project/blob/16eaaaeedbeda17a77387edbb1da96ce8e95f15e/mlir/include/mlir/ExecutionEngine/axi/accelerators/mm_4x4_v3/accelerator.sc.h).

The accelerator has support for different instructions/opcodes. The opcode
values and their meaning are defined
[here](https://github.com/AXI4MLIR/llvm-project/blob/16eaaaeedbeda17a77387edbb1da96ce8e95f15e/mlir/include/mlir/ExecutionEngine/axi/accelerators/mm_4x4_v3/mm_4x4_v3.json#L83C3-L83C3).
Any opcode or data sent to the accelerator, i.e., received by the accelerator,
goes through DMA fifo `0`, using the AXI4-Stream interface. Any data sent by the
accelerator, i.e., received by the CPU, goes through DMA fifo `1`, using the
AXI4-Stream interface. The JSON helper file has keywords from the perspective of
the accelerator, i.e., during `READ`, the accelerator is reading data from the
DMA stream, and during `SEND`, the accelerator is sending data to the DMA
stream.

For example, when the accelerator reads opcode `1` from the DMA stream, it will
consider the subsequent 16 values as the (tile of) matrix A and store them in a
[special buffer](https://github.com/AXI4MLIR/llvm-project/blob/16eaaaeedbeda17a77387edbb1da96ce8e95f15e/mlir/include/mlir/ExecutionEngine/axi/accelerators/mm_4x4_v3/accelerator.sc.h#L168)
inside the accelerator. We call this buffer: `#A_Buffer`. 

```json
"1": [
    {
        "READ": {
            "dma_fifo_id": 0,
            "buffer": "#A_Buffer",
            "offset": 0,
            "length": 16
        }
    }
],
```

Then, after completing this instruction, the accelerator will wait for the next
opcode to be read from the DMA input stream.

Similarly, when the accelerator reads the opcode `2` from the DMA stream,
it will expect the contents of the 16 values of matrix B to follow. Once
received, the accelerator will store the contents of the matrix B in the
`#B_Buffer` for later use.

```json
"2": [
    {
        "READ": {
            "dma_fifo_id": 0,
            "buffer": "#B_Buffer",
            "offset": 0,
            "length": 16
        }
    }
],
```

This accelerator also has individual opcodes for computing the 4x4 matrix
multiplication, and writing (with a `SEND` opcode) data from the `#C_buffer`
into the DMA stream.

```json
"4": [
    {
        "COMPUTE": {
            "kernel_id": 0
        }
    }
],
"8": [
    {
        "SEND": {
            "dma_fifo_id": 1,
            "buffer": "#C_Buffer",
            "offset": 0,
            "length": 16
        }
    }
],
```

## Accelerator in AXI4MLIR

With the opcodes above, the accelerator can compute the matrix
multiplication of two 4x4 matrices, and send the result 4x4 matrix back to the
CPU.

With AXI4MLIR extensions, the `opcode_map` attribute can be used to specify the
relevant opcodes. Note that we now take a CPU perspective, so `op_send` is used
to send data to the accelerator, and `op_recv` is used to receive data from the
accelerator.

```mlir
#my_opcodes = opcode_map< 
  sendA = [op_send_literal(1), op_send(0)],
  sendB = [op_send_literal(2), op_send(1)],
  compute = [op_send_literal(4)],
  recvC = [op_send_literal(8), op_recv(2)],
>
```

With these available opcodes, a valid `opcode_flow` to perform the matrix multiplication is:

```mlir
#my_opcode_flow = (sendA sendB compute recvC)
```

This will generate all the necessary data transfers inside the innermost loop of the matrix multiplication. The `op_send_literal` opcodes are used to send the literal values `1`, `2`, `4`, and `8` to the accelerator. The `op_send` and `op_recv` opcodes are used to send the data from the `A`, `B`, and `C` buffers, which are in the positions `0`, `1`, and `2` of a linear algebra operation we want to accelerate.

This opcode and flow translates to the following:

```mlir
// TODO: add MLIR code after the transformation into the ACCEL dialect
for (int i = 0; i < 4; i++) {
  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      sendA(i, k, A[i][k]);
      sendB(k, j, B[k][j]);
      compute(i, j, k);
      recvC(i, j, C[i][j]);
    }
  }
}
```

However, with the same opcodes map, we can use a different `opcode_flow` to perform a output stationary matrix multiplication:

```mlir
#my_opcode_flow = opcode_flow<((sendA sendB compute)  recvC)>
```

This opcode and flow translates to the following:

```mlir
// TODO: add MLIR code after the transformation into the ACCEL dialect
for (int i = 0; i < 4; i++) {
  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      sendA(i, k, A[i][k]);
      sendB(k, j, B[k][j]);
      compute(i, j, k);
    }
    recvC(i, j, C[i][j]);
  }
}
```

### Additional supported opcodes

The accelerator implemented [here](https://github.com/AXI4MLIR/llvm-project/blob/16eaaaeedbeda17a77387edbb1da96ce8e95f15e/mlir/include/mlir/ExecutionEngine/axi/accelerators/mm_4x4_v3/accelerator.sc.h) with opcodes described [here](https://github.com/AXI4MLIR/llvm-project/blob/16eaaaeedbeda17a77387edbb1da96ce8e95f15e/mlir/include/mlir/ExecutionEngine/axi/accelerators/mm_4x4_v3/mm_4x4_v3.json#L83C3-L83C3)
has support for additional opcodes and flows. The following opcode map shows the complete set of opcodes:

```mlir
#my_opcodes = opcode_map< 
    sendA = [op_send_literal(1), op_send(0)],
    sendB = [op_send_literal(2), op_send(1)],
    sendAB = [op_send_literal(3), op_send(0), op_send(1)],
    compute = [op_send_literal(4)],
    sendA_compute = [op_send_literal(5), op_send(0),],
    sendB_compute = [op_send_literal(6), op_send(1),],
    sendAB_compute = [op_send_literal(7), op_send(0), op_send(1)],
    recvC = [op_send_literal(8), op_recv(2)],
    sendA_recvC = [op_send_literal(9), op_send(0), op_recv(2)],
    sendB_recvC = [op_send_literal(10), op_send(1), op_recv(2)],
    sendAB_recvC = [op_send_literal(11), op_send(0), op_send(1), op_recv(2)],
    compute_recvC = [op_send_literal(12), op_recv(2)],
    sendA_compute_recvC = [op_send_literal(13), op_send(0), op_recv(2)],
    sendB_compute_recvC = [op_send_literal(14), op_send(1), op_recv(2)],
    sendAB_compute_recvC = [op_send_literal(15), op_send(0), op_send(1), op_recv(2)],
>
```

These opcodes allow the accelerator to support different data reuse patterns,
such as keeping nothing in memory or keeping one of the matrices A, B, or C in
memory. The following opcode flows are supported:

```mlir
#nothing_stationary = opcode_flow<(sendA sendB compute recvC)>

#A_stationary_option0 = opcode_flow<(sendA (sendB compute recvC))>
#A_stationary_option1 = opcode_flow<(sendA (sendB_compute recvC))>
#A_stationary_option2 = opcode_flow<(sendA (sendB_compute_recvC))>

#B_stationary_option0 = opcode_flow<(sendB (sendA compute recvC))>
#B_stationary_option1 = opcode_flow<(sendB (sendA_compute recvC))>
#B_stationary_option2 = opcode_flow<(sendB (sendA_compute_recvC))>

#C_stationary_option0 = opcode_flow<((sendA sendB compute)  recvC)>
#C_stationary_option1 = opcode_flow<((sendAB compute)  recvC)>
#C_stationary_option2 = opcode_flow<((sendAB_compute)  recvC)>
```

Note that `*_option_0` flows require more opcodes than `*_option_1` and
`*_option_2` flows. This is because the `*_option_0` flows send one instruction
before each data transfer to trigger the computation, while the `*_option_2`,
such as `A_stationary_option2`, sends one instruction before sending the data
associated with the A tile that is kept in memory for several iterations,
then, in the innermost loop, sends another instruction to tell the accelerator
to expect the B tile, to start the computation, and, once the computation is
done, to send the result back to the CPU.

### Additional remarks

#### Loop permutation

To enable a particular opcode flow, the user has to specify a correct
permutation of the loops in the linalg operation.  This is achieved with the
`loop_permutation` attribute. For example, to enable the `A_stationary_option0`
flow, given the `indexing_maps` attribute of a `linalg.generic` operation:

```mlir
indexing_maps = [
  affine_map<(m, n, k) -> (m, k)>, // A
  affine_map<(m, n, k) -> (k, n)>, // B
  affine_map<(m, n, k) -> (m, n)>  // C
]
```

The following loop permutation is required:

```mlir
// Changes (m, n, k) to (m, k, n)
accel_loop_permutation = [0,2,1] 
```

#### Accumulation on CPU

Depending on the data reuse pattern, the user may want to accumulate the results
on the CPU. This can be achieved with the `acc_on_cpu` attribute. For example,
to enable the `A_stationary_option0` flow, `acc_on_cpu` should be set to `2`,
indicating that the results coming from the accelerator should be accumulated on
the (tile of) data structure `2` of the linalg operation. The following option
will trigger host code generation that accumulates the results on the CPU. 

```mlir
// Attribute in the linalg.generic operation
accel_acc_on_cpu = 2
```

## How to activate the transformations?

AXI4MLIR transformations can be controlled by two passes:

- `test-generic-to-accel`: This pass transforms a generic linalg operation into an accelerator operation. It takes the following arguments:
  - `anchor-op`: The name of the linalg operation to be transformed.
  - `loop-permutation`: The permutation of the loops in the linalg operation.
  - `opcode-map`: The opcode map to be used.
  - `opcode-flow`: The opcode flow to be used.
  - `accel-tile-size`: The tile size to be used in the accelerator.
  - `acc-on-cpu`: Whether the results should be accumulated on the CPU.
- `test-accel-to-axi4mlir`: This pass transforms the `accel` operations into calls to our DMA library. It takes no arguments.

### Using the commandline passes

For a nothing stationary matrix multiplication, the following command can be used:

```bash
mlir-opt \
  -test-generic-to-accel="anchor-op=linalg.matmul loop-permutation=0,1,2 opcode-map=\"opcode_map<sendA=[op_send_literal(1),op_send(0)],sendB=[op_send_literal(2),op_send(1)],compute=[op_send_literal(4)],recvC=[op_send_literal(8),op_recv(2)],>\" opcode-flow=\"(sendA sendB compute recvC)\" acc-on-cpu=2 accel-tile-size=4 \
  linalg.mlir \
  -o accel.mlir
```

For an A stationary matrix multiplication, the following command can be used:

```bash
mlir-opt \
  -test-generic-to-accel="anchor-op=linalg.matmul loop-permutation=0,2,1 opcode-map=\"opcode_map<sendA=[op_send_literal(1),op_send(0)],sendB_compute_recvC=[op_send_literal(14),op_send(1),op_recv(2)],>\" opcode-flow=\"(sendA (sendB_compute_recvC))\" accel-tile-size=4 acc-on-cpu=2 \
  linalg.mlir \
  -o accel.mlir
```

To transform `accel`` operations into calls to our DMA library, use:

```bash
mlir-opt \
  -cse -test-accel-to-axi4mlir \
  accel.mlir \
  -o axi_calls.mlir
```

Additional lowering passes to transform the generated code into LLVM dialect:

```bash
# Updated as of llvm-project version 12
mlir-opt \
  -convert-linalg-to-loops -lower-affine \
  --buffer-loop-hoisting --buffer-deallocation \
  -convert-scf-to-cf  \
  -arith-expand \
  -memref-expand \
  -convert-vector-to-llvm \
  -convert-memref-to-llvm="index-bitwidth=32" \
  -convert-scf-to-cf  \
  -convert-arith-to-llvm="index-bitwidth=32" \
  -convert-std-to-llvm="index-bitwidth=32" \
  -canonicalize \
  -reconcile-unrealized-casts \
  axi_calls.mlir \
  -o output_llvm.mlir
```

Now we have to generate the LLVM IR:

```bash
mlir-translate \
  -mlir-to-llvmir \
  output_llvm.mlir \
  -o output_llvm.ll
```

Finally, we can generate the object file, library, and binary. 

```bash
# Create an object file for "library"
$PROJ_ROOT/builds/llvm-project/build-x86/bin/clang++ \
  --target=arm-linux-gnueabihf -march=armv7-a -marm -mfloat-abi=hard \
  -mfpu=neon -funsafe-math-optimizations -ftree-vectorize \
  -c -o mymlirlib.o \
  output_llvm.ll

# Create a shared library for later linking
# Add any additional libraries to the linker command
$PROJ_ROOT/builds/llvm-project/build-x86/bin/clang++ -shared \
  -o mymlirlib.so \
  mymlirlib.o \
  --target=arm-linux-gnueabihf \
  -Wl,-rpath=$PROJ_ROOT/builds/llvm-project/build-runner-arm/lib \
  -L$PROJ_ROOT/builds/llvm-project/build-runner-arm/lib \
  -lmlir_runner_utils -lmlir_axi_runner_utils

# Create a binary `app`, linking the shared library
$PROJ_ROOT/builds/llvm-project/build-x86/bin/clang++ -o app \
    srcs/matmul_driver_v3.cc \
    -Isrcs \
    -I$PROJ_ROOT/llvm-project/mlir/include \
    --target=arm-linux-gnueabihf -march=armv7-a -marm -mfloat-abi=hard \
    -mfpu=neon -funsafe-math-optimizations -ftree-vectorize \
    -Wl,--copy-dt-needed-entries \
    -Wl,-rpath=$PROJ_ROOT/builds/llvm-project/build-runner-arm/lib \
    -L$PROJ_ROOT/builds/llvm-project/build-runner-arm/lib \
    -lmlir_runner_utils -lmlir_c_runner_utils -lmlir_axi_runner_utils \
```

For more details on additional compilation steps, see scripts:

- [For pynq deployment](../experiments/ex1/compile_pynq.sh)
- [For SystemC simulation](../experiments/ex1/compile_sysc.sh)
- [Example of C++ driver](../experiments/ex1/srcs/matmul_driver_v3.cc)
  - If `mlir` is compiled as a shared library as above.