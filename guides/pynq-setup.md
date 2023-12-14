# Accelerator Setup on PYNQ board

This guide will show how to prepare the accelerator in Vivado to be used on the
PYNQ board through AXI4MLIR. This flow was tested with Vivado and Vivado HLS
v2019.2, which still provide support of SystemC-based HLS. But we believe a
similar approach can be achieved with newer versions of Vitis and Vitis HLS
using plain C++ code with pragmas.

# Prerequisites

- [Vivado HLS or Vitis HLS](https://www.xilinx.com/products/design-tools/vivado/integration/esl-design.html)
- [Vivado or Vitis](https://www.xilinx.com/products/design-tools/vivado.html)
- [PYNQ-Z2 board](http://www.pynq.io/board.html)

# Steps

1. Follow the steps to design the accelerator in VitisHLS.
  - The accelerator must include an AXI4-Stream interface, thus it requires the following pragma directive in the C++ code similar to [MatMulv3 accelerator](https://github.com/AXI4MLIR/llvm-project/blob/f2172866a5c47516dd4f4b823c7a75c96821f6f7/mlir/include/mlir/ExecutionEngine/axi/accelerators/mm_4x4_v3/accelerator.sc.h#L116):
    ```c++
    // With SystemC (Vivado HLS)
    #pragma HLS RESOURCE variable=din1 core=AXI4Stream metadata="-bus_bundle S_AXIS_DATA1" port_map={{din1_0 TDATA} {din1_1 TLAST}}
    #pragma HLS RESOURCE variable=dout1 core=AXI4Stream metadata="-bus_bundle M_AXIS_DATA1" port_map={{dout1_0 TDATA} {dout1_1 TLAST}}
    #pragma HLS RESET variable=reset

    // In plain C++ (Vitis HLS)
    #pragma HLS INTERFACE mode=axis port=A
    #pragma HLS INTERFACE mode=axis port=B
    #pragma HLS RESET variable=reset
    ```
2. Export the accelerator as an IP.
3. Create a new Vitis project.
4. Add the IP to the project.
5. Connect the IP to the system following in a similar manner to the following:
  ![IP Integrator Diagram](res/ip-integrator-diagram.png)
6. Setup the address map for the IP.
  - The address map can be found in the Address Editor tab in the Block Design.
  - The address map should be similar to the following:
  ![Address Map](res/address-map.png)
  - **Take note of the selected DMA addresses**. These are going to be used in AXI4MLIR
6. Generate the bitstream.
8. Copy bitstream and [bitstream loader script](../experiments/ex1/ex1_pynq/load_bitstream.py) to the PYNQ board.

# Resources on Accelerator development and deployment in Vitis

* [More information on AXI-Stream interfaces in Vitis](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-stream)
* [Creating a Design with Vitis HLS IP](https://docs.xilinx.com/r/en-US/ug994-vivado-ip-subsystems/Creating-a-Design-with-Vitis-HLS-IP)
* [Getting Started with Vivado IP Integrator](https://docs.xilinx.com/r/en-US/ug994-vivado-ip-subsystems/Navigating-Content-by-Design-Process)