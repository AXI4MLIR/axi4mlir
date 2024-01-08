# AXI4MLIR

A custom MLIR dialect for generating host driver code for AXI-based linalg accelerators.

Artifact companion accepted at [CGO 2024](https://conf.researchr.org/home/cgo-2024) with the paper:

AXI4MLIR: User-Driven Automatic Host Code Generation for Custom AXI-Based Accelerators, 
N. B. Agostini, J. Haris, P. Gibson, M. Jayaweera, N. Rubin, A. Tumeo, J. Abellán, J. Cano, D. Kaeli. 
In CGO '24: Proceedings of the 2024 IEEE/ACM International Symposium on Code Generation and Optimization

# Artifact Evaluation

## Step 0 - Clone repo

```
git clone -b cgo2024_artifact https://github.com/AXI4MLIR/axi4mlir.git
cd axi4mlir
git submodule init
git submodule update --depth 1
```

## Step 1 - Build docker image


```
./build-docker.sh
```

## Step 2 - Compile required tools from within development container

```bash
./compile-mlir.sh
```

## Step 3 - Experiments

Now that you have all the required binaries built, you can reproduce experiments from the paper.
The experiments [readme](https://github.com/AXI4MLIR/axi4mlir/tree/cgo2024_artifact/experiments/README.md) and the paper's artifact appendix provide the details required to compile and run the experiments. 

---


## AXI4MLIR library
Here are some of the key files where AXI4MLIR is defined within the llvm-project:

[Accel Dialect](https://github.com/AXI4MLIR/llvm-project/blob/f2172866a5c47516dd4f4b823c7a75c96821f6f7/mlir/include/mlir/Dialect/Accel/IR/AccelOps.td)

[New Attributes](https://github.com/AXI4MLIR/llvm-project/blob/f2172866a5c47516dd4f4b823c7a75c96821f6f7/mlir/lib/Parser/OpcodeParser.cpp#L157)

[DMA Library](https://github.com/AXI4MLIR/llvm-project/blob/f2172866a5c47516dd4f4b823c7a75c96821f6f7/mlir/include/mlir/ExecutionEngine/AxiUtils.h)

[Linalg to Accel transformation pass](https://github.com/AXI4MLIR/llvm-project/blob/f2172866a5c47516dd4f4b823c7a75c96821f6f7/mlir/lib/Conversion/LinalgToAXI4MLIR/LinalgGenericToAccel.cpp)


[Accel to AXI4MLIR DMA library transformation pass](https://github.com/AXI4MLIR/llvm-project/blob/f2172866a5c47516dd4f4b823c7a75c96821f6f7/mlir/lib/Conversion/LinalgToAXI4MLIR/AccelToAXI4MLIR.cpp)

[Current Function Prototypes](https://github.com/AXI4MLIR/llvm-project/blob/axi4mlir/mlir/include/mlir/ExecutionEngine/axi/api_v1.h)

[Example](https://github.com/AXI4MLIR/llvm-project/blob/axi4mlir/mlir/test/axi4mlir-runner/run-matmul-v1accel.mlir)

[Available Options (end of the file)](https://github.com/AXI4MLIR/llvm-project/blob/axi4mlir/mlir/include/mlir/Conversion/Passes.td)

Tests [1](https://github.com/AXI4MLIR/llvm-project/blob/f2172866a5c47516dd4f4b823c7a75c96821f6f7/mlir/test/axi4mlir-runner/accelerators/matmul-v4-Cs.mlir) & [2](https://github.com/AXI4MLIR/llvm-project/blob/f2172866a5c47516dd4f4b823c7a75c96821f6f7/mlir/test/axi4mlir-runner/accel-to-runtime.mlir)

## Guides

Here are some relevant guides to get started with AXI4MLIR:

- [Enabling a new accelerator compatible with AXI4MLIR](guides/pynq-setup.md)
- [Create opcode maps and flows for custom accelerators](guides/using-axi4mlir.md)


---

# Building AXI4MLIR from source

## Dependencies:


Download the most recent version of cmake installer and ninja binary and add to
path:

```
https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu#comment1262462_829311
https://cmake.org/download/
https://github.com/ninja-build/ninja/releases
```

Install clang-10 clang++-10 and lld-10.

Download and install latest version of SystemC and export its path:

```
# Follow instructions at the end of the page to use this path
export SYSTEMC_HOME=/opt/systemc/systemc-2.3.3
```

Execute the following commands where you want to have a copy of the LLVM project:

## Instructions to download and build from main repo

To setup the project, use:

```shell
$ git clone https://github.com/AXI4MLIR/axi4mlir.git axi4mlir
$ cd axi4mlir
$ git submodule init
$ git submodule update
```

This should clone the `axi4mlir` branch of `llvm-project`.

A helper script is provided to build the llvm-project

```shell
./build_tools/build_llvm_dev_x86.sh \
    llvm-project \
    builds/llvm-project/build \
    builds/llvm-project/install

# Which generate build files in builds/llvm-project/build-x86
```

Add mlir binaries to path:

```
export PATH=$PATH:$(pwd)/builds/llvm-project/build-x86/bin
```

## Standalone AXI API execution

It is possible to compile the AXI API as a library to use in another project.

```shell
# In the LLVM_BUILD folder run
ninja mlir_axi_runner_utils 
```
This will generate 2 files:

* `<build_folder>/lib/libaxi_api_v1.so` - to link with any C project
* `<build_folder>/lib/libmlir_axi_runner_utils.so` - to be used by mlir-cpu-runner

## MLIR Example

An example of using the API from a MLIR file can be found
[here](https://github.com/AXI4MLIR/llvm-project/blob/axi4mlir/mlir/test/axi4mlir-runner/run-matmul-v1accel.mlir).


## Installing SystemC

```shell
# Prepare important folders
mkdir -p ~/Downloads/systemc/
sudo mkdir -p /opt/systemc/systemc-2.3.3

# Download
pushd ~/Downloads/systemc
wget https://accellera.org/images/downloads/standards/systemc/systemc-2.3.3.tar.gz
tar -xvf systemc-2.3.3.tar.gz 

# Compile and install
mkdir ~/Downloads/systemc/systemc-2.3.3/build
pushd ~/Downloads/systemc/systemc-2.3.3/build
../configure --prefix=/opt/systemc/systemc-2.3.3/
make -j10
sudo make install

# Set the relevant environment variables
export SYSTEMC_HOME=/opt/systemc/systemc-2.3.3
popd
popd
```


## Running SystemC

After compiling llvm and installing SystemC, a systemc library can be compiled
with:

```shell
cd $PROJ_ROOT/builds/llvm-project/build-x86
ninja mlir_syscaxi_runner_utils

# Or for all sysc accelerators:
cmake --build $PROJ_ROOT/builds/llvm-project/build-x86 --target \
    mlir_syscaxi_runner_utils_accv1 \
    mlir_syscaxi_runner_utils_accv2 \
    mlir_syscaxi_runner_utils_accv3 \
    mlir_syscaxi_runner_utils_accv4 \
    mlir_syscaxi_runner_utils_conv_accv1 \
    mlir_syscaxi_runner_utils_conv_accv3
```

A 4 by 4 matmul example can be executed with mlir jitter, triggering a systemC
simulation with:
```
$PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-opt \
        -convert-linalg-to-loops -convert-scf-to-cf   -convert-vector-to-llvm \
        -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts \
        $PROJ_ROOT/llvm-project/mlir/test/axi4mlir-runner/run-axi-v1-data-copy.mlir | \
    $PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-cpu-runner \
        -O0 -e main -entry-point-result=void \
        -shared-libs=$PROJ_ROOT/builds/llvm-project/build-x86/lib/libmlir_syscaxi_runner_utils.so \
        -shared-libs=$PROJ_ROOT/builds/llvm-project/build-x86/lib/libmlir_runner_utils.so 
```

The same mlir code is automatically executed with `ninja check-mlir`, but the
llvm tests only check for the existence of the dma mock library
(`libmlir_mockaxi_runner_utils.so`) without systemc simulation:
```shell
$PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-opt \
        -convert-linalg-to-loops -convert-scf-to-cf   -convert-vector-to-llvm \
        -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts \
        $PROJ_ROOT/llvm-project/mlir/test/axi4mlir-runner/run-axi-v1-data-copy.mlir | \
    $PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-cpu-runner \
        -O0 -e main -entry-point-result=void \
        -shared-libs=$PROJ_ROOT/builds/llvm-project/build-x86/lib/libmlir_mockaxi_runner_utils.so \
        -shared-libs=$PROJ_ROOT/builds/llvm-project/build-x86/lib/libmlir_runner_utils.so 
```


A `C(16x32) = A(16x8) x B(8x32)` accelerator v1 example can be executed with
the mlir jitter, triggering a systemC simulation with:
```
$PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-opt \
        -test-accel-to-axi4mlir \
        -convert-linalg-to-loops -convert-scf-to-cf   -convert-vector-to-llvm \
        -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts \
        $PROJ_ROOT/llvm-project/mlir/test/axi4mlir-runner/run-matmul-v1accel.mlir | \
    $PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-cpu-runner \
        -O0 -e main -entry-point-result=void \
        -shared-libs=$PROJ_ROOT/builds/llvm-project/build-x86/lib/libmlir_syscaxi_runner_utils.so \
        -shared-libs=$PROJ_ROOT/builds/llvm-project/build-x86/lib/libmlir_runner_utils.so 
```

## Running on an ARM-32 bit device

```
export PROJ_ROOT=/working_dir
$PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-opt \
        -test-accel-to-axi4mlir \
        -convert-linalg-to-loops -convert-scf-to-cf   \
        -arith-expand \
        -memref-expand \
        -convert-vector-to-llvm \
        -convert-memref-to-llvm \
        -convert-arith-to-llvm \
        -convert-std-to-llvm \
        -reconcile-unrealized-casts \
        $PROJ_ROOT/llvm-project/mlir/test/axi4mlir-runner/run-matmul-v1accel.mlir | \
    $PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-translate --mlir-to-llvmir -o app.ll

$PROJ_ROOT/builds/llvm-project/build-x86/bin/clang -g \
    --target=arm-linux-gnueabihf -march=armv7-a -marm -mfloat-abi=hard \
    -c -o app.o app.ll

$PROJ_ROOT/builds/llvm-project/build-x86/bin/clang -g -o app app.o \
    --target=arm-linux-gnueabihf \
    -Wl,-rpath=$PROJ_ROOT/builds/llvm-project/build-runner-arm/lib \
    -L$PROJ_ROOT/builds/llvm-project/build-runner-arm/lib \
    -lmlir_runner_utils -lmlir_axi_runner_utils

# Running with qemu
qemu-arm -L /usr/arm-linux-gnueabihf ./app
```

# Compiling AXI4MLIR within docker manually (advanced)

To create a standalone image with all build dependencies installed follow execute:

```
./build-docker.sh
```

To enter the development container

```
./start-docker.sh
```


## Building and testing MLIR inputs in armv7-a devices

Steps:

1. Enter development container
    - With qemu-user, crossbuild-essential-armhf, libz-dev installed
2. Compile bootstrap mlir-tblgen binaries 
    - required when cross compiling for arm
3. Compile clang, llvm, mlir for x86 with the required targets
4. Cross-compile mlir-runner libraries for ARM 
    - this requires arm libs and llvm-project/cmake targeting cross-compilation
    - needs mlir-tblgen from step 2
5. Compile and link desired application for arm, run with qemu
    -  Using compiler from step 3, libraries from step 4, and pointing qemu to look for the right libraries

```bash
# 1. Enter development container
./start-docker.sh # Further steps are executed inside docker container with automatic pwd mounted

# 2. Compile bootstrap mlir-tblgen binaries
./build_tools/build_bootstrap_tblgen.sh llvm-project builds/llvm-project/build builds/llvm-project/install

# 3. Compile clang, llvm, mlir for x86 with the required targets
# This will take a while (20min using 16c/32t)
./build_tools/build_llvm_dev_x86.sh llvm-project builds/llvm-project/build builds/llvm-project/install

# 4. Cross-compile mlir-runner libraries for ARM 
./build_tools/build_runner_arm.sh llvm-project builds/llvm-project/build builds/llvm-project/install

# 5. Compile and link desired application for arm, run with qemu
cd cross-comp/hello-world-mlir
make
make run-arm

# Or for individual steps on AOT compiling app.mlir into an arm binary
cp $PROJ_ROOT/llvm-project/mlir/test/Integration/Dialect/Linalg/CPU/test-conv-1d-nwc-wcf-call.mlir app.mlir
$PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-opt -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm -lower-affine -convert-scf-to-cf --convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts -o app-llvm.mlir app.mlir
$PROJ_ROOT/builds/llvm-project/build-x86/bin/mlir-translate -mlir-to-llvmir -o app.ll app-llvm.mlir
$PROJ_ROOT/builds/llvm-project/build-x86/bin/clang --target=arm-linux-gnueabihf -march=armv7-a -marm -mfloat-abi=hard -c -o app.o app.ll
$PROJ_ROOT/builds/llvm-project/build-x86/bin/clang -o app app.o --target=arm-linux-gnueabihf -Wl,-rpath=$PROJ_ROOT/builds/llvm-project/build-runner-arm/lib -L$PROJ_ROOT/builds/llvm-project/build-runner-arm/lib -lmlir_runner_utils 
# Running with qemu
qemu-arm -L /usr/arm-linux-gnueabihf ./app
```

## Recompiling inside the container

Once these steps finished, build folders will be available in:

```
├── builds/build-runner-arm    # (ARM) targets: mlir_runner_utils mlir_axi_runner_utils mlir_runner_utils
├── builds/build-tblgen-x86    # (x86) targets: mlir-tblgen clang-tblgen llvm-tblgen
└── builds/build-x86           # (x86) targets: mlir-opt mlir-translate mlir_runner_utils mlir_axi_runner_utils

export PROJ_ROOT=/working_dir
cmake --build $PROJ_ROOT/builds/llvm-project/build-runner-arm \
    --target mlir_c_runner_utils mlir_runner_utils mlir_axi_runner_utils mlir_mockaxi_runner_utils

cmake --build $PROJ_ROOT/builds/llvm-project/build-x86 \
    --target mlir_c_runner_utils mlir_runner_utils \
    mlir_axi_runner_utils mlir_mockaxi_runner_utils \
    mlir_syscaxi_runner_utils \
    mlir_syscaxi_runner_utils_accv1 \
    mlir_syscaxi_runner_utils_accv2 \
    mlir_syscaxi_runner_utils_accv3 \
    mlir_syscaxi_runner_utils_accv4 \
    mlir_syscaxi_runner_utils_conv_accv1 \
    mlir_syscaxi_runner_utils_conv_accv3
```

To recompile inside docker, with `PROJ_ROOT=/working_dir`:

```
cmake --build /working_dir/builds/llvm-project/build-x86 --target clang opt mlir-opt mlir-translate mlir-cpu-runner mlir_c_runner_utils mlir_runner_utils mlir_axi_runner_utils mlir_mockaxi_runner_utils mlir-lsp-server FileCheck && cmake --build /working_dir/builds/llvm-project/build-runner-arm --target mlir_c_runner_utils mlir_runner_utils mlir_axi_runner_utils mlir_mockaxi_runner_utils
```

# One commandline to rule them all:
```
cmake --build /working_dir/builds/llvm-project/build-x86 --target clang opt mlir-opt mlir-translate mlir-lsp-server FileCheck \
 mlir-cpu-runner mlir_c_runner_utils mlir_runner_utils mlir_axi_runner_utils mlir_mockaxi_runner_utils \
 mlir_syscaxi_runner_utils \
 mlir_syscaxi_runner_utils_accv1 \
 mlir_syscaxi_runner_utils_accv2 \
 mlir_syscaxi_runner_utils_accv3 \
 mlir_syscaxi_runner_utils_accv4 \
 mlir_syscaxi_runner_utils_conv_accv1 \
mlir_syscaxi_runner_utils_conv_accv3 \
 && \
cmake --build /working_dir/builds/llvm-project/build-runner-arm --target mlir_c_runner_utils mlir_runner_utils mlir_axi_runner_utils mlir_mockaxi_runner_utils
```