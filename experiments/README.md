This folder contains code and scripts to reproduce the experiments and generate the relevant graphs presented in our paper.
# Directory structure

Each subfolder contains code to compile and run experiments described in the paper.
All each experiment folder contains similar set of folder and files:
* analysis/
  * contains results(.csv) produced in the paper
  * "ex?-generate-paper-graphs" notebook can be used to generate graphs used in the paper
  * "ex?-generate-new-graphs" notebook can be used to generate graphs with new data from executing on the PYNQ board (see PYNQ Execution section below)

* ex?_pynq/
  * all pynq libraries, apps and scripts to run the expirements are store in the this folder
* srcs/
  * source code for running the experiment
  * source code for manual C++ implementation used for comparison
  * templates to generate mlir problems for different problem size
* scripts/
  * contains scripts to generate problems for each experiment
* compile_sysc.sh
  * compiles all libraries and apps for the experiment (generates ex?_sysc folder)
* run_sysc.sh
  * runs all the compiled SystemC experiment
* compile_pynq.sh
  * compiles all libraries and apps for the experiment targeting the PYNQ board
* pynq_send.sh
  * sends the experiment to remote PYNQ Z1 board
* pynq_run.sh
  * executes and retrieves results from experiments on the remote PYNQ Z1 board


# Experiments
## Ex1

This experiment will help reproduce code results for **Figure 13** &  **Figure 12b**. Here we use matmul accelerator v2 & v3 with different accelerator size (8,16) across different matmul problems (M=N=K=[64,128,256]) and different dataflow strategies (Ns,As,Bs,Cs) with both AXI4MLIR generated code and handwritten C++ driver code.

## Ex2
This experiment will help reproduce code results for **Figure 14**. Here we use matmul accelerator v4 for different problem sizes, evaluating different tiling and dataflow stratgies.

## Ex3
This experiment will help reproduce code results for **Figure 16**. Here we use convolution-based accelerator for all the convolution layers for ResNet18 with both AXI4MLIR generated code and handwritten C++ driver code. 

## Ex4
This experiment will help reproduce code results for **Figure 17**. Here we use matmul accelerator v4 for all the Matmul layers for tinyBert with both AXI4MLIR generated code. We are able to see the end-to-end speedup over CPU only execution of the model with Nothing stationary execution and Best effort execution.



# SystemC Execution
For demonstration purposes we have integrated SystemC simulation for our accelerators and enabled AXI4MLIR to generate code to drive these SystemC accelerator simulation.

Within each experiment's folder, we provide script to generate systemC-based experiment code and also one to run it and verify correctly of experiments against CPU execution.

The following example is for ex1, but same can be replicated for all the experiments.
Compile & run SystemC experiment:
```
$ ./start_docker
$ cd experiments/ex1/
$ ./compile_sysc.sh
$ ./run_sysc.sh
```

Running the SystemC experiment, you will see which tasks is being executed and whether or not if it passes the evaluation.

Additionally during compilation, the script also generates an itermediate representation of all the tasks. This representation is saved within ./ex1_sysc/intermediate_[acc_name].mlir.

# PYNQ Execution
For testing on the real FPGA device, we have created a test user for the artifact evaluator to access the device remotely. Within each experiments folder, we have created scripts to cross-compile the experiments, copy to and execute the compiled experiments on the device and finally retrieve experimental results back to the host machine.

The following example is for ex1, but the same can be replicated for all the experiments.
Cross-compile and send experiment binaries to the PYNQ board:
```
$ ./start_docker
$ cd experiments/ex1/
$ ./compile_pynq.sh
$ ./pynq_send.sh
```

Execute experiments on the PYNQ board and retrieve results:
```
$ ./pynq_run.sh
```
This script will ssh you into the board remotely, load the correct bitstreams, and run all the tasks for the given experiment.
After running the tasks, the script will copy the results back to the host device at ./ex1_pynq/results/.
We can use these results (csv files) to re-generate graphs presented in the paper.

# Analysis
Each experiment contains an "analysis" folder, which will contain the original
data that was generated for the paper. Additionally, we include two Jupyter
notebooks and two python scripts; both will generate the relevant graphs/plots
presented in experiment. 

All dependencies are already installed inside the Docker container for the
*python scripts*. The python scripts can be executed with variations of the
command below:

```bash
$ ./start_docker.sh
$ cd experiments/ex1/analysis
$ python3 ex1-generate-paper-graphs.py
```

To use the notebooks, please set up the conda environment in your system using
the following commands, or run the first cell (containing the pip3 command) of
any of the notebooks to install the required Python packages.

```bash
$ conda create -n bench python=3 pandas matplotlib numpy
$ conda activate bench
```

The notebook or script (marked with "paper") will use the original data and save
figures in an output folder. The other notebook or script (marked with "new")
will use generated results from the PYNQ board to recreate the figures.
