# ChiaVec

## Overview

ChiaVec is a powerful tool that offers sequence containers, namely Vec and CudaVec. Vec allocates memory on CPUs while CudaVec allocates memory on GPUs using CUDA. ChiaVec performs memory management across different processing units under the hood. This makes it easy for high-performance computing tasks using C++. C and Rust bindings are also provided.

## Getting Started

# Building the project
Create a folder in the project's root directory.
```bash
mkdir build
```
Change the working directory to the "build" folder.
```bash
cd build
```
Initialize the project using 'cmake'.
```bash
cmake ..
```
Build the project using 'make'.
```bash
make
```

# Run Examples
Upon successful compilation of the project, example executables will be located in the ‘build’ directory. To run these examples, use the following commands:
```bash
./examples/c++/cpp_example
```
```bash
./examples/c/c_example
```
