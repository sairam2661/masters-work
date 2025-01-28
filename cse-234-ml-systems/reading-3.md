## GPU Performance

## Overview

- basic structure of a GPU
- how operations are performed on a GPU
- how to estimate performance limits of a GPU
- some deep learning operations and how they are performed on a GPU

## GPU Architecture Fundamentals

- at a high level, GPUs consist of a number of Streaming Multiprocessors (SMs), on-chip L2 cache and a high-bandwidth DRAM
- SMs -> arithmetic and other instructions
- DRAM via L2 cache -> data and code access
- for example, NVIDIA A100 GPU contains 108 SMs, 40 MB L2 cache and 2040 GB/s bandwidth from 80 GB of HBM2 memory
- each SM has its own instruction schedulers and execution units
- multiply-add is the most common operation in deep learning
- for example, an A100 GPU with 108 SMs and 1.41 GHz clock rate can perform 312 TFLOPS of FP16 multiply-add operations
- tensor cores are specialized units for matrix multiplication and addition, and can compute products without loss of precision
- when math operations cannot be formulated in matrix block form, CUDA cores are used

## GPU Execution Model

- GPUs are designed to execute many threads in parallel
- relation between thread count and GPU performance,
  - GPUs execute functions using a 2-level hierarchy of threads
  - threads are grouped into blocks, and blocks are grouped into grids
  - GPUs hide dependent instruction latency by switching to execution of other threads
  - as a result, number of threads needed to effectively utilize a GPU is large
- 2-level thread hierarchy is due to many SMs in GPUs
- at runtime, a thread block is placed on an SM for execution
- all threads in thread block can communicate with each other and synchronize
- one thread block = work to one SM => to fully utilize multiple SMs, need to launch multiple thread blocks at once
- number of thread blocks >>> number of SMs, also helps in minimizing tail effect
- efficient to launch functions that execute in several waves of thread blocks
- for higher-end GPus, launches with <300 thread blocks should be examined for tail effects

## Understanding GPU Performance

- limited by memory bandwidth, math bandwidth and latency
- if memory and math time can be overlapped, math limited => math time is longer, memory limited => memory time is longer
- time spend on memory/math dependent on algorithm, implementation
- T math > T memory can be expresses as \# ops / BW math > \# bytes / BW memory
- we can rewrite this as \# ops / \# bytes > BW math / BW memory
- the LHS is the arithmetic intensity, and the RHS is the ratio of math to memory bandwidth
- generally, most common operations have low arithmetic intensity
- also, we assume that we have a high enough workload while analysing performance
- otherwise, the processor will be underutilized
- in general, arithmetic intensity is a first order approximation of performance

## DNN Operation Categories

- three main categories of operations in deep learning

### Element-wise operations

- can be unary or binary
- layers perform mathematical operatios on each element independently
- examples include ReLU, sigmoid, tanh, etc.

### Reduction operations

- operations that reduce the number of elements in a tensor
- examples include pooling layers that compute values over a window of elements

### Dot-Product Operations

- operations that compute dot products between two tensors
- usually a weight tensor and an activation tensor
- examples include fully connected layers, convolutional layers, etc.
- these can be math-limited if the corresponding matrices are large enough; though for small matrices, they are memory-limited

## Summary

- for understanding the performance of a GPU, we need to understand the architecture of the GPU, the execution model and the performance limits of the GPU
  - first look up number of SMs, and determine the ops:bytes ratio
  - next compute the arithmetic intensity for the algorithm
  - determine if there is sufficient parallelism to saturate the GPU
    - estimate number of thread blocks needed to saturate the GPU
    - normally, 4x higher than the number of SMs is good
  - likely reasons for poor performance
    - latency due to insufficient parallelism
    - math if we have sufficient parallelism and algo arithmetic intensity is higher than GPU ops:bytes ratio
    - memory if we have sufficient parallelism and algo arithmetic intensity is lower than GPU ops:bytes ratio

## Matrix Multiplication

## Background

- General Matrix Multiplication (GEMM) is a fundamental operation in deep learning, used in RNNs, LSTM, etc.
- GEMM is defined as the operation C = alpha _ A _ B + beta \* C, where A, B are matrix inputs, C as a pre-existing matrix overwritten by the output, and alpha and beta are scalars
- for example, in a fully connected layer, A is the input activations, B is the weight matrix, and C is the output activations

## Math and Memory Bounds

- following standard convention, assume A is a `MxK` matrix, B is a `KxN` matrix, and C is a `MxN` matrix
- the number of fused multiply-adds (FMAs) in a GEMM operation is `M * N * K`
- each FMA is 2 operations, a multiply and an add, so a total of `2 * M * N * K` FLOPS are required
- computing the arithmetic intensity, we get `2 * M * N * K / 2 * (M * K + K * N + M * N) =  M * N / M * K + K * N + M * N`
- if `M=1` or `N=1`, the arithmetic intensity is normally less than 1 and hence memory limited
- one thing to remember is that while doing this particular analysis, we do not consider many practical aspects of the computation

### GPU Implementations

- GPUs implement GEMMS by partitioning the input matrices into tiles and computing the output matrix one tile at a time

### Tensor Core Requirements

- tensor cores are used to maximize speed of tensor multiplication
- reformance is better when equivalant matrix dimensions M, N, K are aligned to multiples of 16 bytes (can change depending on GPU)
- this alignment ensures that the tensor cores run efficiently
- experimentally, we can see that the fasted calculation for different cuBLAS implementations are when K is divisible by 8 bytes

### Typical Tile Dimensions in cuBLAS and Performance

- cuBLAS library contains NVIDIA's optimized GPU GEMM implementations
- we have different tiling strategies
  - larger tiles allow more data resuse
  - smaller tiles would ensure higher number of tiltes that can run in parallel
- when PyTorch/Tensorflow call into cuBLAS with a specific GEMM dimension, a heuristic internally determines the best tiling strategy
- a consideration that is made is the tradeoff between tile parallelism and tile efficiency
- experimentally, the larger the GEMM, we observe that the tradeoff becomes less important
- different tile sizes are available with `256 x 128` and `128 x 256` being the most efficient, while `64 x 64` is the least efficient
- hence, in general cuBLAS will aovid using small tiles for GEMMs that are large enough where we can have sufficient parallelism with larger tiles

## Dimension Quantization Effects

- GPU function is executed by launching a number of thread blocks, where every block has the same number of threads
- this introduceds two potential effects on execution efficient - tile and wave quantization

### Tile Quantization

- occurs when matrix dimensions are not divisible by thread block tile size
- for example, consider a matrix with dimensions `256 x 256`
  - for a tile size `128 x 128`, we would only need 4 thread blocks
  - consider a minor change to the initial dimensions, `257 x 256`, we would need 6 thread blocks now => inefficient
  - this is inefficient because all tiles perform the same amount of math, so we are wasting the extra 2 tiles
- hence, we achieve highest utilization when output matrix dimensions are divisible by the tile dimensions

### Wave Quantization

- this qunatizes the total number of tiles to the number of multiprocessors on the GPU
- for example, consider varying N, and fixing `K = 4096` and `M=2304`

  - an NVIDIA A100 GPU has 108 SMs
  - considering `256 x 128` block tiles, it executes one thread block per SM
  - this results in a wave size of 108 tiles that can execute at once
  - hence, the GPU utilization will be the highest when the number of tiles is an integer multiple of 108 or little lower

- in summary, what matters regarding quantization is where the quantization occurs;
  - tile quantization means work is quantized to the size of the tile
  - wave quantization means work is quantized to the size of the GPU

# MI300X vs H100 vs H200 s

- tbd
