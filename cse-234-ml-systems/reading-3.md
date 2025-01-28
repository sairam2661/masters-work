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

## Intro

- SemiAnalysis has analyzed NVIDIA's H100, H200 and AMDs MI300X and contrasted their various advantages and shortcomings

## Key Findings

- comparing on paper FLOP/s and HBM Bandwidth/Capacity can oft be misleading, the right way is to run benchmarking
- NVIDIA has great out of the box performance and experience, no bugs
- AMD's out of the box is very poor, lots of bugs were found, needs lot of workarounds
- ran unofficial MLPerf Training GPT-3 175B on 256 H100 in collaboration with Sustainable Metal Cloud to test the effects of different VBoost setting
- AMD's real world performence is vastly inferior to the on paper TFLOP/s, NVIDIA also falls, but not by much
- MI300X performance is held back by AMD software
  - many AMD AI libaries are forks of NVIDIA AI libraries, leading to suboptimal outcomes and compatibility issues
  - lags in training performance as well, due to weaker RCCL and lower degree of vertical integration
- AMD customs tend to use hand crafter kernels only for inferenced, resulting in poor performance outside of narrow well defined use cases

## Summary of AMD vs NVIDIA narrative

- MI300X was launched in 2023 with good specs - 1,307 TFLOP/s of FP16 compute (stronger than the H100’s 989 TFLOP/s), 5.3 TB/s of memory bandwidth, and 192GB of HBM3, 3.35 TB/s of memory bandwidth, and 80GB of HBM3
- on paper way better than H200 and H100
- also has lower total cost of owernship, not only due to lower ASP but also as a result of deployment due to cheaper Ethernet networking
- at face value, it provides higher performance and lower total cost of ownership
- however, this is not the case

## General Matrix Multiply Performance

- GEMM performance is a good measure for how well modern transformers such as ChatGPT, Llama would train on the hardware
- testing was performed using OpenAI's do_bench benchmark setup
- for BF16 GEMM, experimental results showed that MI300X is 14% slower than H100 and H200, and way off the marketed BF16 TFLOP/s
- for FP8 GEMM, experimental results showed that MI300X is 22% slower than H100 and H200, and again way off the marketed FP8 TFLOP/s
- additionally, even though GEMM is a simple task, AMD software bugs were encountered - one being a difference in performance between torch.matmul and F.Linear, even though the API calls to the same internal libraries

## Popular GEMM Benchmarks is not Accurate

- a popular benchmark claimed that MI300X performed comparatively to H100
- however, the benchmark had two main issues
  - it did not carry out L2 cache clearing properly
  - takes the max performance, instead of median/mean TFLOP/s over course of iterations

## HBM Memory Bandwidth Performance

- MID300X has better memory bandwidth than H100 and H200, offerring 5.3 TB/s compared to 4.8 TB/s and 3.35 TB/s respectively
- this is very useful in inference and sometimes in training
- although it does not meet its on paper performance, experimental results do show that it performs better than H100 and H200

## AMD Custom Builds

- the stable public release of MI300X was riddled with software bugs, and performance issues
- AMD provided hand-crafted VIP builds, built via a huge dockerfile, requiring ~5 hours for the image to build from source and install dependencies
- not a good experience, as in contrast we can use H100/H200 out of the box with a single command
- NVIDIA's docker images contain complete set of developer tools needed for profiling and debugging
- another thing is AMD also lags behind in using the latest PyTorch releases for its build, something NVIDIA has always synced with
- AMD provided another custom development build, however this were off a hanging development branch having multiples issues
  - required installation of everything from source code, including PyTorch
  - off the development branch, complete QA has not been done

## Training Testing Methodology

- MLPerf GPT3 175B training is a good proxy to measure the time it takes to train a specific convergence
- NVIDIA has public reuslts available for MLPerf training, however AMD has never made theirs public, likely due to weaker results
- SemiAnalysis developed design a benchmark to measure the performance, with a goal on developing a benchmark that is as simple to run as possible, while being a good proxy for performance
- for the model training benchmark, for models are tested
  - GPT 1.5B DDP, representative of small-scale expreiments before scale-out to bigger model sizes
  - Llama3 8B and Llama3 70B 4 Layer Proxy as a baseline for a popular model’s performance
  - Mistral 7B v0.1, which evaluates if hardware will perform well when adding a bit of complexity as it uses sliding window attention instead of the standard causal attention

## Single Node Training Performance

- one thing to be considered is the H100/H200 results are out of the box, where as the MI300X has been fine-tuned and refined over the course of many months
- experimentally, for all models, H100/H200 performs much better than the MI300X public release
- in particular, MI300X does not perform well on smaller models such as GPT 1.5B or any model using a non-causal attention layer like Mistral 7B v0.1
- in fact, H100/H200 even outperforms the custom builds, except the FSDP Llama3 8B and Llama 3 70B Proxy
- we also observe that for anything that is not a simple model, like Mistral 7B v0.1, MI300X is not competitive with H100/H200 even after months of tuning
- hence, for models that doesn't use causal attention, AMD MI300X automatically loses

## Multi Node Training Performance

- H100 performs better than MI300X by around 10-25%
- H200 could not be tested due to no access to a multi-node H200 deployment

## AMD PYTORCH_TUNABLE_OPS FLAG is a Bad User Experience

- AMD has a specific prototype flag for the end user to tune GEMMS
- this is a prototype feature with multiple bugs
- takes 1-2 hours to tune any modern LLM model
- results are not cached, any minor changes would require the entire run to go again
- NVIDIA does not require this flag as cuBLASLt coms with an out of the box heuristic model, that pick the correct algorithm experimentally
- AMD's hipBLASLt/rocBLAS's heuristic model tends to pick the wrong algorithm for most shapes out of the box (this is why so much time-consuming tuning is required in the first place)

## Scale Up NVLinx/xGMI Topology

- extremely important for GPU clusters, as it provides a fast path for tensor and expert parallelism used in frontier model training
- NVLink, scale up fabric of H100/H200 provides 450 GByte/s of bandwidth per GPU and connects 8 GPUs together
- xGMI, scale up fabric of MI300X provides 448 GByte/s of bandwidth per GPU and also connects 8 GPUs together
- on paper, they seem really similar but actually are very different
- xGMI uses a point to point mesh topology - it does not actually provide 448 Gbyte/s of bandwidth between GPU pairs, rather it is 64 GByte/s due to the topology
- NVLink, uses a switched topography and allows a GPU to talk to another at the full 450 GByte/s; additionally, the NVSwitches also support in-network reduction to reducde data movements

## Collectives Overview

- main set of collectives used in frontier LLM training are tested,
  - all_reduce - data parallelism and tensor parallelism
  - all_gather - ZeRO/FSDP parallelism and tensor parallelism
  - reducde_scatter - Zero/FDSP parallelism
  - all to all
- real-world message sizes range from 16 MiB to 256 MiB, this is used for testing

### Single Node NCCL Collective

- H100/H200 outperforms MI300X across all benchmarks, mainly due to the topology of NVLink when compared to xGMI

### Multi Node RCCL/NCCL Collectives and Scale Out Network Benchmarks

- on both the H100/H200 and MI300X, each GPu is connect to other nodes over the scale out network using a 400G Network Interface Card, connected directly to every GPU
- H100/H200 referencfe design typicaly uses Spectrum-X (NVIDIA's custom ethernet solution), while MI300X recommends using RoCEv2 Ethernet with Broadcom Thor-2 NIC
- Similar to NVLS, NVIDIA offers InfiniBand SHARP, which provides in-network reduction exclusive to NVIDIA
- AMD does not have any analgous offerings for this

- NVIDIA's InfiniBand SHARP provides in-network reduction, significantly reducing network traffic and maintaining constant all-reduce performance regardless of GPU count, while AMD lacks an analogous solution for the MI300X.
- AMD's MI300X with RoCEv2 Ethernet shows significantly slower collective performance (2-4x slower) compared to NVIDIA's InfiniBand and Spectrum-X solutions, particularly in all-reduce, all-gather, and reduce-scatter operations, highlighting AMD's need for improved scale-out network integration and software optimization.

## AMD has a suboptimal User Experience for MI300X out of the box

- caused due to poor internal tesitng and a lack of automated testing
- even though state to run over 200k tests a day, has done little to solve previous/existing issues
- existing docoumentation and implementation differs in implementation at times
- not in sync with latest PyTorch releases
- for example, FlexAttention that could help improve performancef by 10-20 times with sliding windo attention, cannot be used in MI300X due to multiple AMD software bugs (fixed in latest release), where as NVIDIA's FlexAttention has been functional since release for over six months
- six months is a very long time for frontier AI labs

## Exploring Ideas for Better Performance on AMD

- MI300X requires multiple environment flags for making it usable and improving performance, many having complex interactions with each other making troubleshooting difficult as well
- instead of using flags for tuning performance, it would be better to fix internal AMD libraries

### AMDs Forked Libraries

- many AMD libraries are forked from NVIDIA's open source libraries
- by doing this they cannot expect to surpass NVIDIAs user experience and software development strategy
- ideally, they should try to contribute their software to the AMD ecosystem directly, by performing native training on their own hardware

### Recommendations to AMD for fixing their Software

- focus on more software engineering resources
- procure more GPUs for in-house development work and submitting an MLPerf GP3 178 result
- increase standard for public images by using them interanlly and with customers, intead of developing custom images every time
- transform approach to environmental flags, to faciliate runs from out of the box and an easier user experience
- collaborate with Meta to get production training workloads
