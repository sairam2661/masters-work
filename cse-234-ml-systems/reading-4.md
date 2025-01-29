# TVM

- a compiler that exposes graph-level and operator-level optimizations to provide performance portability to DL workloads for diverse hardware back-ends
- solves optimization challenges specific to deep learning
  - high-level operator fusion
  - mapping to arbitary hardware primitives
  - memory latency hiding
- deviers performance across diverse hardware back-ends, competitive with SOTA hand-tuned libs
- open sourced, in prod use across multiple companies

## Introduction

- DL is being used across multiple domains - image recognition, NLP, etc.
  - growing demand to deploy these smart apps on different types of devices - cloud, edge, embedded, etc.
  - difficult to map DL workloads to these devices
  - CPU, GPU and TPU-like accerlators require different on-chip memory architectures and compute primitives
- current frameworks such as PyTorch and TensorFlow rely on a computational graph intermediate representation to implement optimizations
  - though, graph-level optimizations are too high-level to handle hardware back-end specific operator-level transformations
  - these operator-level libraries often require singificant manual tuning, and to support different DL frameworks, lot of manual engineering effort
- How can we enable both graph and operator level optimizations at the same time? In current frameworks, we have to choose between either
  - need a fundamentally different end-to-end approach => TVM
- TVM - takes a high-level spec of a DL program and generates low-level optimized code for different hardware back-ends
  - needs to have performance comparable to manually tuned libs, for this need to remember the challenges,
    - leveraging specific hardware features and abstractions - optimized tensor compute primitives, etc.
    - large search space for optimization - produce efficient code without manually tuning operators
  - for doing this, TVM uses three key modules,
    - tensor expression language - builds operators and provides program transformation primitives that generate different versions of optimized programs
    - automated program optimization framework - finds optimized tensor operators (ML-based cost model)
    - graph rewriter - uses the previous high-level and operator-level optimizations

## Key Contributions

- identifies major optimization challenges in providing performance portability to DL workloads for different hardware back-ends
- introduces new schedule primititves that that advantage of cross-thread memory reuse, solving optimization challenges specific to DL
- implements a ML based optimization system to automatically explore search for optimized tensor operators
- builds an end-to-end compilation and optimization stack that allows deployment of DL workfloads specific in high-level frameworks to diverse hardware back-ends
- achieves speedups ranging from 1.2x to 3.8x over existing frameworks w/ hand-optimized libraries

## Overview

- input model from existing framework
- transform into a computational graph
- perform high level graph rewriting
- generate optimized computational graph
- perform operator-level optimization and code generation
- do ML based automated optimization
- obtain an optimized low level loop program
- finish the transformation, and produce a deployable module

## Optimization Copmutational Graphs

- computations graphs are used to represent programs in DL frameworks
  - provide a global view of operators
- TVM exploits this representation to apply high-level optimizations
  - node represens operation on tensors
  - edges represent data dependencies between operations
  - implements many graph-level optimizations such as
    - operator fusion
      - combines multiple operators into a single kernerl without saving intermediate results in memory
      - greatly reduces execution in GPUs and specialized accelerators
      - fuses four categories of graph operators - injective, reduciton, complex-out-fusable, opaque
      - experimentally, we observe better relative speedup with fused operations
  - constant folding
    - static memory planning pass
    - data layout transformations
      - most common data layout choice for storage is column major and row major
      - converts a computational graph into one that can use better internal data layouts for execution on target hawrdware
      - layout transformation based on prefered layout for each operator
      - existing approach is not sustaible since an increasing number of hardware back-ends would require combinatorially higher number of data layouts, types and accelerator intrinsics for support
      - not feasible to handcraft operator kernels

## Generating Tensor Operations

- produces efficient code for each operator by generating many valid implementations on each hardware back-end, and then choosing an optimized one
- built on Halide's idea of decopuling descriptions from computation rules
  - extend to support new optimizations liek nested parallelism, tensorzation and latency hiding

### Tensor Expression and Schedule Space

- introduces a tensor expression language to support automatic code generation
- each expression is described in an index formula expression language
- supports common arithmetic and math operations covering common DL oeprator patterns
- has flexibility for adding harware-aware optimizations
- uses a schedule to denote a specific mapping from a tensor expression to low-level code
- builds a schedule by incrementally applying basic transformations
  - get the tensor expression
  - select schedule primitives from available space
  - get the final schedule
  - perform code lowering
  - produce low level code
- reuses primitives and low-level loop program AST from Halide and introducdes new primitives that optimize GPU and accelerator performance

### Nested Parallelism with Cooperation

- key to improving efficiency of compute-intensive kernels in DL workloads (supported by modern GPUs)
- existing solutions adopt a kind of nested-parallelism
  - uses a parallel scheduling primitives
  - shared-nothing nested parallelism -> one working thread cannot look at data of its sibling within same parallel computation stage
  - alternatively, we can fetch data cooperatively by using a shared memory space (takes advantage of GPU memory hierarchy)
  - introduces memory scopes to mark the above shared memory
  - requires memory synchronization barries for proper and safe sharing
  - lets TVM tag special memory buffers and create special lowering rules when targeting specialized DL accelerators

### Tensorzation

- DL workloads have high arithmetic intensity, decomposed into tensor operators
  - can use tensor compute primitives
  - using these can improve performance, but needs to be seamlessly integrated
  - analgous to vectorization for SIMD architectures
  - need an extensible solution to support new accerlators that could have their own tensor instructions
- this is done by separting the target hardware intrinsic from the schedule using a mechanism for tensor-intrinsic declaration
- introduces a tensorize schdeule primitive to replace a unit of computation with corresponding intrinsics
- compiler then matches the computation pattern with a hardware declaration
- hence, this decouples the schedule from specific hardware primitives, making it extensible
- can also take advantage of hand-crafter microkernels
  - minimizes memory footprint

### Explicit Memory Latency Hiding

- proess of overlapping memory operations with computation
  - this maximizes utilization of memory and compute resources
  - however, it is dependent on target hardware back-end
  - CPU -> does simultaneous multithreading, hardware prefetching
  - GPU -> does rapid context switching of many warps of threads
  - TPU -> does leaner control with a decoupled access-execute architecture
  - programming DAE accelerators with explicit low-level synchronization is difficult
  - to do this, TVM implements a virtual threading scheduling primitive that lets programms specify a high-level data parallel program
  - then it automatically lowers the program to a single instruction stream with above synchronization
  - experimentally, the latency hiding optimizationn resulted in an 18% improvement in peak compute utilization for ResNet inference

## Automating Optimization

- we have a rich set of schedule primitives
  - how to find an optimal operator implementation for each layer of DL model?
- create a specialized operator for input shape and layout associated with each layer
- offers singificant performance benefits
- to achieve this, builds an automated schedule optimizer with two main components
  - schedule explore that propsoes promising new configurations
  - machine learning cost model that predicts the performance of a given configuration

### Schedule Space Specification

- lets developer declare knobs in schedule space
- developer can incorporate domain-specific knowledge while specifiying possible schedules
- also has a generic template for each hardware back-end that automatically extracts possible knobs based on the computation description
- considers as many configrurations as possible -> optimizer manages what to select

### ML-Based Cost Model

- we can do auto-tuning to find the best schedule from large configuration space
  - however, this needs many experiments to identify a good configuration
- alternatively, we can build a predefined cost model to guide the search for a particular hardware baack-end instead of running all possibility and measuring their performance
  - this should consider all factors affecting performance - memory access patterns, data reuse, pipeline dependencies, threading patterns, etc.
    - but doing this would make it burdensome due to increasing complexity of modern hardware
  - instead, we could do a statistical approach to solve the cost modelling problem
    - a schedule explorer proposes configurations that may improve an operators performance
    - for each schedule configuration, an ML model that takes the lower loop program as input and aims to predict the runtime on a given hardware back end
    - model is trained using runtime measurement data collected during exploration, and does not require input user information regarding the hardware
    - quality of ML model improves with more trials
    - strikes a balancfe between auto-tuning and predefined cost modelling
- two key choices to consider while choosing model - quality and speed
  - schedule queries cost model frequently, will incur overhead
  - cannot ignore this overhead as it can be in the order of seconds
  - hence need to choose the right model

### Schedule Exploration

- the cost model from above is used to select promising configurations, which iteratively runs real measurements
- for each iteration, the explorer selects a batch of candidates on which the measurements need to be run
- collected data is then used as training data to update the model
- can use an enumerative algorithm that selects top-k predicted performers, but this is not scalable
- instead, uses a parallel simulated annealing algorithm
  - starts with random configuration, then walks to naerby configuration
  - transition is successful if cost decreases as predicted by the cost model

### Distributed Device Pool and RPC

- scales up the running of on-hardware trials
- enables fine grained resource sharing
- TVM implements a customized RPC-based distributed device pool that enables this
- the same infrastructure can perform a single workload optimization and end-to-end graph inference
- automates the compile, run, and profile steps across multiple devices - no manual effort for embedded devices

## Evaluation

- does it optimize DL workloads over multiple platforms
  - evaluated by comparing end-to-end performance of deep neural networks on an Nvidia Titan X
  - outperforms the baselines, with speedups ranging from 1.6x to 3.8x due to joint graph optimization and the automatic optimizer
- how does it compare to existing DL frameworks on each back-end
  - evaluated against TensorFlowLite for ResNet and MobileNet on an ARM Cortex A53
  - outperforms hand-optimized ones for both neural network workloads
  - supports ultra low-precision operators
- support new, emerging DL workloads (depthwise concolution, low precision operations)
  - runs the end-to-end pipeline on a Firefly-RK3399 board with an ARM Mali-T860MP4 GPU against a baseline vender library
  - outperforms the baseline on the three available models, with speedups ranging from 1.2x to 1.6x
- support and optimize for new specialized accelerators
  - TVM demonstrates its ability to generate efficient schedules for the Vanilla Deep Learning Accelerator (VDLA), achieving a 40x speedup for offloaded convolution layers on an FPGA
  - the overall performance is bottlenecked by CPU-bound operations, highlighting the potential for further optimization by extending VDLA to support additional operators

## Related Work

- Frameworks and Libraries
  - extends existing frameworks by enabling optimized code generation for diverse hardware, reducing reliance on vendor-specific libraries
- Scheduling and Optimization
  - adopts Halide's scheduling principles and introduces domain-aware auto-tuning to optimize DL workloads for GPUs and accelerators
- Accelerator Compilation
  - addresses challenges in compiling for accelerators like TPUs and FPGAs, offering a generic solution through tensorization and latency hiding

## Conclusion

- presents an end-to-end compilation stack for deep learning, automating optimization across diverse hardware back-ends and paving the way for future system software-hardware co-design advancements

# Triton

- language and compiler centered around the concept of a tile, statically shaped multi-dimesnsional sub-arrays

## Introduction

## Related Work

## Triton-C Language

### Syntax

### Semantics

## Triton IR

### Structure

### Support for Tile-Level Data-Flow Analysis

## Triton-JIT Compiler

### Machine-Independent Passes

### Machine-Dependent Passes

### Auto-tuner

## Numerical Experiments

### Matrix Multiplication

### Convolutions

## Conclusion
