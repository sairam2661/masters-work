# TVM - An Automated End-to-End Optimizing Compiler for Deep Learning

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

# Triton - An Intermediate Language and Compiler for Tiled Neural Network Computations

- language and compiler centered around the concept of a tile, statically shaped multi-dimesnsional sub-arrays

## Introduction

- there has been continous improvements in performance of core-arcitectures (like GPUS)
- allowed researchers and engineers to explore larger models and use more data
  - supported by vendor libraries like cuBLAS and cuDNN, brining latest hardware innovations to researchers
  - but these libraries support a restricted set of tensor operations
  - novel primitives need to be implemented by experts
- for solving this, may Domain-Specific Languages (DSLs) for DNNs are being developed
  - these systems perform well for certain classes of problems, but are still slower than the vendor libraries in practice

### Key Contributions

- Triton-C
  - a C-like language for expression tensor programs as parametric tile variables
  - provides a stable interface for existing DNN transcompilers
- Triton-IR
  - An LLVM-based Intermediate Representation that provides an environment suitable for tile-level program analysis, transformation and optimization
  - constructed directly from Triton-C during parsing
- Triton-JIT
  - a JIT compiler and code generation backend for compiling Triton-IR programs into efficient LLVM bitcode
- Numerical experiments
  - evaluates ability of Tritan against cuBLAS, cuDNN and alternate DSLs

## Related Work

- current DL software still rely on hand-optimized sub-routines (cuBLAS and cuDNN)
- has led to development of various DSLs and copmilers for DNNs
  - Tensor-level IRs - used by XLA and Glow to transform tensor programs into predefined LLVM-IR and CUDA-C operation templates
  - Polyhedral model - used by Tensor Comprehensions and Diessel to parametrize, automate the compilation of DNN layers into LLVM-IR and CUDA-C programs
  - Loop synthesierzs - used by Halide and TVM to transform tensor computations into loop nests that can be further manually optimized

## Triton-C Language

- aim is to provide a stable frontend for existing and future DNN transcopmilers
- be similar to low=level GPU programming for programmers

### Syntax

- syntax based on ANSI C, key changes are,
  - tile declarations - represents multi-dimensional arays (`int tile [10, 10]`) to emphasize their semantical difference with nested arrays (`int tile[10][10]`)
  - built-in functions - to support tile semantics and the SPMD programming model
  - broadcasting - N_dimensional tiles can be broadcast along any particular axis
  - predication - for basic control-flow within tile operations

### Semantics

- tile semantics - built-in _tile_ types and operations
  - simplifies the structure of tensor programs by abstraction
  - opens the door for compilers to perform automatic optimizations
- broadcasting semantics - strongly typed, instructions statically require their operands to obey strict shape constraints
  - a set of rules are needed to perform its conventions
    - padding
    - broadcast
- programming model
  - generally, execution of CUDE code on GPUs is supported by an SPMD programming model
  - Triton programming model is similar, but each kernel is single-threaded and associate with global ranges that varie from instance to instance
  - leads to simpler kernels in which CUDA-like concurrency primitives are inexistent

## Triton IR

- LLVM-based intermediate representaiton whose purpose is to provide an environment suitable for tile-level program analysis, transformation and optimization
- constructed directly from Triton-C during parsing
- shares same high-level structure as LLVM-IR

### Structure

- modules
  - basic units of compilation
  - compiled independently from one other, then aggregated by a linker
  - composed of functions, global variables, consants and other symbols
- functions
  - consists of a return type, name and an arguments list
  - function attributes and parameter attributes can be specified, allowing compiler backends to perform more aggressive optimization
- basic blocks
  - straight line code sequences that may contain terminator instructions at the end
  - uses Static Single Assignment form
  - created direct from ASTs

### Support for Tile-Level Data-Flow Analysis

- types
  - multi-dimensional tiles are central to data-flow analysis in Titon-IR
  - uses syntax similar to LLVM-IR
- instructions
  - introduces set of retiling instructions for suporting broadcasting semantics
  - traditional scala instructions are preserverd and extended to signify element-wise operations on tile operands
  - exposes specialized arithmetic instructions for transpositions and matrix multiplications

### Support for Tile-Level Control-Flow Analysis

- difficult to express the divergent control flow within tiles
- uses predicate SSA form and psi functions to solve this
  - needs addition of two instructions classes (cmpp instructions and psi instructions)

## Triton-JIT Compiler

- simplifies and compiles Triton-IR programs into efficient machine code
  - uses machine-independent and machine-dependent passes backed by an auto-tuning engine

### Machine-Independent Passes

- pre-fetching
  - handling tile-level memory operations inslide loops can be problematic, as could induce severe latency
  - mitigated by directly detecting loops and adding prefetching code where necessary
- tile-level peephole optimization
  - offers new opportunities for peephole optimizers

### Machine-Dependent Passes

- hierarchical tiling
  - nested tiling strategies aim at decomposing tiles into micro-tiles, and eventually into nao-tiles to fit the machin's compute capabilities and memory hierarchy as tightly as we can
  - can automatically enumerate and optimize valid nested tiling configurations
- memory coalescing
  - as Triton-IR programs are single-threaded and automatically parallelized, compiler backed orders threads internally within each micro-tile to avoid uncoalesced memory accesses whenever possible
  - reduces number of memory transactions necessary to load a tile column
- shared memory allocation
  - tile-level operations with high arithmetic intensity can benefit from temporarily storing their operands in a fast shared memory
- shared memory synchronization
  - reads from and write to shared memory are asyncronous in the model
  - this automatically inserts barries in the generated GPU source code to preserve pgoram correctness
  - done by detecting read-after-writes and write-after-read hazards using forward data-flow analysis

### Auto-tuner

- traditionally, this relies on hand-written parametrized code templates
- Triton-JIT can extract optimization spaces directly from Triton-IR programs by combining the meta-parameters associated with the previous optimization passes

## Numerical Experiments

- evaluated on NVIDIA GeForce GTX1070, compared against current vendor libaries (cuBLAS 10.0, cuDNN 7.0), and related compiler technology (Auto-TVM, TC, PlaidML)

### Matrix Multiplication

- on par with cuBLAS (both achieve >90% of peak performance)
- existing DSLs are 2-3x slower than Triton

### Convolutions

- for dense convolutions,
  - outperforms cuDNN in ResNet, on par with cuDNN on DeepSpeech2
- for shift convolutions,
  - hides the cost of shifting, outperforms cuBLAS implementation

## Conclusion

- Triton is an open-source language and compiler for expressing and compiling tiled neural network computations into efficient machine code
- adding a few data-flow and contorl-flow extensions to LLVM-IR enables various tile-level optimization passes, leading to performance on par with vendor libraries
- Triton-C, a higher-level language which implements efficient kernels for novel neural network architectures for CNNs
