# TensorFlow

## Introduction

- machine learning has been doing really well in the past few years because of
  - sophisticated models
  - large datasets
  - computational resources

### What is tensorflow?

- open-source machine learning library
- developed by Google Brain Team
- used for building, training machine learning models, and moving them to production
- based on DistBelief, a deep learning library developed by Google in 2011
- supports both large-scale training and inference
- uses unified dataflow graph to represent computation and the state in an algorithm
- graph nodes represent computations on mutable states
- edges carry tensors between nodes
- allows experimentation of different parallelism strategies
- supports distributed computing

## Background

### Requirements

#### Distributed Execution

- modern machine learning models perform better on larger datasets
- data-parallel approach to training, eliminates I/O bottleneck
- models also have large number of parameters, use distributed system for this
- need to use network efficiently, e.g., mini-batch gradient descent

#### Accelerator support

- ML algorithms perform expensive computations
- general purpose CPUs are not efficient for this
- GPUs are better for this
- TPUs are even better, really good performance-per-watt when compared to GPUs and CPUs
- TensorFlow supports both GPUs and TPUs

#### Training and Inference Support

- scalable and high-performance inference is a requirement for production ML systems
- can be required on a very low latency service, or even disconnected mobile devices
- requires distributed computing
- training and inference require similar performance, need optimized system for both computations
- need to be able to accelerate both training and inference on GPUs

#### Extensibility

- provides ability to experiment
- scale code in production
- support expressive control flow constructs

### Related Work

- Single-machine frameworks: Theano, Torch, Caffe, etc. Tensorflow's model is close to Theano's dataflow representation
- Batch dataflow systems: MapReduce, Spark, Naiad, etc. Traditional systems require input data to be immutable, TensorFlow allows mutable state (similar to Naiad)
- Parameter servers: DistBelief, Project Adam, etc. TensorFlow uses a similar architecture to DistBelief, but with more flexibility and performance

## TensorFlow Execution Model

- uses a single dataflow graph to represent all computation and state
- supports multiple concurrent executions on overlapping subgraphs of overall graph
- mutable state on graph nodes, can be shared between different graph executions
- mutable state is very important
- able to experiment with different opimization algos, consistency schemes, etc.

### Dataflow Graph Elements

- node = atomic unit of computation (operation)
- edge = output of one node, input of another (tensor)

#### Tensor

- all data is modeled as tensors
- tensor = typed, dense, multi-dimensional array
- for sparse tesnors, we encode them or use a tuple of dense tensors

#### Operation

- takes m>=0 tensors as input, produces n>=0 tensors as output
- operation = function that computes output tensors from input tensors
- operation has a name, type, and attributes
- operation can be stateful, has internal state that affects output
- variable = operation with state
- queue = operation with state that can be mutated by enqueue and dequeue operations

### Partial and Concurrent Execution

- TensorFlow allows partial execution of the graph
- can execute subgraphs of the overall graph, called a step
- can execute multiple steps concurrently

### Distributed Execution

- dataflow simplifies distributed execution, communication between subgraphs is explicit
- same TensorFlow program can be deployed on a single machine, a cluster
  or a mobile device
- each operaation resides on a _device_, which can be a CPU, GPU, or TPU, in a particular _task_
- device is responsible for executing a _kernel_ for the operation
- can add constraints to the graph to specify where operations should run
- TensorFlow runtime schedules operations to devices, executes
- optimized to execute large subgraphs with low latency
- _session_ = runtime state, holds all the state (variables, queues, etc.)
- model favors static, resuable graphs but can also support a dynamic control flow

### Dynamic Control Flow

- most evaluation is _strict_ (eager)
- but some algos, like RNNs, require dynamic control flow
- this is supposrted using _Switch_ and _Merge_ operations
- also used for implementing loops
- supports multiple concurrent iterations and nested loops

## Extensibility

### Differentiation and Optimization

- contains a librabay that automatically differentiates functions
- for example, a neural network can be defined as a composition of layers and a loss function, library will derive the backpropagation
- can also define custom gradients for operations
- have access to a large set of optimization algorithms
- for example, Momentum, Adagrad, RMSProp, Adam, etc.

### Handling Very Large Models

- common to use a _distributed representation_ for models with high-dimensional data
- for example, a word embedding model can have a large number of parameters
- inference produces a much smaller dense matrix representation via embeddings
- sparse embedding layers are achieve using primitive operations - _Gather_, _Part_, _Stitch_, etc.
- able to handle models with billions of parameters

### Fault Tolerance

- training can take very long
- need to be able to recover from failures
- TensorFlow has a mechanism to checkpoint the state of the system
- uses primitive operations like _Save_ and _Restore_ to save and restore the state of the system
- checkpointing behavior can be customized based on user needs

### Synchronous Replica Coordination

- common belief is that synchronous training does not scale
- initially designed for asynchronous training
- experimentation with synchronous training ongoing
- synchronous replication is limited by overall worker throughput
- introduced backup workers to reduce stragglers

## Implementation

- extensible, cross platform library, open-source
- distribute master and worker processes
- master process is responsible for scheduling and coordinating worker processes
- a _dataflow executor_ is used to execute subgraphs
- subgraphs are cached to reduce overhead
- master applies optimizations to the graph
- multiple protocols are supported for communication
- features ported to C++ for performance
- other tools like TensorBoard, TensorFlow Serving, etc. are built on top of TensorFlow to provide additional functionality

## Evaluation

- demonstrats that system has little overhead and can employ large amounts of computation to accelerate real-word applications
- even though idea of TensorFlow is to be scalable, does well on small scales as well
- synchronous training is lmited by the coordination implementation
- however, it can be made efficient by using backup workers
- TensorFlow can be used to train large models on large datasets - for image classification, language modeling, etc.

## Conclusion

- TensorFlow is a scalable, high-performance machine learning library
- supports distributed execution, accelerator support, training and inference support, extensibility
- uses a single dataflow graph to represent all computation and state
- many groups at Google use TensorFlow for research and production
- is open-source, has a large community of users and contributors
- is a good choice for building and deploying machine learning models
- limitations of static dataflow graph still exist, for algos like deep reinforcement learning
- hope is that TensorFlow will continue to evolve and improve

# PyTorch

## Introduction

- popular frameworks construct a static dataflow graph that represents the computation
- this is good for performance, but not for flexibility
- another line of work is to use dynamic eager execution for flexibility, but at the cost of performance
- PyTorch is a Python library that performs immediate execution of dynamic tensor computations, with automatic differentiation and GPU acceleration while maintaining state-of-the-art performance
- PyTorch is used by researchers and developers for building and training neural networks

## Background

Four major trends in scientific computing have become really important for deep learning.

### Array-based Programming

- started with development of domain-specific languages like APL
- turned multidimensional arrays into first-class objects supported by set of mathematical primitives

### Automatic Differentiation

- automtes computing derivatives
- easier to experiment with new models and algorithms
- allows efficient gradient based optimization

### Open Source Python Ecosystem

- Python is a popular language for scientific computing
- packages like NumPy, SciPy, and Matplotlib are widely used
- able to address new problems quickly, share code, and leverage existing code

### Hardware Acceleration

- GPUs have become popular for deep learning
- provide significant speedup for training neural networks
- libraries like cuDNN and cuBLAS provide optimized implementations of common operations

PyTorch combines these trends to provide a flexible and efficient deep learning research platform.

## PyTorch Design Principles

### Be Pythonic

- PyTorch is designed to be idiomatic Python
- integrates naturally with the Python data science ecosystem

### Put Researchers First

- aims to make writing models, data loaders, and optimizers as easy as possible
- provides a flexible, intuitive, and expressive API

### Provide Pragmatic Performance

- provides performance that is competitive with other frameworks
- implementation accepts added complexity to achieve better performance
- providess tool that allows users to understand and optimize performance

### Worse is Better

- PyTorch is designed to be simple and easy to use
- provides a minimal set of abstractions that are easy to understand and use

## Usability Centric Design

### Deep Learning Models are just Python Programs

- machine learning has rapidly evolved over time
- forgoes potential benefits of a graph-metaprogramming approach to preserve the flexibility of Python
- allows users to use Python control flow, data structures, and libraries
- this ensures that any new neural network architecture can be implemented in PyTorch, using Pythonic idioms
- this is true not only for models, but even for data loading and optimization
- these programs execute eagerly, so all feature of Python are available throughout the design process
- immediate execution allows for easy debugging and introspection

### Interoperability and Extensibility

- PyTorch is designed to be interoperable with other Python libraries
- leverage existing libraries for data processing, visualization, and scientific computing
- also extensible
- for example, user can define custom differentiable functions
- free to replace any component of the system with a custom implementation

### Automatic Differentiation

- gradient based optimization is a key component of deep learning
- but python is a dynamic language, so it is difficult to implement automatic differentiation
- PyTorch performs reverse-mode automatic differentiation, which computes the gradient of a scalar output wrt a multivariate input
- can be easily extended to perform forward-mode automatic differentiation using array-level dual numbers
- can also differentiate through control flow, which is useful for imperative programming

## Performance Focused Implementation

- need to learn deep learning algorithms efficiently from a Python interpreter
- normally, DL frameworks based on static data-flow graphs avoid this by deferring the evaluation of the computation to a custom interpreter
- PyTorch solves this by optimizing every aspect of the execution

### Efficient C++ Core

= despite being integrated into Python ecosystem, it is written in C++ for high performance

- provides a tensor library that is optimized for deep learning
- solves the problem of the Python global interpreter lock by releasing the GIL during computation
- allows creation of bindings to other languages

### Separate Control and Data Flow

- maintain strict separation between control (program branches and loops) and data flow (tensors)
- executes operators asynchronously, allowing for parallel execution of operators
- allows tensor operations to reach peak performance, even on Python interpreter

### Custom Caching Tensor Allocator

- need to optimized speed of the dynamic memory allocation
- Python can handle this on CPU, but not on GPU
- PyTorch implements a custom allocator that builds up a cache of CUDA memory and reassigns later
- further tuned to specific memory usage patterns of deep learning
- might not be optimal for some corner cases, though almost never observed

### Multiprocessing

- extends Python's multiprocessing module to allow for efficient data loading
- closely resembles regular threaded programs
- transparently handles sharing of CUDA tensors as well

### Reference Counting

- need to treat memory as a scarce resource that needs to be managed carefully
- traditionally, things like garbage collection is good for eager semantics for managing tensor memory, but given scarcity of GPU memory this is not good
- PyTorch uses a reference counting scheme to track number of uses of each tensor and frees memory when no longer needed
- memory is released exactly when tensors become unused

## Evaluation

- achieves almost perfect device utiliziation using an asynchronous dataflow
- improved memory management and caching strategies observed to be effective
- outputs popular deep learning frameworks on most benchmarks, always within 17% of fastest framework
- explosively adopted by machine learning community

## Conclusion

- PyTorch is a deep learning framework that is flexible, efficient, and easy to use
- designed to be idiomatic Python, and interoperable with other Python libraries
- provides automatic differentiation, eager execution, and GPU acceleration
- current work is on PyTorch JIT compiler, which will allow for more optimizations
- also intent to improve support for distributed computation
