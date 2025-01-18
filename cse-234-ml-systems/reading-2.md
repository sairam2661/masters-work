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
