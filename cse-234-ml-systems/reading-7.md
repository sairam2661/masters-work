# GPipe

## Introduction

Deep learning has had really good developments the past few years. This growth has mainly been seen in image classification, natural language processing attributed to model capacity. Hoewever, scaling these neural networks that enable the above advancements is difficult. In fact, efficient model parllaleism is really hard to design and implement. With rapid growth in the demand for infrastructure for these neural networks, is super important. GPipe comes into picture here as a flexible library that enables efficient training of large neural netowkrs. It allows the scaling of an arbitary deep neural network architecture beyond memory limitations of a single accelerator by partiitioning the model across different accelerators and supporting re-materilization on every accelerator. This is implmeneted using a pipeline parallelism algorithm with batch splitting. This can further be complemented with data parallelism to further scale training.

## GPipe Library

It is an open source library implemented using the ingvo framework. The interface is designed to be simple and intuitive which requires the user to specify the number of model paritions, micro-batches and the sequence and definitions of layers that define the model.

Once the user has defined the above in the network, GPipe partitions the network into cells and places the k-th cell on the k-th accelerator. It then inserts communication primitives that enables data transfer between neighbouring partitions. During the forward pass, it divides the mini-batch into equal micro-batches which are pipelined through the accelerators. During the backward pass, the gradients for each micro0batch are computed based on the same model parameters used for the forward pass. At the end, the gradients are accumulatede and applied to update the model parameters across all the accelerators.

To redice the activation memory requirements, it supports re-materialization. Now, during forward copmutation, each accelerator only stories output activations at partition boundaries. Similarly, during backward pass the kth acceleartor will recompute the composite forward function, and the peak activation memory requirement is reduced.

There is also a way to reduce the communication overhead as we only need to pass the activation tensors at the partition boundaries between acelerators. As a result, we can scale the performance even without high-speed interconnects.

## Performance Analysis

GPipe performance is evaluated on two types of model architecture - an AmoebaNet convolutional model and a Transformer sequence-to-sequence model. The study demonstrates substantial scalability improvements - GPipe enables training a 1.8-billion-parameter AmoebaNet (25× larger than baseline) on 8 accelerators and scales Transformers to 83.9B parameters (298× increase) using 128 partitions. Transformers achieve near-linear throughput scaling due to uniform layer sizes, while AmoebaNet exhibits sub-linear speedups from imbalanced computation. Communication costs remain minimal, as GPipe transfers only activation tensors between partitions, achieving efficient scaling even without high-speed interconnects.

## Image Classification

GPipe enables training a 557M-parameter AmoebaNet on ImageNet, achieving 84.4% top-1 accuracy. Fine-tuning this model on datasets like CIFAR-10 and CIFAR-100 yields state-of-the-art results (e.g., 99.0% and 91.3% accuracy), demonstrating the effectiveness of large-scale pre-trained models for transfer learning.

## Massive Multilingual Machine Translation

GPipe is also able to scale up models used for natural language processing. For this, Gpipe experiments were carried on a larg-escale mu;tilingual neural machine translation task. For this, a corpus of parallel documents from over 102 languages containing 25 billition training examples was used. Fpr the first time in this space, a large enough MPT model was able to learn the mapping between more than 100 language pairs simultaneoursly whiley ahcieving better than bilingual model performance using a single Transformer. The architecture is scaled along two dimensions to stress test GPipe - depth by incrasing number of layers in the model, and width by increasing the hidden diminsions in feed-forwad layers.

The experiments show that increasing model capacity from 400M to 1.3B parameters significantly improved translation quality across all languages, with deeper models showing better generalization for low-resource languages. However, scaling beyond 1.3B parameters exhibited diminishing returns. To address trainability challenges in deep models, techniques such as scaling down initialization and clipping logit predictions were employed. Additionally, large batch sizes (up to 4M tokens) were tested, resulting in improved BLEU scores and validation loss, demonstrating the effectiveness of GPipe in handling large-scale NLP tasks efficiently.

## Design Features and Trade-Offs

There have been many approachers that have been proposed for efficient large-scale model parallelism. However, every approach chooses its own set of trade-offs, making them suitable for specific architecture situations, under some hardware constraints. For GPipe, ???.

For GPipe, the core idea revolves around partitioning a network into computational units distributed across devices, enabling scalability for a wide range of models. Unlike Single Program Multiple Data (SPMD) approaches, which suffer from high communication overhead and limited applicability, GPipe employs a novel batch-splitting pipeline parallelism algorithm combined with re-materialization. This minimizes bubble overhead and avoids asynchronous gradient updates, allowing linear scaling of model size with the number of accelerators.

GPipe introduces minimal communication overhead, as inter-device communication only occurs at partition boundaries, making it effective even without high-speed interconnects. However, GPipe assumes each layer fits within a single accelerator's memory and requires careful handling of layers that operate across the batch, such as BatchNorm.

## Conclusion

The work introduces GPipe, a scalable library designed for training large neural networks using model parallelism. It proposes a novel batch-splitting pipeline-parallelism algorithm that employs synchronous gradient updates, ensuring efficient hardware utilization and stable training. GPipe is applied to train large-scale convolutional and transformer models, achieving strong results in tasks such as image classification and multilingual machine translation. The library is characterized by three main features: Efficiency, Flexibility, and Reliability - maintaining consistent training through synchronous gradient updates regardless of the number of partitions.

# Alpa

## Introduction

There have been a lot of adcancements in deep learning, leading to significant increases in model size. Particularly, scaling language models such as GPT-3 to hundreds of billions of parameters and training on much larger datasets have presented fundamentally new challenges. Training these models on distributed clusters requires a significant amount of engineering effort specific to both the model definition and cluster environment. In general, this requires tuning a complex combination of data, operator and pipeline parallelization approaches at the granularity of individual tensor operators. Correctly tuning the strategy has been shown to deliver really good impormvenerts in training performance, but depends on strong machine learning and system expertice.

AUtomating the parallelization would help to accelerate ML research production by enabling developers to quickly explore new model designs without regard for underlying system challenges. However, this is really hard to navigate automatically, and very complex. This is caused due to the interplay of different parallelization methods fand their strong dependence on model and cluster setups.

Alpa addresses these challenges by automating the generation of parallel execution plans for distributed deep learning models. It introduces a hierarchical approach that combines intra-operator and inter-operator parallelism, enabling efficient scaling of large models across distributed compute devices. By automating the optimization of parallelism strategies, Alpa reduces the need for manual tuning and allows developers to focus on model design rather than system-level complexities.

The major contributions of tha paper are,

- constructing a two-level parallel execution plance space where plans are specific heirarchilly using inter and intra operator parllelism
- optimization algorithms to derive near optimial execution plan at every level
- Alpa, a compiler sustem for distributed DL on GPU clusters. Some features are,
  - compilations passes that generate execute plans using hierarchilcal optimization
  - runtime architecture that orchestras the inter-op parallelism between devices meshes
  - system optimizations that improve compilation and address cross-mesh communications
- evaluation results that demonstragte significant speeds ups over contemporary methods

## Background

DL computation is represented by popular ML frameworks as a dataflow graph. Edge represent multi-diminesonal tensors, nodels represnet the computational operators that transform the inputs into output tensors. Training a model for one iteration consists of computing a loss by forwarding a batch of data through the graph, deriving updates via a reverse backward pass and applying the udpates to eh parameters via weight update operations.

Conventionally, existing approaches work with data parallelism where we partitional the training data into distributed workers, operator parallelism, where the model is too large to fit in one device and the computation fo a specific operator is computed in parallel across multiple devices, pipeline parallelism where instead of partitioning ops, different groups of ops from the model graph are placed on different workers after splitting teh training data into microbatchers and some other manual combinations of parallelism. Auto-parallelism aims to combine the prior techniques, however suffers from the limitation of combining data parallelism with at modst one model parallelism arpproach, missing out on a lot of performance opportunities.

Different from this, we can also categorize parallelism as intra- and inter-operator parallelisms. Intra-operator parallelism partitions individual operators across devices, while inter-operator parallelism assigns different operators to different devices without partitioning them. This hierarchical approach allows Alpa to optimize parallelism at both fine-grained and coarse-grained levels, leveraging the asymmetric communication bandwidth in compute clusters.

## Overview

Alpa is a compiler that generates model-parallel execution plans by hierarchically optimization the plan at two different levels - intra-op and inter-op parallelism. In the intra-op level, it minimizees the cost of executing a stage of the graph, with respect ot its plan on a given device mesh. At the inter-op level, it minimizes the inter-op parallelization latency with respect to how to slice the model and device cluster into stages and device meshes. This depends on knowing the execution cost of each stage-mesh pair reported by intra-op optimizers.

Alpa achieves this through a series of compilation passes that hierarchically optimize the execution plan. It first slices the model into stages and the cluster into device meshes, then assigns stages to meshes and optimizes intra-op parallelism within each mesh. The runtime orchestrates the execution across meshes, ensuring efficient communication and computation.

## Intra-Operator Parallelism

Alpa optimizes intra-operator parallelism by partitioning operators along tensor axes and assigning partitions to devices within a mesh. It uses an Integer Linear Programming (ILP) formulation to minimize the execution cost of the computational graph, considering both computation and communication costs. The ILP solver selects the best parallel algorithm for each operator, ensuring efficient execution within the device mesh.

This approach allows Alpa to handle a wide range of parallelization strategies, including data parallelism, operator parallelism, and weight update sharding. By formalizing the problem as an ILP, Alpa can efficiently find near-optimal solutions for large computational graphs, even for models with tens of thousands of operators.

## Inter-Operator Parallelism

Alpa optimizes inter-operator parallelism by slicing the model into stages and assigning them to device meshes. It uses a Dynamic Programming (DP) algorithm to minimize the end-to-end pipeline execution latency, considering the execution cost of each stage-mesh pair. The DP algorithm ensures that the model is sliced into balanced stages, and the device meshes are fully utilized. This flexibility allows Alpa to handle heterogeneous models and cluster setups, achieving near-linear scaling for large models. By combining intra-op and inter-op parallelism, Alpa can automatically discover efficient parallelization plans that outperform manually tuned systems.

## Parallelism Orchestration

After stages and device meshes are assigned, Alpa compiles each stage against its assigned mesh and generates parallel executables. The runtime orchestration pass handles cross-mesh communication between stages, ensuring efficient data transfer. Alpa uses a local all-gather optimization to minimize communication overhead between meshes with different shapes. The runtime generates static execution instructions for each device mesh, enabling efficient inter-op parallel execution. This MPMD-style runtime architecture allows Alpa to handle heterogeneous stages and meshes, ensuring optimal performance across the cluster.

## Limitations

Alpa's optimization algorithms have some limitations, including the lack of modeling for cross-stage communication costs and the static nature of the pipeline schedule. Additionally, Alpa does not optimize for overlapping computation and communication, and it requires tensor shapes to be known at compilation time. Despite these limitations, Alpa demonstrates strong performance in generating near-optimal execution plans for large models, making it a valuable tool for distributed deep learning.

## Evaluation

Alpa is evaluated on large-scale models, including GPT-3, GShard MoE, and Wide-ResNet, using a cluster of 64 GPUs. The results show that Alpa matches or outperforms state-of-the-art systems like Megatron-LM and DeepSpeed, achieving significant speedups on models with heterogeneous architectures. Alpa also demonstrates strong scaling efficiency, maintaining near-linear performance as the model size and number of GPUs increase. The evaluation highlights Alpa's ability to automatically generate efficient parallelization plans for a wide range of models, reducing the need for manual tuning and enabling scalable training of large deep learning models.

## Conclusion

Alpa introduces a new architecture for automated model-parallel distributed training, leveraging a hierarchical view of intra- and inter-operator parallelisms. By automating the generation of parallel execution plans, Alpa democratizes distributed model-parallel learning and accelerates the adoption of large deep learning models. The system's ability to unify data, operator, and pipeline parallelism makes it a powerful tool for scaling complex models across distributed compute devices. Alpa's hierarchical optimization and runtime orchestration enable efficient training of large models, reducing the engineering effort required for distributed deep learning.
