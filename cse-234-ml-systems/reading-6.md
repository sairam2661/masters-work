# ML Parallelism Blog

## Introduction

There are tons of large language models these days, and have a large open source community supporting them - finetuning, inference and a better development ecosystem.As a result, a lot of open-source codebases have been developed using the HuggingFace ecosystem. This makes one thing clear - we will be using a lot more compute, and will need more GPUs. This blog posts covers the basics of distributed training, and how to scale up your training to use more GPUs.

## Distributed Training Basics

When we are working with training or finetuning LLMs, we are dealing with verylarge model sizes, and very large datasets. How do we achieve the maximum possible throughput given this? We know that LLMs need a lot of GPUvRAM to train, because of the large model weights and the optimizer states. As a result, we will be needing smart distributed training strategies to make the most of the available resources. The main strategies are,

1. Data Parallelism - Each GPU worker gets a mini-batch of data, and then computes the gradients, which are then averaged across all workers. This is kinda like the PyTorch DDP paradigm.
2. Model Parallelism - Models are vertically sliced here, where different layers of the model are placed on different GPU workers.
3. Pipeline Parallelism - This is an improvement on the above, which gives us the illusion of paralleism by overlapping computation for different layers of the model.
4. Tensor Parallelism - This is a new paradigm, where we split the model weights horizontally across different GPUs, and then perform computation on the split weights.

## ZeRO-powered Data-Parallelism

This is really efficient and popular strategy for distributed training presently. This is a form of data paralleism aimed to improve memory efficiency. The idea is that it exploits memory redudancy xin data-parallel training and especially the newest improvements in inter-GPU communitcations help in improving the throughput. This has two components that take care of the data parallelism and residual memory. It also has multiple optimizations that help in further improving the throughput.

This has different stages of optimization, which are,

1. ZeRO-1 - This is the basic version of ZeRO, which is a memory optimization technique that reduces the memory footprint of the optimizer states. The idea is to only do optimizer state partitioning here.
2. ZeRO-2 - This is an extension of ZeRO-1, which also partiions the gradients and the model weights.
3. ZeRO-3 - This isthe most advanced version of ZeRO, which additionally partitions the parameters of the optimizer.
4. ZeRO-R - The improves ZeRO-DP by focusing on memory consumption by activations and maanging memory fragmentation.
5. ZeRO-Offload - This is a new version of ZeRO, which offloads the optimizer states to the CPU, and the gradients to the GPU.
6. Zero-Infinity - This is an improvement over ZeRO-Offload, which allows offloading to disk.
7. ZeRO++ - This is an improvement over ZeRO-3, which allows for more efficient memory management using quantized weights and hierarchical partitioning.

## Fully-Sharded Data-Parallelism

This is another data parallelism technique that aims to improve memory efficiency with limited communication overhead, and thereby resulting in better throughput. This is inspired from ZeRO. This has two sharding strategies, which are,

1. Full Sharding - This is kind of similar to ZeRO-3 where we have parameters, an optimizer state and gradients that are being shared across the devices. Each devices holds a subset of the weights and there is on-demand communicaiton to compute intermediate activations and gradeints.
2. Hybrid Sharding - This contains both sharding and replication. So, for a given number of decices, the sharding happens only across a subset, with replication across different subsets. The idea is kind of similar to the hierarchical partitioning in ZeRO++.

## Implementation

The implementation of these strategies is quite simple, and can be done using the PyTorch and HuggingFace accelerate libraries.

## Efficient Finetuning Methods

Some popular methods for efficient finetuning are,

1. Mixed Precision - Weights, activations and gradients are stored in half-precision formats whilemainintaing a master copy of the weights in single-precision.
2. Parameter-Efficient Fine-Tuning - This aims to reduce the memory requirements during finetuning, by freezing the model weights and having a subset of parameters that are training. Most popular method in this space are LoRA, IA^3 and QLoRA.
3. Flash Attention - This is a new attention mechanism that is more efficient than the standard self-attention mechanism. This supports Ampere, A100 and H100 GPUs.
4. Gradient Checkpointing - Traditionally, in each forward pass, the intermediate activations are retained in memory as theyare needed to compute the backward pass. This method aims to reduce this memory consumption by only retaining a subset of intermediate activations and recomputing the rest on the fly. Though, this results in more recomputation time.
5. Quantization - This is a method to reduce the memory footprint of the model by reducing the precision of the model weights. This is done by quantizing the weights to 8-bit integers, which reduces the memory footprint by a lot (normally around 4x).
6. Gradient Accumalation - This is a method to increase the effective batch size by accumulating the gradients at some drop in throughput. This is especially good with multi GPU training as we get larger batch sizes along with faster training times.

## Practical Guidelines

- Use BF16/FP16 by default - pretty convenient and gives a good speedup.
- Use LoRA with trainable parameters added to all the linear layers.
- Use Flash Attention for better performance, if supported by the GPUs.
- Use gradient checkpointing for large models, as it helps in reducing the memory footprint.
- Use quantization for reducing the memory footprint of the model, especially when working with very limited GPU memory.
- For a small-scale multi-node setup, best option is to use ZeRO-3 with hierarchical partitioning.
- If we are short on GPU memory, then we need to activate CPU/disk offloading.
- Also, we need to ensure that we calculate the effective batch size and adjust hyperparameters accordingly.
- Finally, we need to remember to monitor the training process and ensure that GPUs are being utilized to the maximum.

## Open-Source Implementations

There are a lot of open-source implementations that support these strategies. Some of the popular ones are,

1. FastChart - This is a platform that supports finetuning,serving and evaluating LLM-based chatbots for LMSys.
2. Axolots - This is a largeopen-source effort for finetuning language models, and is built on top of HuggingFace.

## Conclusion

This blog post covers the basics of distributed training, and how to scale up your training to use more GPUs. It also covered the different strategies for distributed training, and the different optimizations that can be done to improve the throughput. It also shared some popular methods for efficient finetuning, and some practical guidelines for implementing these strategies. Finally, we see some open-source implementations that support these strategies.

# Megatron-LM

## Introduction

NLP has been growring quickly due to an increase in compute and dataset sizes. This has led to the development of large language models that have billion of parameters. Experimentally, these models are very useful for NLP tasks. By finetuning these models, we can get SOTA results on a variety of tasks.

As these models are very large, they require a lot of compute to train. However, widely used optimization algorithms require additional memory per paramter which reduces the size of the model that can be trained. This is where Megatron-LM comes in. It is a framework that allows for training very large language models with billions of parameters. It is built on top of PyTorch and is designed to scale to thousands of GPUs.

Its key contributions are,

- a simpple and efficient model parallel approach over existing PyTorch transformer implemenetations
- In-depth empirical analysis of model and data parallel technique that displays over 76% scaling efficiecny.
- demonstrates that scaling model size results in improved accurcasies
- models achieve SOTA results on test sets

## Background

Pretrained models have become very important in NLP research. Early examples of pretraining and tranfering neural represntation of previous pre-trained word embedding tables improved downstream tasks results. Further, the SOTA has advanced from tranferring word embedding tables to transferring entire billiton parameter language models. This progression has called for the need of hardware, systems tecniques and efficient frameworks to be able to operatioe efficiently at scale Current work uses transformer models, due to their higher accuracy and compute efficiency.

Another important idea is the use of data and model parallelism in deep learning. These are crucuial when i tcomes to scaling out deepneural network training to different hardware accelerators. Data parallelism aims to use a training minibatch while model paralleism deals with memory usage and computation of themodel. However, these techniques have one limitation - the model size needs to fit onone worker. With language models ofincreasing size and complexity, neural netowrks have approach memory capcityof modern hardware accelerators. To alleviate this, one could do parameter sharing, but this would limit the overall capcaity of the model. The idea of Megatron-LM draws on this to use model parallelism to split the model across multiple accelerators. However, instead of implementaitng a framework and compiler for it, it performs some target modifcations to the existing PyTorch transformer implementations.

## Model Parallel Transformers

A transformer layer consists of a self attention block followed by a two-layer, multi-layer perceptron. Megatron-LM introduces model parallelism in both of these blocks separately, with strategic partitioning of matrix operations to minimize communication overhead. For the MLP block, the architecture splits weight matrices column-wise for the first GEMM operation and row-wise for the second, requiring only one all-reduce operation in each forward and backward pass. The self-attention block leverages inherent parallelism in multihead attention by distributing computations for key, query, and value matrices across GPUs per attention head, followed by a row-parallel output linear layer.

The embedding layer poses unique challenges due to its large size and shared weights between input and output embeddings. To address this, the implementation partitions the embedding matrix along the vocabulary dimension and introduces strategic all-reduce operations. The approach minimizes communication by fusing parallel GEMM outputs with cross-entropy loss and duplicating smaller computations like dropout and layer normalization across GPUs rather than broadcasting results, allowing each GPU to optimize its own set of parameters independently.

## Setup

The work focuses on GPT-2 and BERT (two transformer models), with different configurations. First, multiple large language modelling datasets are collected and aggregated. This includes Wikipeda, CC-Stories, RealNews and BooksCorpus. Next, it filters documents with content length less than 129 tokens. Finally, after using locality-sensitive hashing, it results in an aggregate corpus of 174GB of deduplicated text.

Next, for training the models efficiently it utilizes mixed precision training with dynamic loss scaling to maximize the use of V100's tensor cores. For both models, activation checkpointing is implemented after each transformer layer to optimize memory usage. GPT-2's learning rate follows a cosine decay schedule with a 3k iteration warmup, while BERT uses a linear decay over 10k warmup iterations. Both architectures employ dropout of 0.1 and initialize weights using a normal distribution, with weights scaled before residual layers.

## Experiments

The authors conducted experiments using up to 512 NVIDIA V100 GPUs across 32 DGX-2H servers, optimized for multi-node deep learning with high-speed interconnects. They scaled GPT-2 models from 1.2 billion to 8.3 billion parameters using model and data parallelism, achieving up to 77% efficiency in weak scaling. Larger models, like the 8.3 billion parameter GPT-2, showed faster convergence and lower validation perplexity, setting new state-of-the-art results on WikiText103 and LAMBADA datasets.

The study also highlighted the importance of architectural tweaks, such as rearranging layer normalization in BERT models, to enable stable training of larger models (up to 3.9 billion parameters) and achieve SOTA performance on tasks like MNLI, QQP, SQuAD, and RACE. These findings demonstrate the benefits of scaling model size and optimizing parallelism for training large language models.

## Conclusion

In this work, the authors implement model parallelism to train transformer models up to 8.3 billion parameters on 512 NVIDIA V100 GPUs, achieving 15.1 PetaFLOPs. They emphasize the importance of layer normalization in BERT-like models for scaling accuracy and set new SOTA results on WikiText103, LAMBADA, and RACE datasets. The code is open-sourced for future research. Future directions include scaling to larger models, optimizing memory, exploring hybrid parallelism, and applying knowledge distillation to train smaller models.
