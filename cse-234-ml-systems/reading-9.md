# FlashAttention

## Introduction

The paper introduces FlashAttention which aims to address the inefficiencies of Transformer models, mainly their quadratic time and memory complexity in sequence length, which makes them slow and memory-hungry on long sequences. Other works, such as different approximate attention methods try to reduce compute and memory requirements, they do not achieve wall-clock speedup due to focusing on FLOP reduction and ignoring memory access overhead. This paper shows that FlashAttention succeeds in introducing an IO-aware exact attention algorithm that reduces memory reads/writes between GPU high bandwidth memory (HBM) and on-chip SRAM using tiling. This approach helps to speed up attention computation but also enables longer context in transformers. As a result we get better model quality and are able to enable new capabilities like better-than-chance performance on long-sequence tasks.

The paper also mentions that modern GPUs are increasingly bottlenecked by memory accesses, making IO-aware algorithms crucial for performance. As a result, FlashAttention uses tiling and recomputation to avoid storing large intermediate matrices, and endes up with fewer HBM accesses and optimal performance for a range of SRAM sizes. The algorithm is extended to block-sparse attention, providing even faster approximate attention. Finally, results show significant speedups in training transformers, with improvements in model quality and the ability to handle longer sequences.

## Background

This section provides an overview of the performance characteristics of deep learning operations on modern GPUs, mainly focusing on the memory hierarchy and execution model. GPUs have a memory hierarchy with fast on-chip SRAM and slower HBM, and operations are increasingly bottlenecked by memory accesses. We also learn about the difference between compute-bound and memory-bound operations, with attention being a memory-bound operation due to its quadratic complexity in sequence length. Additionally, kernel fusion is introduced as a common technique to accelerate memory-bound operations by reducing redundant memory accesses.

We also learn about the normal implementaiton of attention, learning about its inefficiency due to materializing large intermediate matrices in HBM. This leads to a large number of memory accesses, which slows down wall-clock time. The focus on IO complexity and memory hierarchy is crucial for understanding how FlashAttention actually works.

## Algorithm, Analysis, Extensions

FlashAttention is an efficient attention algorithm that uses tiling and recomputation to reduce HBM accesses. The algorithm splits input matrices into blocks, loads them into SRAM, and computes attention incrementally. Tiling allows the softmax computation to be performed in blocks, while recomputation avoids storing large intermediate matrices for the backward pass. This algorithm is implemented in CUDA, fusing all attention operations into a single GPU kernel, which reduces memory access and speeds up computation.

On analyzing the IO complexity of FlashAttention, we see that it requires fewer HBM accesses compared to standard attention. The paper also proves that FlashAttention is optimal for a range of SRAM sizes and provides a lower bound on the number of HBM accesses for exact attention algorithms. The algorithm is extended to block-sparse attention, which further reduces IO complexity by a factor proportional to the sparsity ratio. This extension enables FlashAttention to scale with even longer sequences, making it faster than existing approximate attention methods.

## Experiments

The experiments demonstrate the effectiveness of FlashAttention in speeding up model training and improving model quality. FlashAttention achieves a 15% speedup in training BERT-large compared to the MLPerf 1.1 record, and up to 3x speedup for GPT-2. It also improves model quality by enabling longer context, resulting in better results on GPT-2 and significant improvements in long-document classification tasks. FlashAttention is also the first transformer to achieve better than chance performance on the Path-X and Path-256 challenges, demonstrating its ability to handle extremely long sequences.

Other results show that FlashAttention is up to 3x faster than standard attention for common sequence lengths and scales linearly with sequence length. In fact, block-sparse FlashAttention is even faster, outperforming all existing approximate attention methods. The experiments also demonstrate that FlashAttention is more memory-efficient, with a memory footprint that scales linearly with sequence length, making it suitable for long-sequence tasks where memory is a critical constraint.

## Limitations

The paper also acknowledges some limitations of FlashAttention. One major limitation is the need to write new CUDA kernels for each attention implementation, which requires significant engineering effort and may not be transferable across GPU architectures. The authors suggest the need for a high-level language that can compile to IO-aware CUDA implementations. This would end up making it easier to implement and optimize attention algorithms without requiring low-level CUDA programming.

Another limitation is that it is currently optimized for single-GPU attention computation. As a result, there still needs to be exploration done for multi-GPU IO-aware methods, where attention computation is parallelized across multiple GPUs. This would require accounting for data transfer between GPUs, adding another layer of complexity to the IO analysis.

## Conclusion

FlashAttention optimizes attention computation by reducing memory bottlenecks through tiling and recomputation, achieving significant speedups while maintaining accuracy. It enables efficient training of large models and handling longer sequences. Despite challenges in CUDA implementation and multi-GPU scalability, FlashAttention highlights the importance of IO-aware design for future Transformer optimizations.

# PagedAttention

## Introduction

LLMs have started becoming increasingly popular, but delivering them efficiently is challenging due to high computational costs. A key factor in the cost is the memory required to store the intermediate states (KV cache) during the generation process. Hence, it is important to increase the throughput of these LLM serving systems, thereby reducing the cost per request. The paper introduces introduces PagedAttention, a novel attention algorithm designed to address memory management bottlenecks in LLM serving.

## Background

Language models predict the probability of a sequence of tokens, and transformer models have become the standard when it comes to modern development. A component of these, self-attention layers have become by far, the most crucial components. Normally, LLMs are deployed as services that generate output tokens based on some given input prompts. This generation process involves caching key and value vectors (KV cache) for previously generated tokens. The computation involves a prompt phase and an autoregressive generation phase. If we go with batching requests, we are able to improve efficiency. But this leads to some challenges - requests can arrive at different times and have varying lengths, this needs to be handled. As a result, fine-grained batching techniques have been proposed to address these challenges.

## Memory Challenges in LLM Serving

Even with fine-grained batching, memory capacity, particularly for the KV cache, limits the number of requests that can be processed. The KV cache size grows rapidly, and inefficient memory management worsens the problem. LLM services also offer different decoding algorithms that further complicate this memory management. These requests also have variable input and output lengths as we saw before, making memory scheduling difficult. Existing systems allocate contiguous memory chunks for the KV cache based on the maximum possible sequence length, leading to significant memory waste due to reserved slots, internal fragmentation, and external fragmentation. This pre-allocation also prevents effective memory sharing.

## Methodology

To overcome the memory challenges, the paper introduces PagedAttention, a novel attention based approach to address these challenges and build an LLM serving engine called vLLM. vLLM uses a centralized scheduler and a KV cache manager that efficiently manages the KV cache in a paged fashion, enabled by PagedAttention. We see that this enables efficient KV cache management by storing and accessing key-value pairs in a paged manner, reducing fragmentation and memory overhead. This approach powers vLLM, a serving engine with a centralized scheduler that optimizes memory usage and improves LLM inference efficiency.

## Implementation

The implementation details are mainly focused on the system architecture and the KV cache management. vLLM employs a distributed system with multiple GPU workers coordinated by a centralized scheduler. The centralized scheduler efficiently distributes workloads across multiple GPU workers, ensuring optimal resource utilization. The KV cache manager is a core component, responsible for allocating and deallocating KV blocks, similar to how a virtual memory manager handles pages. Hence, the KV cache manager dynamically manages KV blocks, minimizing fragmentation and maximizing memory efficiency.

## Evaluation

The paper evaluates vLLM's performance and compares it to existing LLM serving systems. The metrics that are focused on are throughput, latency, and memory usage. From the results, we see that the vLLM significantly improves throughput and reduces memory waste compared to SOTA systems like FasterTransformer and Orca. The improvements are more visible with longer sequences, larger models, and complex decoding algorithms. These results highlight vLLM’s ability to handle high-throughput LLM inference efficiently. Its optimizations make it well-suited for serving large models with minimal memory overhead.

## Discussion and Related Work

The discussion section provides further insights into the advantages and limitations of PagedAttention and vLLM. It also addresses related work in LLM serving, attention mechanisms, and memory management. The authors conclude by summarizing the key contributions of their work, emphasizing that PagedAttention and vLLM effectively address the memory bottleneck in LLM serving. They highlight that vLLM achieves near-zero waste in KV cache memory and enables flexible sharing of the KV cache, leading to substantial throughput improvements.
