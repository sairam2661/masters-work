# Deep Compression

## Introduction

- Deep neural networks have evoled to the SOTA techniques for CV tasks. They are very powerful, but are also very large and computational expensive. (Example - AlexNet Caffemodel, VGG-16, etc.)
- This makes it very hard to deploy them on mobile systems.
- Issues?
  - Many mobile-first companies are very sensitive to the siez of their binary files.
  - Energy consumption. Runing large models requires a lot of memory bandwidth to get the weights and do the computations. Mobile batteries are not very powerful.
    - An interesting point here is that the energy consumption is dominated by memory accesses.
- Can we reduce the storage and energy required to run inference on such large networks, so that we can deploy them on mobile devices? This is the question that the paper tries to answer.
- The paper presents "deep compression", a three stage pipeline that reduces the storage required by neural networks while preserving their original accuracy. The three stages are broadly,

1. Pruning: Remove redundant connections from the network.
2. Quantization: Quantize the weights so that multiple connections share the same weight. As a result only the effective weights and indices need to be stored
3. Huffman Coding: We have a biased distribution of weights now. We can use Huffman coding to encode this, and use it to our advantage.

## Network Pruning

- Widely used previous to compress CNN models.
- Previously, it was mainly used for reducing network complexity, overfitting and pruning in CNN models without reducing accuracy.
- How does it work?
  - Start by learning the connectivity of the network.
  - Prune the small weight connections (below a threshold)
  - Retrain the network to learn final weights for remaining sparse connections
  - This helped reduce the number of parameters by 9x for AlexNet and 13x for VGG16.
  - The results are stored using a compressed sparse row or a compressed sparse column format.
- For furhter compression, the index difference is encoded and stored instead of absolute positions.

## Trained Quantization and Weight Sharing

- Now, the number of bits required to represent each weight is reduced.
- This can be done by enabling sharing of effective weights between multiple connections, and then fine-tuning those shared weights.

### Weight Sharing

- The paper uses a k-means clustering algorithm to group similar weights together.
- The weights are then quantized to the nearest centroid.
- However, it differs from HashNet in that the weight sharing is determined after a network is fully trained, so that the shared weights are able to resemble the original network.

### Initializaiton of Shared Weights

- The initialization impaces the quality of clustering, thereby affecting the networks prediction accuracy. To examine different initialization methods, three approaches are considered:
  - Forgy(random) - Randomly select k samples from the dataset and use them as the initial centroids.
  - Density-based - This method aims to make the centroides denser in the region where the data points are more concentrated.
  - Linear - this linearly spaces the centroids between the minimum and maximum values of the weights.
- The observation from the experiments is that large weights play an important role, but there are few of them in our cases. As a result, the Forgy and Density-based initialization have a large absolute value which results in a poor representation of the lesser large weights.
- As a result, linear initalizaiton that does not suffer from this problem produces better results.

### Feed-Forward and Back-Propagation

- The centroids of the 1-D k-means clustering are the shared weights.
- There is one level of indirection during feed-forward and back-propagation phase to access the shared weights.
- An index into the shared weight table is stored for each connection in the network.

## Huffman Coding

- This is an optimal prefix code, commonly used for lossless data compression
- It uses variable-length codewards to represent the symbols in the input data.
- Essentially, the more frequent a symbol is, the shorter its codeword.
- The quantized weights have a biased distribution, which can be exploited by Huffman coding.
- Experimentally, using Huffman coding saves 20-30% of the storage required for the quantized weights.

## Experiments

- Four networks based on MNIST and ImageNet datasets are used for the experiments.
- The training is performed with the Caffe framework.
- The paper uses LeNet-5, LeNet-300-100, AlexNet and VGG-16 for the experiments.

### LeNet-300-100 and LeNet-5 on MNIST

- 40x Compression Rate with no loss in accuracy (32x from pruning and quantization, totally 40x when including Huffman coding)

### AlexNet on ImageNet

- 35x Compression with no loss in accuracy (27x from pruning and quantization, totally 35x when including Huffman coding)

### VGG-16 on ImageNet

- Larger than AlexNet!
- 49x compression with ~0.4% loss in accuracy (31x from pruning and quantization, totally 49x when including Huffman coding)
- Super good results, especially when considering real time image processing, and for fast object detection algorithms

## Discussion

### Pruning and Quantization Working Together

- When working indivually, the accuracy of the network drops significantly after the compression.
- However, when used together they complement each other and end up preserving accuracy.
- Hence, we can takeaway that pruning works well with quantization, as pruned network have much lower weights that need to be quantized.

### Centroid Initialization

- We see that linear intialization outperforms forgy and density-based intialization. (in almost all cases)
- This helps in maining the role of the large weights in the network.

### Speedup and Energy Efficency

- Our focus is extremely latency-focused applications that run on mobile devices, and need real-time inference.
- The results show that the compressed networks are faster and more energy efficient than the original networks.
- Experimentally we see a 4.2x speedup on mobile GPUs, and a 3.4x speedup on GPUs.
- We also see 3.3x less energy usage on GPUs and 4.2x less on mobile GPUs.

### Ratio of Weights, Index and Codebook

- On pruning, we get a sparse weight matrix, meaning we need extra space to store the indexes of non-zero elements.
- The quantization step reduces the number of bits required to store the weights.
- The overhead of the codebook (shared weights) is very small.

## Related Work

- Several studies have explored reducing the precision of network parameters to save memory. Fixed-point implementations and ternary-weight networks have been proposed to minimize redundancy while maintaining accuracy.
- low-rank approximations have been applied to neural network parameters, preserving accuracy within 1% of the original model while reducing complexity.
- HashedNets and vector quantization compressed model weights by clustering parameters into buckets, though some methods suffered from accuracy loss.
- Techniques like Optimal Brain Damage and recent pruning methods significantly reduced network parameters without major accuracy loss, improving efficiency.

### Future Work

- pruned network has been benchmarked, but quantized network with weight sharing needs to be benchmarked still (due to lack of hardware support)

### Conclusion

- The paper presents a three-stage pipeline for compressing neural networks.
- The pipeline is able to compress the networks by 35-49x with no loss in accuracy.
- The compressed networks are faster and more energy efficient than the original networks.
- The paper also shows that pruning and quantization work well together, and that linear initialization is better than forgy and density-based initialization.
- As a result, deep neural networks are more energy efficient to run on mobile devices.

# Quantization Survey

## Introduction

- over the years, accuracy of Neural Networks has been consistenly going up for a lot of problems. though, this is mainly achieved by over-parametrized models.
- due to this, there has been a high increase in the size of these NN models, making it hard to deploy them on resource constrained environments.
- this makes it difficult for us to get pervasive deep learning
- many applications such as real-time healthcare monitoring, autonomous driving, require it.
- as a result, there has been a lot of work being done to address different areas of this broad problems
  - Designing efficient neural network model architectures - AutoML, NAS. These aim to find in an automated way, the best architecture for a given problem.
  - Co-designing neural network models and hardware accelerators - This aims to design hardware accelerators that are optimized for neural network models. This is especially good because the overhead of a NN component is hardware dependent.
  - Pruning - This aims to remove redundant connections from the network, in a structured/unstructured way.
  - Knowledge Distillation - This aims to train a smaller network to mimic the behavior of a larger network.
  - Quantization - This aims to reduce the precision of the weights and activations in the network, to reduce the memory footprint and improve latency.
  - Quantization and Neuroscience - This aims to understand the impact of quantization on the biological plausibility of neural networks.
- This paper focuses on the quantization aspect of the problem.

## General History of Quantization

- Quantization is a technique that maps continuous real-valued numbers to a discrete set of numbers.
- This has always been around, such as rounding and truncation.
- In the context of deep learning, quantization has been used to reduce the memory footprint and improve latency.
  - The idea is that inference and training of NN are computationally intensive.
  - So, we need to efficiently represent the numerical values in the network.
  - In recent developments in quantization in NN applications, there have been different error metrics that have been used to measure the impact of quantization on the network.
  - Have to consider many things when quantizing in a NN.

## Basics of Quantization

### Problem Setup and Notations

- The problem is to map a real-valued number to a discrete set of numbers.

### Uniform Quantization

- Need to quantize NN wieghts and activations to a finite set of values.
- We do this by uniform quantization, where we divide the range of the real numbers into equal intervals.
- We also have non-uniform quantization methods whose values are not always uniformly spaced.
- We recover values using dequantization, which maps the quantized values back to the real values.

### Symmetric and Asymmetric Quantization

- In uniform quantization, the sclaing factor is an important factor to consider.
- This is kind of dependent on the clipping range of the weights.
- If we were to simply use the mix/max of a single as the clipping range, we would end up in an asymmetric quantization scheme.
  - This is mainly cause the clipping range is not necessarily symmetric.
- Instead, if we were to use a clipping range of alpha = -beta, we would end up in a symmetric quantization scheme.
- Assymetric quantization results in a tighter range of values when compared to symmetric.
- This is kinda important when the target weights or activations are imbalanced.
- So in general, when the rage could be skewed and not symetric, assymetric quantization is preferred. Though, if an easier implementation is needed, symmetric quantization can be used.

### Static vs Dynamic Quantization (Range Calibration Algorithms)

- The previous methods deal with determining the clipping range. Another important factor to consider is when the clipping range is determined.
- If the range is computed statically for weights, it is called static quantization.
- Instead, if it is dynamically calcualted for each activation map during runtime, it is called dynamic quantization.
- Static quantization is easier to implement, but dynamic quantization is more accurate. Though, this could be computationally expensive as there is a lot of overhead in determining the range for each activation map.
- In Static Quantization, we pre-calculate the clipping range. As a result there is no computational overhead, but we get lower accuracy.

### Quantization Granularity

- In most CV tasks, the activation input to a layer is convolved with different convolutional filters. Each of these filters can have a different range of values. Hence, another factor to consider is the granularity of quantization.
  - Layerwise Quantization - the clipping range is determind by considering all the weights in the convolutional filters of a layer. Simple to implement, but sub-optimal accuracy.
  - Groupwise Quantization - the clipping range is determined by grouping multiple channels inside a layer. This is more accurate than layerwise quantization. Though, we need to consider the extra cost of different scaling factors.
  - Channelwise Quantization - used a fixed range for the clipping range for each convolutional filter independent of the other channels. Better quantization resolution, often results in better accuracy as well.
  - Sub-channelwise Quantization - the clipping range is determined for eany group of parameters a convolutional filter. Previous approach taken to the extreme. This is the most accurate, but also the most computationally expensive.

### Non-Unifrom Quantization

- the idea is we space quantization steps as well as the quantization levels non uniformly.
- this may achieve higher accuracy for a fixde bit-width, as one could better capture the distrubitons by focusing more on important value regions or finding appropriate dynamic ranges.
- many non-uniform quantization methods have been deesgined for bell-shaped distributions of the weight and activations.
- the idea is to use a logarithmic distribution of the quantization levels, as it is more suitable for the bell-shaped distributions.
- additionally, clustering can also be useful in non-uniform quantization as it would help in grouping similar values together.

### Fine-tuning Methods

- often, we need to adjust the parameters in the NN after quantization
- we can do this by retraining the model (quantization awarwe training), or without retraining the model (post-training quantization)
- retraining the model is more accurate, but also computationally expensive.
- post-training quantization is faster, but also less accurate.

#### Quantization Aware Training

- given a trained model, quantization could introduce perturbations in the weights and activations.
- as a result, the model needs to be retrained to adjust to these perturbations.
- this is done by adding a quantization error term to the loss function.
- this would converge to a model that is more robust to quantization.
- the model parameters are updated using the gradients of the quantization error term. (straight through estimator)
- this is computationally expensive, but also more accurate.
- we also have other methods such as combinatorial optimization, target propagation, etc.
- the idea is to minimize the quantization error term, while also minimizing the original loss function.
- overall, if a quantized model is going to be deplyoed for an extended period of time, and if efficieny and accuracy are especially important,then retraining the model is the way to go.
- this is especially important for real-time applications, where the model is going to be used for a long time.
- but what if the model had a short life span?

#### Post-training Quantization

- performs the quanztization after the model has been trained, without and fine-tuning.
- this is a faster method, but also less accurate.
- ovehead is very low, and often negligble
- can be applied in situations where data is limited or unlabeled.
- this is especially useful in cases where the model is going to be used for a short period of time.
- we have multiple approaches to mitigate the accuracy degradation of PTQ such as ACIQ, OMSE, AdaRound, etc.

#### Zero-shot Quantization

- quantization without access to the original training data, making it useful for scenarios where data privacy or availability is a concern.
- involves generating synthetic data or leveraging model statistics to calibrate and fine-tune the quantized model.

### Stochastic Quantization

- during inference, the quantization scheme is deterministic
- some works have explored the idea of using stochastic quantization during training
- the idea is that this may allow a NN to explore more, when compared to deterministic quantization
- Example - QuantNoise. This is a stochastic quantization method that uses a noise term to quantize the weights and activations.
- A major challenge here is the overhead of creating random numbers for every weight update.

## Quantization below 8 Bits

- most of the works have focused on quantization to 8 bits

### Simultated and Integer-only Quantization

- in simulated quantization, the quantized model parameters are stored in low-precision, but the operations are carried out with floating point arithmetic.
- hence, the quantizedf parameters need to be dequantized before the floating point operations are carried out.
- however, in integer-only quantizzation all the oeprations are performed using low precision integer arithmetic.
- this perims in the entire inference being carried out with efficient integer arithmeic without the need for any floating point deuqntization of parameters or activations.
- Dyadic quantization is a method that uses integer-only quantization, where the scaling is performed with dyadic numbers.
- As a result integer-only and dyadic quantization are preferred over simulated quantization.
- Howeverm fake quantization can be used for problems that are bandwidth bound rather than compute bound (such as reccomendation systems)

### Mixed Precision Quantization

- Has been an aeffective and hardware efficient way to quantize NNs.
- The idea is to use different precisions for different layers in the network.
- The layers are grouped into sensitive/insestive layers, and the precisions are assigned accordingly.
- This is especially useful when the network has a lot of layers, and the sensitivity of the layers is different.
- As a result, one can minimize accuracy degration and still benefit from the memory and latency improvements.

### Hardware Aware Quantization

- One of the aims of quantization is to improve the inference latency.
- However, not all hardware provide the same speed up after a ceratin operation is quantized.
- Hence, we need to focus on hardware aware quantization where the quantization scheme is chosen based on the hardware.

### Distillation-Assisted Quantization

- the idea is to incorporate model distillation to boost quantization accuracy.
- a model with a higher accuracy is used as a teacher to help the training of a compact student model.
- this is especially useful when the student model is going to be quantized.

### Extreme Quantization

- Binarization, where the quantizde values are constrained to a single bit, drastically reduces the memory requirement (32x) is the most extreme quantization method.
- Binary and ternary operations can be computed efficient with bit-wise arithmetic and can achieve significant accelration over higher precisions
- One thing to remember is that binaration and ternarization methods generally result in sever accuracy degradation, especially for complex tasks.
- However, a lot of work is being done to mitigate this accuracy degradation.
  - Quantization Error Minimization - this aims to minimize the gap between quantized values and the values.
  - Improved Loss Function - this aims to minimize the quantization error term in the loss function.
  - Improved trianing method - this aims to improve the training method to better handle the quantization error term. (such as straight through estimator)

### Vector Quantization

- more of a classical approach
- clusters weights into groups, using centroids as quantized values, reducing model size without significant accuracy loss.
- product quantization is an extension of vector quantization, where the quantization is divided between multiple submatrices and then performed.

## Quantization and Hardware Processors

- is particularly important for edge devices, which often lack floating-point units and have strict power constraints.
- Hardware platforms like ARM Cortex-M and Google Edge TPU benefit significantly from quantization, enabling efficient NN inference on low-power devices.

## Future Directions for Research in Quantization

- Quantization software - there is a need for more efficient quantization software that can be used to quantize models in a more efficient way.
  - recent work has shown that low precision and mixed-precision quantization works well in practice.
  - we need efficient software APIs for lwoer precision quantization.
- Hardware and NN Architecture Co-Design - there is a need for hardware accelerators that are optimized for quantized NNs.
  - this would help in improving the inference latency and energy efficiency.
  - this would also help in improving the accuracy of the quantized models.
- Coupled Compression Methods - there is a need for coupled compression methods that can be used to compress NNs in a more efficient way.
  - this would help in reducing the memory footprint and improving the inference latency.
  - this would also help in improving the accuracy of the compressed models.
- Quantized Training - need to accelerate NN training with half-precision
  - enabled the use of much faster and power-efficient logic for training
  - though, this has been difficult to push further down to INT8 precision training.

## Conclusion

- Quantization is a technique that maps continuous real-valued numbers to a discrete set of numbers.
- This has been used to reduce the memory footprint and improve latency in NNs.
- There are many factors to consider when quantizing a NN, such as the quantization granularity, the quantization error metric, the quantization method, etc.
- There are many methods that have been proposed to mitigate the accuracy degradation of quantization, such as quantization aware training, post-training quantization, etc.
- There are many future directions for research in quantization, such as quantization software, hardware and NN architecture co-design, coupled compression methods, quantized training, etc.
- Overall, quantization is an important technique that can be used to reduce the memory footprint and improve latency in NNs.
