# GPT3

## Introduction

In recent years, there has been a trend on using pre-trained language models in NLP systems applied to various usecases. However, even though the architecture for this was task-agnostic, there was a need for task-specific datasets and fine-tuning for better results. This is not ideal for multiple reasons such as practicallity, correlations in training data, and a lack of adaptability.

An approach to overcome this is metalearning, or in-context learning as a form of task specification. This seems promising, but results show it being inferiror to fine-tuning. Another interesting approach in this realm is transformer language models with increasingly larger parameters and better results across multiple benchmarks. The mpaper focuses on a 175 billion parameters autoregressive language model called GPT3 and aims to measure its in-context learning abilities, which are evaluated over a lot of NLP datasets under different conditions. The results appear to be promising across multiple benchmarks, while displaying a study of data contamination as well. It also contrasts smaller models across the same conditions and benchmarks, and their respective performances.

## Approach

GPT-3 is evaluated on a variety of settings, namely:

- fine-tuning: most common approach in recent years, requires lot of labelled exams, poor generalization capabilities, not ideal for task-agnostic performance
- few-shot: model given a few demonstrations of the task at inference time without any weight updates. requires lesser task-specific data, though results not as good as SOTA fine-tuned models.
- one-shot: using only one demonstration along with a natural language description of task. aims to represent how tasks are normally communicated to humans (given one example to work with)
- zero-shot: similar to one shot, excluding the demonstration. this is the easiest to do, avoids spurious correlations, is robust. though, this is challenging because it could be difficult to understand a task without any examples.

The model and architecture used is similar to GPT-2 settings. Different sizes of the model are trained for evaluation purposes with different architectural specifications. The dataset used is a filtered, deduplicated version on the Common Crawl dataset with high quality references corpora. For the training, model parallelism was used with a larger batch size and smaller learning rate on V100 GPUs. Different evaluation metrics were identified depending on the nature of the task such as binary classification using semantically meaningful names followed by a multiple choice approach, and so on.

## Results

The above models are evaluated on nine categories of datasets representing roughly similar tasks.

1. Language Modelling, Cloze and Completion Tasks - Tested on traditional tasks of language modelling along with other related tasks. The largest model settings sets a new SOTA benchmark with the few-shot specifications, while the others produce comparable results. This was tested against different datasets in HellaSwag, StoryCloze and LAMBADA w/ the GPT3 few-shot specification emerging as the best setting against SOTA results.
2. Closed Book Question Answering - Tested on the ability to answer questions about factual knowledge. Again, the few-shot specification produces the best results for evaluation performed on TriviQA, Natural Questions and WebQuestions, falling a little short of existing SOTA results, but still being comparable.
3. Translation - GPT-3 had a multi-linguual training dataset with almost 7% of the dataset comprising of non-english data. Once again few-shot GPT-3 outerperforms unsupervised NMT work while providing comparable results to SOTA (supervised) translation.
4. Winograd-Style Tasks - This task involves determining which word a pronoun refers to when the pronoun is grammatically ambiguous. Current fine-tuned lanaguage models display near-human performance on the original Winograd dataset, but fall short on the more difficult reason. Suprisingly, different settings of GPT-3 achieve similar results showing no clear advantage of in-context learning in this task on the easier Winograd dataset, while few-shot performs much better than the others in the harder variant.
5. Common Sense Reasoning - This task tries to capture physical or scientific reasoning, and is evaluated on the PIQA, ARC and OpenBookQA datasets. Few-shot ones again displays the best results, but is far away from fine-tuned SOTA results for ARC and OpenBookQA.
6. Reading Comprehension - This tasks tries to evalute reading comprehsnion on five different datasets spanning different question settings. It performs comparabily on most of the datasets, but falls short of SOTA, with the largest disparity in results on the RACED dataset where it is almost 45% behind.
7. SuperGLUE - This is a benchmark consisting a standardized collection of datasets to evaluated NLP tasks in a more systematic way. GPT-3 achieves near-SOTA performance in one-shot and few-shot settings while falling behind on WiC (comparing two sentence or snippets), RTE and CB (similar format). Though, it still outperforms a fine-tuned BERT-large on half the tasks. Another observation is that SuperGLUE score improve for larger model sizes and more examples used for in-context learning.
8. Natural Language Inference - This task aims to understand the relationship between two sentences on the RTE and ANLI datasets. Results show that this is a very difficult task for language models, with the largest, few-shot providing the best results, but far away from the desired level.
9. Synthetic and Qualitiative Tasks - This aims to evaluate the on-thefly computation reasoning, or the adaptibility to an unusal task. This was evaluated on different tasks such as arithmetic, word scrambling, SAT analogies, news article generation and correcting english-grammar. Results show reasonable performance for moderately complex arithmetic across all settings, mediocre performance in word scrambling (specifically for reversed words), excellent results in SAT analogies (beating human average score) by 5%, decent results for news generation with the alrgest model performing the best and average results in correct grammar.

## Measuring and Preventing Memorization of Benchmarks

One concern while evaluating benchmarks is the possibility of the model being trained on a benchmark test set. Evaluation shows that overlapped data does not significantly impact reported results due to a small percent of contaminated data. The evaluation was done using a set of "clean" benchmarks where potentially overlapped data is stripped from the original benchmark and evaluated on both the versions. Analysing the results, we notice a major change in results only for very few categories (DROP and Reversed Words) while others display almost no change in results for both versions. Although some of the tasks identified a large possibilty of contaminated data (almost 90% in reading comprehension), it did not affect the experiment in a major way as the model consistenly inferred the background data in most cases.

## Limitations

GPT-3 displays strong quantitative and qualitative improvements over its predecessor in GPT-2 but still displays several weaknesses in text synthesis and some NLP tasks. In text synthesis, it loses coherence over sufficiently long passages, struggles with common sense physics, and does not show significantly better performance on few-shot learning for some tasks compared to zero-shot/one-shot approaches. Additionally, GPT-3's autoregressive nature limits its performance on tasks requiring bidirectional context or fine-grained comparisons, such as reading comprehension or sentence entailment tasks. The model also faces challenges in sample efficiency during pre-training and ambiguity in whether few-shot learning truly adapts to new tasks or simply recognizes patterns from prior training.

## Broader Impacts

While discussing broader impacts, we observe that GPT-3 has the potential to advance both beneficial and harmful applications of language models. On the positive side, it can improve tasks like code auto-completion, grammar assistance, and question answering. However, its ability to generate high-quality synthetic text raises concerns about misuse, such as spreading misinformation, spam, or phishing. The model also reflects biases present in its training data, particularly around gender, race, and religion, which can perpetuate harmful stereotypes. Addressing these biases and ensuring fair representation remains a critical challenge. Additionally, the energy consumption of training large models like GPT-3 is significant, though post-training inference is relatively efficient. Future work should focus on mitigating biases, improving energy efficiency, and developing safeguards against misuse.

## Related Work

We see that GPT-3 builds on a long line of research in scaling language models, with prior work exploring larger parameter counts, increased computation, and novel architectures like mixture-of-experts. Efforts to improve sample efficiency, such as distillation and multi-task learning, are complementary to GPT-3's approach. The model also draws inspiration from meta-learning and few-shot learning techniques, though it differs in its reliance on in-context learning without weight updates. Recent advances in algorithmic innovation, such as bidirectional architectures and denoising objectives, could further enhance GPT-3's performance, particularly in fine-tuning scenarios.

## Conclusion

In conclusion, GPT-3 demonstrates strong performance across a wide range of NLP tasks in zero-shot, one-shot, and few-shot settings, often approaching the capabilities of fine-tuned models. Its ability to generate high-quality text and adapt to new tasks on-the-fly highlights the potential of large-scale language models for building general-purpose language systems. However, challenges remain in addressing limitations such as coherence over long passages, common sense reasoning, and biases in the model.

# Chinchilla Scaling Law

## Introduction

Recently a lot of large language models have been introduces with the largest ones having over 500 billion parameters. These large transformers have demonstrated great performance across multiple benchmarks under different settings such as zero-shot, few-shot and fine-tuning. However, they require a substantial amount of compute and energy const for training these models, increasing with model size. Previous work has shown a relationship between model size and performance resulting in a need for larger models for better performance across benchmarks.

## Related Work

Large language models have been becoming really popular in the previous few years, and the largest ones with billions of parameters have been responsible for achieveing SOTA results across multiple language modelling tasks. However, this still has its own drawbacks mainly in the form of computational requirements and a need for higher quality training data. As a result, it has become important to understand how exactly these models scale with their properties, such as the relation between model siez and loss, optimizal model size to train given a specifci budget, etc. Another common concern is estimating hyperparameters for these models, in particular the impact of other parameters besides model size and number of training tokens on the performance of the model. Finally, there have been alternatives to traditional dense transformers, and work done across transformers such as GLaM model shows that their performance may be more dependent on the size of the training data than actually imagined.

## Estimating the optimal parameters/training tokens allocation

The main question the research tries to answer is - "Given a fixed FLOPs budget, how should one trade-off model size and number of training tokens?". For this, three different approachers are considered. The first approach aims to fix the model size and vary the number of training steps. On investigating the results with power laws, and further analysis enables the prediction of optimal parameters and training tokens effectively. The second approach involves with varying the model size for a fixed set of different training FLOP counts. For each FLOP budget the final lost is plotted against the parameter count and analyzed. From this, a parabola is fitted to each IsoFLOP to estimate the model size at which minimum loss is achieved, thereby achevining a loss-optimal model size and number of trianing tokens. The last approach involves fitting a parametric loss function, where the first term captures the ideal generative process loss, the second term accounts for the transformer's underperformance compared to the ideal process, and the third term reflects the finite optimization steps during training. All three approaches suggest that model size and training tokens should scale proportionally with compute budget, challenging previous assumptions that model size should scale faster than data.

## Chinchilla

Based on the scaling analysis, a 70-billion-parameter model named Chinchilla was trained on 1.4 trillion tokens, using the same compute budget as Gopher (280B parameters). Chinchilla outperformed Gopher and other large models across nearly all evaluation tasks, including language modeling, reading comprehension, and common sense reasoning. Despite being smaller, Chinchilla achieved higher accuracy on benchmarks like MMLU and BIG-bench, demonstrating that smaller models trained on more data can outperform larger models with the same compute budget. Chinchilla also showed improvements in reducing gender bias and toxicity, though challenges remain in fully mitigating these issues.

## Discussion and Conclusion

The trend of scaling up model size without proportionally increasing training data has led to suboptimal performance given the compute budget. The analysis and results from Chinchilla suggest that smaller models trained on more data can achieve better performance, challenging the prevailing focus on ever-larger models. However, scaling datasets introduces challenges, including ensuring data quality, addressing ethical concerns, and mitigating biases. Future work should focus on responsibly collecting larger, high-quality datasets and exploring the trade-offs between model size and data scaling across different modalities. This research highlights the importance of optimizing both model size and training data to achieve the best performance within a given compute budget.
