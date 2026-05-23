# Interpreting CLIP with Hierarchical Sparse Autoencoders

**Vladimir Zaigrajew**¹ **Hubert Baniecki**¹ ² **Przemyslaw Biecek**¹ ²

¹ Warsaw University of Technology, Warsaw, Poland
² University of Warsaw, Warsaw, Poland.
Correspondence to: Vladimir Zaigrajew <vladimir.zaigrajew.dokt@pw.edu.pl>

## Abstract

Sparse autoencoders (SAEs) are useful for detecting and steering interpretable features in neural networks, with particular potential for understanding complex multimodal representations. Given their ability to uncover interpretable features, SAEs are particularly valuable for analyzing vision-language models (e.g., CLIP and SigLIP), which are fundamental building blocks in modern large-scale systems yet remain challenging to interpret and control. However, current SAE methods are limited by optimizing both reconstruction quality and sparsity simultaneously, as they rely on either activation suppression or rigid sparsity constraints. To this end, we introduce Matryoshka SAE (MSAE), a new architecture that learns hierarchical representations at multiple granularities simultaneously, enabling a direct optimization of both metrics without compromise. MSAE establishes a state-of-the-art Pareto frontier between reconstruction quality and sparsity for CLIP, achieving 0.99 cosine similarity and less than 0.1 fraction of variance unexplained while maintaining 80% sparsity. Finally, we demonstrate the utility of MSAE as a tool for interpreting and controlling CLIP by extracting over 120 semantic concepts from its representation to perform concept-based similarity search and bias analysis in downstream tasks like CelebA. We make the codebase available at https://github.com/WolodjaZ/MSAE.

## 1. Introduction

Vision-language models, particularly contrastive language-image pre-training (CLIP, Radford et al., 2021; Cherti et al., 2023), revolutionize multimodal understanding by learning robust representations that bridge visual and textual information. Through contrastive learning on massive datasets, CLIP and its less adopted successor SigLIP (Zhai et al., 2023) demonstrate remarkable capabilities that extend far beyond their primary objective of cross-modal similarity search. Its representation is a foundational component in text-to-image generation models like Stable Diffusion (Podell et al., 2024) and serves as a powerful feature extractor for numerous downstream vision and language tasks (Shen et al., 2022), establishing CLIP as a crucial building block in modern VLMs (Liu et al., 2023; Wang et al., 2024).

Despite CLIP’s widespread adoption, understanding how it processes and represents information remains a challenge. The distributed nature of its learned representations and the complexity of the optimized loss function make it particularly difficult to interpret. Traditional explainability approaches have limited success in addressing this challenge: gradient-based feature attributions (Simonyan, 2013; Shrikumar et al., 2017; Selvaraju et al., 2017; Sundararajan et al., 2017; Abnar & Zuidema, 2020) struggle to provide human-interpretable explanations, perturbation-based approaches (Zeiler & Fergus, 2014; Ribeiro et al., 2016; Lundberg & Lee, 2017; Adebayo et al., 2018; Baniecki et al., 2025) yield inconsistent results, and concept-based methods (Ramaswamy et al., 2023; Oikarinen et al., 2023) are constrained by their reliance on manually curated concept datasets. This interpretability gap hinders our ability to identify and mitigate potential biases or failure modes of CLIP in downstream applications (Biecek & Samek, 2024).

Recent advances in mechanical interpretability (Conmy et al., 2023; Bereska & Gavves, 2024) use sparse autoencoders (SAEs) as a tool for disentangling interpretable features in neural networks (Cunningham et al., 2024). When applied to CLIP’s representation space, SAEs offer the potential to decompose complex, distributed representations into human-interpretable components through self-supervised learning. It eliminates the need for concept datasets and limits predefined concept sets in favor of natural concept emergence. However, training effective SAEs poses unique challenges. The richness of data distribution and high dimensionality of CLIP’s multimodal embedding space require tuning the sparsity-reconstruction trade-off (Bricken et al., 2023; Gao et al., 2025). Furthermore, evaluating SAE effectiveness extends beyond traditional metrics, requiring the discovery of interpretable features that maintain their semantic meaning across both visual and textual modalities.

Current approaches for enforcing sparsity in autoencoders use either L¹ (Bricken et al., 2023) or TopK (Gao et al., 2025) proxy functions, each with significant drawbacks. L¹ regularization results in activation shrinkage, systematically underestimating feature activations and potentially missing subtle but important concepts. TopK enforces a fixed number of active neurons, imposing rigid constraints that may not align with the natural concept density in different regions of CLIP’s embedding space (Gao et al., 2025; Bussmann et al., 2024).

To this end, we propose a hierarchical approach to sparse autoencoders, a new architecture inspired by Matryoshka representation learning (Kusupati et al., 2022), as illustrated in Figure 1. While Matryoshka SAE (MSAE) can be applied to interpret any neural network representation, we demonstrate its utility in CLIP’s complex multimodal embedding space. At its core, MSAE applies TopK operations *h*-times with progressively increasing numbers of *k* neurons, learning representations at *h* granularities simultaneously – from coarse concepts to fine-grained features. By combining reconstruction losses across all granularity levels, MSAE achieves a more flexible and adaptive sparsity pattern. We remove the rigid constraints of simple TopK while avoiding the activation shrinkage problems associated with L¹ regularization, resulting in the state-of-the-art Pareto frontier between the reconstruction quality and sparsity.

### Contributions

*   We introduce a hierarchical SAE architecture that establishes a new leading Pareto frontier between reconstruction quality (0.99 cosine similarity and < 0.1 FVU) and sparsity (~80%), while maintaining computational efficiency comparable to standard SAEs at inference time.
*   We develop a robust methodology for validating discovered concepts in CLIP’s multimodal embedding space, successfully identifying and verifying over 120 interpretable concepts across both image and text domains.
*   Through extensive empirical evaluation on CC3M and ImageNet datasets, we demonstrate progressive recovery capabilities and the effectiveness of hierarchical sparsity thresholds compared to existing approaches.
*   We showcase the practical utility of MSAE in two key applications: concept-based similarity search with controllable concept strength and systematic analysis of gender biases in downstream classification models through SAE activations and concept-level interventions on the CelebA dataset.

## 2. Related Work

### Interpreting CLIP models

CLIP interpretability research follows two main directions: a direct interpretation of CLIP’s behavior and using CLIP to explain other models. Direct interpretation studies focus on understanding CLIP’s components through feature attributions (Joukovsky et al., 2023; Sammani et al., 2024; Zhao et al., 2024), residual transformations (Balasubramanian et al., 2024), attention heads (Gandelsman et al., 2024), and individual neurons (Goh et al., 2021; Li et al., 2022). Li et al. (2022) discovered CLIP’s tendency to focus on image backgrounds through saliency analysis, while Goh et al. (2021) identified CLIP’s multimodal neurons responding consistently to concepts across modalities. For model explanation, CLIP is used to analyze challenging examples (Jain et al., 2023), robustness to distribution shifts (Crabbé et al., 2024), and label individual neurons (Oikarinen & Weng, 2023). In this work, we explore both directions in Section 5 via the detection of semantic concepts learned by CLIP using MSAE (Section 5.1) and the analysis of biases in downstream models built on MSAE-explained CLIP embeddings (Section 5.3).

### Mechanistic interpretability

Mechanistic interpretability seeks to reverse engineer neural networks analogously to decompiling computer programs (Conmy et al., 2023; Bereska & Gavves, 2024). While early approaches focus on generating natural language descriptions of individual neurons (Hernandez et al., 2021; Bills et al., 2023), the polysemantic nature of neural representations makes this challenging. A breakthrough comes with sparse autoencoders (SAEs) (Bricken et al., 2023; Cunningham et al., 2024), which demonstrate the ability to recover monosemantic features. Recent architectural advancements like Gated (Rajamanoharan et al., 2024a) and TopK SAE variants (Gao et al., 2025) improve the sparsity–reconstruction trade-off, enabling successful application to LLMs (Templeton et al., 2024), diffusion models (Surkov et al., 2024), and medical imaging (Abdulaal et al., 2024). Recent work on SAE-based interpretation of CLIP embeddings (Rao et al., 2024) shows promise in extracting interpretable features.

### Concept-based explainability

Concept-based explanations provide interpretability by identifying human-coherent concepts within neural networks’ latent spaces. While early approaches relied on manually curated concept datasets (Kim et al., 2018; Zhou et al., 2018; Bykov et al., 2023), recent work has explored automated concept extraction (Ghorbani et al., 2019; Kopf et al., 2024) and explicit concept learning (Liu et al., 2020; Koh et al., 2020; Espinosa Zarlenga et al., 2022), with successful applications in out-of-distribution detection (Madeira et al., 2023), image generation (Misino et al., 2022), and medicine (Lucieri et al., 2020). However, existing methods often struggle to scale to modern transformer architectures with hundreds of millions of parameters. Our approach addresses this limitation by first training SAE without supervision on concept learning, then efficiently mapping unit-norm decoder columns to defined vocabulary concepts using cosine similarity with CLIP embeddings.

## 3. Matryoshka Sparse Autoencoder

### 3.1. Preliminaries

Sparse autoencoders (SAEs) decompose model activations $x \in \mathbb{R}^n$ into sparse linear combinations of learned directions, aiming for interpretability and monosemanticity. The standard SAE architecture consists of:

$$
z = \text{ReLU}(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}}), \quad \hat{x} = W_{\text{dec}}z + b_{\text{pre}}, \quad (1)
$$

where encoder matrix $W_{\text{enc}} \in \mathbb{R}^{n \times d}$, encoder bias $b_{\text{enc}} \in \mathbb{R}^d$, decoder matrix $W_{\text{dec}} \in \mathbb{R}^{d \times n}$, and preprocessing bias $b_{\text{pre}} \in \mathbb{R}^n$ are the learnable parameters, with $d$ being the dimension of the latent space. The basic reconstruction objective is $L(x) := \|x - \hat{x}\|_2^2$.

Existing approaches established two primary sparsity mechanisms. ReLU SAE (Bricken et al., 2023) uses L¹ regularization with the objective $L(x) := \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$, while TopK SAE (Gao et al., 2025) enforces fixed sparsity through $z = \text{ReLU}(\text{TopK}(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}}))$. However, each approach faces distinct limitations: L¹ regularization causes activation shrinkage (Rajamanoharan et al., 2024a), while TopK imposes rigid sparsity constraints.

A recent effort to address the rigidity of TopK is BatchTopK (Bussmann et al., 2024), which replaces the standard TopK function with BatchTopK within the TopK SAE method. The BatchTopK function treats all batch activations as a single, flattened vector before applying TopK. This allows for a flexible number of active features per sample, with the total number of active features across the batch averaging to *k* × batch size. Although BatchTopK relaxes the fixed sparsity of traditional TopK, it still relies on a predetermined *k* parameter that requires careful tuning and continues to suffer from the potential for certain features to become ‘dead’ or rarely activated if they consistently fall outside the TopK selection.

### 3.2. Matryoshka SAE Architecture

Following Matryoshka representation learning (Kusupati et al., 2022), we propose a SAE architecture that learns representations at multiple granularities simultaneously. Instead of enforcing a single sparsity threshold *k* or using L¹ regularization, our approach applies multiple TopK operations with increasing *k* values, optimizing across all granularity levels. We set *k* values as powers of 2, i.e., $k_i = 2^i$ up to dimension $d$, which provides effective coverage of the representation space while maintaining reasonable computational costs. For a given input $x$, MSAE computes $h$ latent representations during training using a sequence of increasing *k* values $\{k_1, k_2, \dots, k_h\}$ with $k_1 < k_2 < \dots < k_h \le d$:

$$
z_i = \text{ReLU}(\text{TopK}_i(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}})), \quad \hat{x}_i = W_{\text{dec}}z_i + b_{\text{pre}}, \quad L(x) := \sum_{i=1}^h \alpha_i \|x - \hat{x}_i\|_2^2, \quad (2)
$$

where $\alpha_i$ are weighting coefficients for each granularity level. At inference time, we can either apply TopK with any desired granularity or discard it entirely, leaving only ReLU, which allows the model to utilize all neurons it deems essential for reconstruction.

#### Hierarchical learning

The key insight of our approach is that different samples require different levels of sparsity (numbers of concepts) for an optimal representation. By simultaneously optimizing across multiple *k* values, MSAE learns a natural hierarchy of features. Our TopK operations maintain a nested structure where features selected at each level form a subset of those selected at higher *k* values, i.e., $\text{TopK}_1 \subseteq \text{TopK}_2 \subseteq \dots \subseteq \text{TopK}_h$. Such a hierarchical structure ensures coherence between granularity levels, where low *k* values capture coarse, high-level concepts while higher *k* values progressively enable fine-grained feature representation.

#### Sparsity coefficient weighting

We propose and evaluate two strategies for setting the weighting coefficients $\alpha_i$. The uniform weighting (UW) approach sets $\alpha_i = 1$ for all $i$, while the reverse weighting (RW) strategy uses $\alpha_i = h - i + 1$, giving higher weights to lower *k* values. By weighting the loss more heavily for sparser reconstructions, RW encourages the model to learn features that maintain reconstruction quality at lower *k* values. As shown in Table 1, this results in improved sparsity without significant performance degradation as the model learns that sparse representations achieve better loss even with slightly worse reconstruction quality, compared to UW which focuses primarily on reconstruction quality.

### 3.3. Training and Inference

CLIP embeddings exhibit misalignment across modalities, which can impact SAE training convergence and cross-modal transferability. Following (Bhalla et al., 2024), we normalize embeddings to ensure consistent behavior across modalities. We first center embeddings by subtracting the per-modality mean estimated from the training dataset. Next, we scale the centered embeddings by a dataset-computed scaling factor following (Conerly et al., 2024) to obtain $E_{x \in X}[ \|x\|_2 ] = \sqrt{n}$. This scaling ensures that $\lambda$ has consistent effects across different CLIP architectures and modalities. For training, we compute the mean vector and scaling factor from the image modality. During inference on text embeddings, we apply the text-specific mean and scaling factor. Additionally, at inference, we remove TopK constraints from TopK-trained models, allowing the model to adaptively select the number of active features based only on ReLU activation.

## 4. Evaluating MSAE

In this section, we conduct extensive experiments to evaluate MSAE against ReLU and TopK SAEs. We compare the sparsity–fidelity trade-off (Section 4.2), at multiple granularity levels (Section 4.3). We follow with evaluating the semantic quality of learned representations beyond traditional distance metrics (Section 4.4), analyzing decoder orthogonality (Section 4.5), and examining the statistical properties of SAE activation magnitudes (Section 4.6). To verify that MSAE successfully learns hierarchical features, we conduct experiments on the progressive recovery task (Section 4.7). We conclude with an ablation study comparing the influence of different training modalities in Section 4.8. In response to reviewer feedback, we’ve incorporated an analysis of BatchTopK models within Section 4.2. However, given their performance characteristics were not as competitive, we didn’t extend their evaluation to subsequent sections.

### Setup

All SAE models are trained on the CC3M (Sharma et al., 2018) training set with features (post-pooled) from the CLIP ViT-L/14 or ViT-B/16 model. Image modality is evaluated on ImageNet-1k training set (Russakovsky et al., 2015), while text modality is evaluated on the CC3M validation set. Each SAE is trained with expansion rates of 8×, 16× and 32×, effectively scaling the latent layer from 768 to {6144, 12288, 24576} neurons for ViT-L/14, and from 512 to {4096, 8192, 16384} neurons for ViT-B/16. We provide further details on the implementation and hyperparameter settings in Appendix B.

### 4.1. Evaluation Metrics

Here, we briefly define each metric used to evaluate SAE.

*   **L⁰**: Denotes the mean proportion of zero elements in SAE activations.
*   **Fraction of variance unexplained (FVU)**: Also known as Normalized MSE (Gao et al., 2025), measures reconstruction fidelity by normalizing the mean squared reconstruction error $L(x)$ by the mean squared value of the (mean-centered) input.
*   **Explained variance ratio (EVR)**: Is FVU’s complement metric, defined as 1 − FVU.
*   **Linear probing (LP)**: Assesses how well SAE preserves semantic information in the reconstructed embeddings on the downstream task. To evaluate this, we train a linear probe model on ImageNet-1k using CLIP embeddings as a backbone, with AdamW optimizer (lr = 1e-3), ReduceLROnPlateau scheduler, and batch size=256. We measure performance by comparing predictions from original versus reconstructed embeddings using two metrics: Kullback-Leibler divergence (KL) between predicted class distributions and classification accuracy (Acc), where accuracy uses argmax predictions from original embeddings as targets.
*   **Centered kernel nearest neighbor alignment (CKNNA)** (Huh et al., 2024): Measures kernel alignment based on mutual nearest neighbors, providing a quantitative assessment of alignment between SAE activations and input embeddings. A detailed explanation is provided in Appendix E.
*   **Decoder orthogonality (DO)**: Calculates the mean cosine similarity of the lower triangular portion of the SAE decoder, where 0 indicates perfect orthogonality. This metric assesses how orthogonal the monosemantic feature directions are in the decoder.
*   **Number of dead neurons (NDN)**: Is a metric that measures how many neurons remain consistently inactive (zero in the SAE activations layer) across all inputs during training or evaluation, indicating the network’s inability to fully utilize its capacity for learning semantic features.

### 4.2. Sparsity–Fidelity Trade-off

We assess SAE performance using sparsity–fidelity trade-off, measuring sparsity with L⁰ and reconstruction quality with EVR, following previous work. Figure 2 reveals that ReLU SAE shows difficulty balancing performance, achieving either high fidelity with low L⁰ or the opposite, with expansion rate primarily improving sparsity. TopK SAE with higher *k* values achieves better but not ReLU-level fidelity while offering improved sparsity yet consistently suffering from at least 5% of dead neurons (Table 1). While the BatchTopK variant performs similarly to TopK, it exhibits higher fidelity with lesser sparsity when trained on the same *k*. Both variants of MSAE achieve better sparsity than ReLU and better fidelity than TopK or BatchTopK, establishing a superior Pareto frontier while maintaining less than 1% of dead neurons. The RW variant further improves sparsity as expected, with only minor fidelity degradation. Notably, only Matryoshka consistently improves on both metrics with higher expansion rates, while TopK struggles with reconstruction, BatchTopK with sparsity, and ReLU shows improvements only in the highest $\lambda$ at increased expansion rates. As an ablation, we evaluate Cosine Similarity as an alternative reconstruction metric, motivated by observations that SAEs primarily struggle with embedding magnitude reconstruction and CLIP embeddings are commonly L²-normalized. Results in Appendix F.1 show consistent findings, with MSAE showing even clearer advantages through stable, low-variance performance across both modalities.

### 4.3. Ablation: Matryoshka at Lower Granularity Levels

We train both MSAE variants (RW and UW) on two granularities [128, 256] and compare them against TopK with *k* = 128 and *k* = 256 to analyze MSAE behavior at lower granularity levels. Figure 3 shows that Matryoshka achieves similar sparsity to at least the lower TopK variant while maintaining CKNNA and EVR performance comparable to the best TopK variant, and even better with MSAE RW. This demonstrates that even at small granularity, MSAE maintains or improves the Pareto frontier over TopK across various metrics, with RW achieving better trade-offs. As observed also in Section 4.2, MSAE’s performance advantages over TopK increase at higher expansion rates.

### 4.4. Semantic Preservation Analysis

In Section 4.2, we only evaluated SAEs using L⁰ for activation sparsity and EVR for reconstruction fidelity; however, these metrics have limitations. L⁰ only counts active neurons without assessing how well SAE representations align with original embeddings, and EVR focuses solely on distance reconstruction rather than semantic preservation. To address these limitations, we introduce additional metrics. Following (Yu et al., 2025), we adopt the CKNNA metric to assess how well SAE activations preserve the neighborhood structure of CLIP embeddings. We also evaluate semantic preservation through linear probing metrics (Gao et al., 2025; Lieberum et al., 2024); we use LP (KL) to measure prediction distribution alignment and LP (Acc) to compare classification accuracy. All metrics are defined in Section 4.1 and presented in Table 1. Our analysis reveals that while cosine similarity and FVU correlate well with linear probing metrics, the alignment metric demonstrates Matryoshka’s strength in preserving semantic structures.

### 4.5. Orthogonality of SAE Features

SAEs can disentangle polysemantic representations into monosemantic features, as shown and explained by (Bricken et al., 2023). To evaluate feature monosemanticity, we measure decoder orthogonality using the DO metric, with results reported in Table 1. While all methods achieve high orthogonality as indicated by low DO values, none reach perfect orthogonality. This might stem from multiple factors, including feature absorption as noted in (Chanin et al., 2025), or just learning similar concepts (such as different numbers). We argue that understanding these sources of non-orthogonality is crucial for advancing the development of more effective monosemantic feature learning in SAEs.

### 4.6. Activations Magnitudes Analysis

To analyze the impact of sparsity proxies on SAE, we examine non-zero activation distributions across ViT-L with expansion rate 8 in Figure 4. Matryoshka models display a distinctive double-curvature distribution similar to ReLU-based models, with values between 5 to 10 appearing almost linear in log₁₀ space. Following (Templeton et al., 2024), we attribute low activations to reconstruction purposes rather than semantic meaning. The second curvature reflects natural images’ complexity, which requires multiple concept reconstructions rather than single dominant features, as evidenced by the small number of very high values corresponding to rare, nearly singular concept images (Figure 10). As the sparsity parameter *k* in TopK methods increases (Fig. 14), the transition from one to double-curvature behavior suggests that stronger sparsity constraints create composite features, supported by Appendix C showing that high-activation features (> 15) in TopK methods have a lower ratio of valid named features compared to Matryoshka.

### 4.7. Progressive Recovery

To verify that our method learns hierarchical structure, we perform a progressive reconstruction task by using an increasing number of SAE activations, ordered by magnitude, to recover the original vector. Figure 5 shows that reconstruction quality improves with decreasing sparsity thresholds (increasing *k*) during inference. TopK variants exhibit performance plateaus shortly after their training thresholds (*k* = {32, 64}), while ReLU-based models show continued improvement but with inferior performance at higher sparsity. MSAE demonstrates a better hierarchical structure that combines TopK’s efficient high-sparsity performance with ReLU’s scaling capabilities. While our method performs slightly below TopK (*k* = 32) at the highest sparsity, it quickly surpasses TopK’s plateau at lower sparsity, achieving performance levels above ReLU models. We observe similar patterns in the CKNNA alignment metric, with MSAE outperforming both TopK (*k* = 32) and ReLU models beyond *k* = 10 while performing only slightly below TopK (*k* = 256) at the lowest sparsity. Evidence of improved hierarchical feature learning across metrics and modalities is presented in Appendix F.4.

### 4.8. Training Modality: Language and Vision

We evaluate how training modality affects MSAE performance by comparing models trained on text with the original image-trained models, validating both modality models across text and image domains in Table 2. While both variants perform best in their training domains, text-trained models achieve superior cross-modal performance, demonstrating stronger generalization capabilities. Moreover, text-trained models achieve higher sparsity on both modalities with no dead neurons, showing better utilization of learned features. These findings position text training as a preferred approach for multi-modal applications where balanced performance is desired. Future research could explore training SAEs on varying ratios of text and image data to optimize cross-modal performance or try to train crosscoders (Lindsey et al., 2024) on both modalities simultaneously. We defer extended MSAE evaluations to Appendix F.

## 5. Interpreting CLIP with MSAE

In this section, we demonstrate how MSAE can enhance interpretability and control interpretable features in CLIP-based applications. We first establish neuron-concept mappings in the activation layer through an automated technique described in Section 5.1. Then, we show its effectiveness in concept-based similarity search across the ImageNet validation set, enabling retrieval of images with varying degrees of explicit concept presence. Moreover, we leverage MSAE to study potential conceptual biases in a gender classification model trained on the CelebA dataset (Liu et al., 2015).

### 5.1. Concept Naming

While self-supervised training of SAE enables learning up to $d$ monosemantic concepts, mapping these concepts to specific neurons remains non-trivial. Previous work used LLMs for identifying neuron-encoded concepts (Bills et al., 2023), but we adopt the more efficient method for CLIP-trained SAE proposed in (Rao et al., 2024), which leverages CLIP’s representation space. Our concept detection and validation methodology is detailed in Appendix A, with comprehensive results on valid concept counts across SAE models presented in Table 3. Figure 8 demonstrates the highest-activating text and image examples for the best-matched feature concept ‘face’ across ReLU, TopK, and MSAE. The consistent concept presence across diverse inputs observed primarily in MSAE variants suggests that only Matryoshka-based methods were capable of learning this monosemantic feature. Supplementary analysis of highly activated concept examples in Appendix G.1 showcases SAE’s ability to learn a wide range of concepts, from simple textures and colors to more complex ones like light (lights in darkness), countable concepts like trio (groups of three), and even nationality-related concepts like ireland or germany. Naming SAE features enables using SAE to conduct diverse interpretability analyses related to CLIP. We present two use cases where we apply the MSAE RW variant with an expansion rate of 8.

### 5.2. Similarity Search

CLIP embeddings are widely used for cross-modal similarity search between images and text through cosine similarity metric, primarily for retrieval engines. We extend this capability using SAE in three ways. First, SAE provides interpretable insights into nearest neighbor (NN) image retrievals. Figure 21 shows the top 8 concepts for the two closest retrieved images, revealing shared semantic concept patterns and explaining why both NNs match the query image of an Irish police vehicle, with the first NN (Irish police vehicle) being closer than the second (British police vehicle). Second, we compare similarity search in CLIP embedding space against SAE activation space using Manhattan distance (detailed and visualized in Appendix G.2). While the first NN remains consistent across both spaces, the second NN in the SAE space shows the same vehicle type from a different angle, demonstrating that similarity searches can be done in both spaces while SAE enables additional concept-based interpretability. Finally, we demonstrate a controlled similarity search by manipulating concept magnitudes. In Figure 7, increasing the germany concept strength preserves the original image as the top match but shifts the second NN from an Irish to a German police vehicle, while preserving the overall input image structure. The increasing distances from the original image embedding show how larger magnitude adjustments affect embedding coherence.

### 5.3. Bias Validation on a Downstream Task

CLIP models are commonly used as feature extractors for downstream tasks, enabling efficient fine-tuning with limited data. With MSAE, we can investigate whether downstream models learn to associate specific concepts with classes. To demonstrate this, we train a single-layer classifier on CLIP embeddings from the CelebA dataset to perform binary gender classification (1 for female, 0 for male), achieving an F1 score of approximately 0.99. Through statistical analysis in Appendix G.3, we uncover several concept-gender associations: bearded biases toward male classification, blonde toward female, and glasses showing modest male bias. To validate these findings, Figure 6 demonstrates an example of how increasing these concepts’ magnitudes affects classification scores for a female example (see Figure 23 for a male example). The results confirm our statistical analysis, and the plateaus in classification probabilities as concept magnitudes increase help quantify the strength of concept-gender associations in the model.

## 6. Conclusion

We propose Matryoshka SAE to advance our understanding of CLIP embeddings through hierarchical sparse autoencoders. MSAE improves upon both TopK and ReLU approaches, achieving superior sparsity–fidelity trade-off while providing flexible sparsity control via the $\alpha$ coefficient. Our experiments demonstrate MSAE’s effectiveness through near-optimal metrics, progressive feature recovery, and extraction of over 120 validated concepts, enabling new applications in concept-based similarity search and bias detection in downstream tasks.

### Limitations and future work

MSAE faces three limitations with clear paths for future improvement. The current implementation’s use of multiple decoder passes with different TopK activations introduces computational overhead, which could be addressed through optimized CUDA kernels enabling parallel processing of multiple granularities. While we demonstrated MSAE’s effectiveness using CLIP embeddings, it has great potential to explain hierarchical representations in other embedding spaces, such as SigLIP (Zhai et al., 2023) or modality-specific representations. Finally, since not all neurons correspond to simple concepts in our vocabulary, investigating complex semantic features through LLM-based interpretability methods could provide deeper insights into the learned hierarchical representations. In concurrent independent work, Bussmann et al. (2025) propose an MSAE approach to interpreting language models.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning Interpretability. There are some potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgement

Work on this project was financially supported from the SONATA BIS grant 2019/34/E/ST6/00052 funded by Polish National Science Centre (NCN). We also thank the anonymous reviewers for their useful comments.

## References

Abdulaal, A., Fry, H., Monta ̃ na-Brown, N., Ijishakin, A., Gao, J., Hyland, S., Alexander, D. C., and Castro, D. C. An x-ray is worth 15 features: Sparse autoencoders for interpretable radiology report generation. *arXiv preprint arXiv:2410.03334*, 2024.

Abnar, S. and Zuidema, W. Quantifying attention flow in transformers. In *ACL*, 2020.

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B. Sanity checks for saliency maps. In *NeurIPS*, 2018.

Balasubramanian, S., Basu, S., and Feizi, S. Decomposing and interpreting image representations via text in ViTs beyond CLIP. In *NeurIPS*, 2024.

Baniecki, H., Casalicchio, G., Bischl, B., and Biecek, P. Efficient and accurate explanation estimation with distribution compression. In *ICLR*, 2025.

Bereska, L. and Gavves, E. Mechanistic interpretability for AI safety–A review. *Transactions on Machine Learning Research*, 2024.

Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F., and Lakkaraju, H. Interpreting CLIP with sparse linear concept embeddings (SpLiCE). In *NeurIPS*, 2024.

Biecek, P. and Samek, W. Position: Explain to question not to justify. In *ICML*, 2024.

Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J., and Saunders, W. Language models can explain neurons in language models. https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html, May 2023.

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N. L., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. https://transformer-circuits.pub/2023/monosemantic-features/index.html, October 2023. Transformer Circuits Thread.

Bussmann, B., Leask, P., and Nanda, N. BatchTopK sparse autoencoders. *arXiv preprint arXiv:2412.06410*, 2024.

Bussmann, B., Nabeshima, N., Karvonen, A., and Nanda, N. Learning multi-level features with matryoshka sparse autoencoders. *arXiv preprint arXiv:2503.17547*, 2025.

Bykov, K., Kopf, L., Nakajima, S., Kloft, M., and H ̈ ohne, M. M. Labeling neural representations with inverse recognition. In *NeurIPS*, 2023.

Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., and Bloom, J. A is for absorption: Studying feature splitting and absorption in sparse autoencoders. In *ICLR*, 2025.

Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In *CVPR*, 2023.

Conerly, T., Templeton, A., Bricken, T., Marcus, J., and Henighan, T. Update on how we train saes. https://transformer-circuits.pub/2024/april-update/index.html, April 2024. Transformer Circuits Thread.

Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. Towards automated circuit discovery for mechanistic interpretability. In *NeurIPS*, 2023.

Crabbé, J., Rodriguez, P., Shankar, V., Zappella, L., and Blaas, A. Interpreting CLIP: Insights on the robustness to imagenet distribution shifts. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856.

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse autoencoders find highly interpretable features in language models. In *ICLR*, 2024.

Espinosa Zarlenga, M., Barbiero, P., Ciravegna, G., Marra, G., Giannini, F., Diligenti, M., Shams, Z., Precioso, F., Melacci, S., Weller, A., et al. Concept embedding models: Beyond the accuracy-explainability trade-off. In *NeurIPS*, 2022.

Gandelsman, Y., Efros, A. A., and Steinhardt, J. Interpreting CLIP’s image representation via text-based decomposition. In *ICLR*, 2024.

Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. In *ICLR*, 2025.

Gemma, T. and DeepMind, G. Gemma 2: Improving open language models at a practical size. *arXiv preprint arXiv:2408.00118*, 2024.

Ghorbani, A., Wexler, J., Zou, J. Y., and Kim, B. Towards automatic concept-based explanations. In *NeurIPS*, 2019.

Goh, G., Cammarata, N., Voss, C., Carter, S., Petrov, M., Schubert, L., Radford, A., and Olah, C. Multimodal neurons in artificial neural networks. *Distill*, 6(3):e30, 2021.

Hernandez, E., Schwettmann, S., Bau, D., Bagashvili, T., Torralba, A., and Andreas, J. Natural language descriptions of deep visual features. In *ICLR*, 2021.

Huh, M., Cheung, B., Wang, T., and Isola, P. The platonic representation hypothesis. In *ICML*, 2024.

Jain, S., Lawrence, H., Moitra, A., and Madry, A. Distilling model failures as directions in latent space. In *ICLR*, 2023.

Joukovsky, B., Sammani, F., and Deligiannis, N. Model-agnostic visual explanations via approximate bilinear models. In *ICIP*, 2023.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In *ICML*, 2018.

Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., and Liang, P. Concept bottleneck models. In *ICML*, 2020.

Kopf, L., Bommer, P. L., Hedstr ̈ om, A., Lapuschkin, S., H ̈ ohne, M. M., and Bykov, K. Cosy: Evaluating textual explanations of neurons. In *NeurIPS*, 2024.

Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. Similarity of neural network representations revisited. In *ICML*, 2019.

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., et al. Matryoshka representation learning. In *NeurIPS*, 2022.

Li, Y., Wang, H., Duan, Y., Xu, H., and Li, X. Exploring visual interpretability for contrastive language-image pre-training. *arXiv preprint arXiv:2209.07046*, 2022.

Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kram ́ ar, J., Dragan, A., Shah, R., and Nanda, N. Gemma Scope: Open sparse autoencoders everywhere all at once on Gemma 2. *arXiv preprint arXiv:2408.05147*, 2024.

Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., and Olah, C. Sparse cross-coders for cross-layer features and model diffing. https://transformer-circuits.pub/2024/crosscoders/index.html, October 2024. Transformer Circuits Thread.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In *NeurIPS*, 2023.

Liu, Y., Zhang, X., Zhang, S., and He, X. Part-aware pro- totype network for few-shot semantic segmentation. In *ECCV*, 2020.

Liu, Z., Luo, P., Wang, X., and Tang, X. Deep learning face attributes in the wild. In *ICCV*, 2015.

Lucieri, A., Bajwa, M. N., Braun, S. A., Malik, M. I., Den- gel, A., and Ahmed, S. On interpretability of deep learn- ing based skin lesion classifiers using concept activation vectors. In *IJCNN*, 2020.

Lundberg, S. and Lee, S.-I. A unified approach to interpret- ing model predictions. In *NeurIPS*, 2017.

Madeira, P., Carreiro, A., Gaudio, A., Rosado, L., Soares, F., and Smailagic, A. ZEBRA: Explaining rare cases through outlying interpretable concepts. In *CVPR*, 2023.

Misino, E., Marra, G., and Sansone, E. VAEL: Bridging variational autoencoders and probabilistic logic program- ming. In *NeurIPS*, 2022.

Oikarinen, T. and Weng, T.-W. CLIP-Dissect: Automatic description of neuron representations in deep vision net- works. In *ICLR*, 2023.

Oikarinen, T., Das, S., Nguyen, L. M., and Weng, T.-W. Label-free concept bottleneck models. In *ICLR*, 2023.

Paulo, G. and Belrose, N. Sparse autoencoders trained on the same data learn different features. *arXiv preprint arXiv:2501.16615*, 2025.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M ̈ uller, J., Penna, J., and Rombach, R. SDXL: Im- proving latent diffusion models for high-resolution image synthesis. In *ICLR*, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In *ICML*, 2021.

Rajamanoharan, S., Conmy, A., Smith, L., Lieberum, T., Varma, V., Kram ́ ar, J., Shah, R., and Nanda, N. Improving dictionary learning with gated sparse autoencoders. *arXiv preprint arXiv:2404.16014*, 2024a.

Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kram ́ ar, J., and Nanda, N. Jumping ahead: Improving reconstruction fidelity with JumpReLU sparse autoencoders. *arXiv preprint arXiv:2407.14435*, 2024b.

Ramaswamy, V. V., Kim, S. S. Y., Fong, R., and Rus- sakovsky, O. Overlooked factors in concept-based expla- nations: Dataset choice, concept learnability, and human capability. In *CVPR*, 2023.

Rao, S., Mahajan, S., B ̈ ohle, M., and Schiele, B. Discover- then-name: Task-agnostic concept bottlenecks via auto- mated concept discovery. In *ECCV*, 2024.

Ribeiro, M. T., Singh, S., and Guestrin, C. “Why should I trust you?”: Explaining the predictions of any classifier. In *KDD*, 2016.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. ImageNet large scale visual recognition chal- lenge. *International Journal of Computer Vision*, 115: 211–252, 2015.

Sammani, F., Joukovsky, B., and Deligiannis, N. Visualizing and understanding contrastive learning. *IEEE Transac- tions on Image Processing*, 33:541–555, 2024.

Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., and Komatsuzaki, A. LAION-400M: Open dataset of CLIP- filtered 400 million image-text pairs. *arXiv preprint arXiv:2111.02114*, 2021.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *ICCV*, 2017.

Sharma, P., Ding, N., Goodman, S., and Soricut, R. Con- ceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In *ACL*, 2018.

Shen, S., Li, L. H., Tan, H., Bansal, M., Rohrbach, A., Chang, K.-W., Yao, Z., and Keutzer, K. How much can CLIP benefit vision-and-language tasks? In *ICLR*, 2022.

Shrikumar, A., Greenside, P., Shcherbina, A., and Kundaje, A. Not just a black box: Learning important features through propagating activation differences. In *ICML*, 2017.

Simonyan, K. Deep inside convolutional networks: Visu- alising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*, 2013.

Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribu- tion for deep networks. In *ICML*, 2017.

Surkov, V., Wendler, C., Terekhov, M., Deschenaux, J., West, R., and Gulcehre, C. Unpacking sdxl turbo: Inter- preting text-to-image models with sparse autoencoders. *arXiv preprint arXiv:2410.22366*, 2024.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Tamkin, A., Durmus, E., Hume, T., Mosconi, F., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T. Scaling monosemanticity: Extracting inter- pretable features from Claude 3 Sonnet. https://transformer-circuits.pub/2024/ scaling-monosemanticity/index.html, May 2024. Transformer Circuits Thread.

Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., Xu, J., Xu, B., Li, J., Dong, Y., Ding, M., and Tang, J. CogVLM: Visual expert for pretrained language models. In *NeurIPS*, 2024.

Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., and Xie, S. Representation alignment for generation: Training diffusion transformers is easier than you think. In *ICLR*, 2025.

Zeiler, M. D. and Fergus, R. Visualizing and understanding convolutional networks. In *ECCV*, 2014.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sig- moid loss for language image pre-training. In *ICCV*, 2023.

Zhao, C., Wang, K., Zeng, X., Zhao, R., and Chan, A. B. Gradient-based visual explanation for transformer-based CLIP. In *ICML*, 2024.

Zhou, B., Sun, Y., Bau, D., and Torralba, A. Interpretable basis decomposition for visual explanation. In *ECCV*, 2018.

## Appendix

### A. Concept Discovery and Validation

Here, we describe our approach for detection, which concepts SAE learned, and how we validated the mappings of these concepts to specific neurons. While LLMs are commonly used to identify neuron-encoded concepts (Bereska & Gavves, 2024; Conmy et al., 2023), we follow Rao et al. (2024) in implementing a more computationally efficient approach, which is more tailored to the CLIP-based SAE.

#### CLIP-Based concept matching

The method uses a predefined vocabulary of concepts (e.g., ‘hair’, ‘pink’) to compute cosine similarity between CLIP embeddings and SAE decoder columns. After mapping concepts to CLIP’s embedding space and applying the same preprocessing as during SAE training, we remove $b_{\text{pre}}$ from the preprocessed CLIP embeddings for comparison with the decoder. For the feature columns in the SAE decoder, which are unit-magnitude by definition, the best matching concept to the neuron is determined by maximizing cosine similarity, where a value of 1 indicates perfect alignment. Thus, the optimal concept $s_c$ for neuron $p_c$ is defined as:

$$
s_c = \arg \max_{v \in V} [\cos(p_c, \text{CLIP}(v))] = \arg \max_{v \in V} \left\| \frac{p_c \cdot \text{CLIP}(v)}{\|p_c\| \|\text{CLIP}(v)\|} \right\|. \quad (3)
$$

#### Pre-activation bias in similarity calculations

While the method above suggests removing $b_{\text{pre}}$ from CLIP embeddings, our empirical analysis revealed this significantly masks neuron-concept relationships. Without $b_{\text{pre}}$, in Figure 9, we show that similarities cluster around 0.1 (mean) with maxima around 0.15–0.2, whereas retaining $b_{\text{pre}}$ yields higher similarity scores (> 0.42) that correspond to correct concepts. Importantly, both approaches preserve neuron rankings, with over 95% of concepts sharing identical highest matching neurons, so not removing bias doesn’t destroy ranking. Manual evaluation confirmed that neurons with bias-removed similarities (~0.2) are underestimated compared to their bias-inclusive counterparts (~0.5). Based on these findings, we retain $b_{\text{pre}}$ in our calculations.

#### Limitations of the current approach

We identify several limitations in the current concept mapping approach. First, the method assigns concepts to neurons based on the highest similarity regardless of the absolute matching quality, potentially leading to poor concept assignments when no good matches exist. Second, hierarchical concepts pose a challenge when matching with more specific neuronal features. For example, a high-level concept like ‘mammal’ may show strong similarity to both ‘cat’ and ‘dog’ neurons, resulting in imprecise assignments. This issue stems from either semantic feature vectors that aren’t perfectly orthogonal or incomplete vocabulary coverage.

#### Threshold-based validation

To address the challenges identified above, we propose three validations to remove weak assignments. Before applying the validations, we switch the mapping from concepts to neurons to a mapping of neurons to concepts to reduce spurious assignments. Based on this, we threshold results by either:
1.  Cosine similarity > 0.42, which ensures that the neurons exhibit strong alignment with their assigned concepts, preventing weak or ambiguous concept mappings.
2.  Concept similarity ratio: $\frac{\text{Top similarity}}{\text{Second-highest similarity}} > 2.0$, confirms concept uniqueness by requiring the best match to be at least twice as strong as the second-best concept, avoiding distributed representations.
3.  One concept per neuron (with the highest similarity) enforces monosemanticity by assigning only the most strongly aligned concept to each neuron, which is needed due to the vocabulary structure containing multiple variations of the same concept (e.g., ‘bird’ and ‘birdie’).

#### Vocab data

Following (Ramaswamy et al., 2023) principle that vocabulary concepts should be simple, we adopted the vocabulary from (Bhalla et al., 2024). This vocabulary comprises the most frequent unigrams from LAION-400m captions dataset (Schuhmann et al., 2021). To account for semantic relationships between concepts, we perform manual validation of top concepts for each discovered neuron.

#### Semantic consistency

Manual evaluation of top concepts per neuron verifies concept consistency and identifies hierarchical relationships, where top vocab similarities (such as dog breeds) can indicate broader categorical concepts (such as ‘dog’).

#### Results across SAE architectures

We evaluate concept neurons across architectures in Table 3. From 37,445 neurons at expansion rate 8, only a small fraction passed similarity validation: ~10% for TopK and 1–3% for ReLU and Matryoshka architectures. While higher expansion rates typically reduce valid neurons, both TopK variants and ReLU ($\lambda$ = 0.001) exhibit increased valid mappings under best vector validation. Although these results suggest limited concept learning or concept distribution across neurons, the vocabulary structure prevents definitive conclusions, due to the dominance of non-semantic unigrams, and many semantically similar concepts appear across vocabulary (e.g., ‘blue’, ‘blau’ and ‘bleu’). The validation results from the table demonstrate that sparser architectures (TopK) yield 3–8 times more interpretable concept neurons compared to more dense ones (ReLU), with Matryoshka being between the two, supporting the hypothesis that sparsity promotes concept specialization.

### B. Implementation Details

We conducted experiments using CLIP ViT-L/14 (and ViT-B/16, reported later in the appendix) pre-trained on the CC3M dataset image training subset. Following (Bricken et al., 2023; Gao et al., 2025) and this blog (Conerly et al., 2024), our SAE implementation uses unit-norm constraint on the decoder columns with untied encoder and decoder. We initialized $b_{\text{pre}}$ and $b_{\text{enc}}$ with zeros, decoder with uniform Kaiming initialization (scaled L² norm to 0.1), and encoder as the decoder’s transpose. Gradient clipping was set to 1. For data preprocessing, we centralized embeddings per modality (Bhalla et al., 2024) and scaled by a constant to achieve $E_{x \in X}[ \|x\|_2 ] = \sqrt{n}$. All models were trained for 30 epochs on a single NVIDIA A100 GPU with batch size 4096, except for the model with an expansion rate of 32, which was trained for 20 epochs. While MSAE and TopK showed dead neurons, we omitted revival strategies as only in TopK the number of dead neurons exceeded 1%.

#### B.1. Hyperparameters

We first conducted experiments on CLIP RN50 using hyperparameters from (Rao et al., 2024), later validating them on ViT-L/14. For ViT-L/14, we explored parameters near RN50-optimal values to ensure cross-architecture consistency. With expansion factor 8 (768 → 6144), we explore:

*   Learning rates per method: $1 \cdot 10^{-5}, 5 \cdot 10^{-5}, 1 \cdot 10^{-4}, 5 \cdot 10^{-4}, 1 \cdot 10^{-3}$
*   ReLU L¹ coefficients ($\lambda$): $1 \cdot 10^{-4}, 3 \cdot 10^{-3}, 1 \cdot 10^{-3}, 3 \cdot 10^{-2}$
*   TopK values: $k \in \{32, 64, 128, 256\}$, up to 256 as (Gao et al., 2025) suggests higher values do not learn interpretable features
*   Matryoshka K-lists: $\{32, \dots, 6144\}$ and $\{64, \dots, 6144\}$, for higher expansion rates we adjust the upper limit
*   $\alpha$ coefficients: uniform weighting (UW) $\{1,1,1,1,1,1,1\}$ and reverse weighting (RW) $\{7,6,5,4,3,2,1\}$

The optimal parameters from these experiments were applied to larger expansion factors of 16 (768 → 12288), 32 (768 → 24576), and all expansion rates of VIT-B/16. Following reviewer suggestions, we extended our evaluations to include TopK ($k$ = 512) and BatchTopK for $k \in \{16, 32, 64, 128, 256\}$. The BatchTopK models were trained using the same hyperparameters as TopK models. These additional results are presented in the Appendix. We also attempted to integrate JumpReLU (Rajamanoharan et al., 2024b) into our evaluations, but did not achieve meaningful results, which may be attributed to an implementation error in our code.

#### B.2. Optimal Parameters

Based on RN50 experiments and subsequent adjustment to VIT-L/14 with expansion factor 8, we selected the following optimal configurations: ReLU with learning rate $5 \cdot 10^{-5}$ and $\lambda$ values of $1 \cdot 10^{-3}, 3 \cdot 10^{-3}, 3 \cdot 10^{-2}$; TopK with learning rate $5 \cdot 10^{-4}$ and $k$ values of 32, 64, 128, and 256; MSAE with learning rate $1 \cdot 10^{-4}$, K-list $\{64, \dots, 6144\}$, for both uniform (UW) and reverse weighting (RW) $\alpha$ strategies.

### C. Highest Neuron Magnitudes

Based on results from Figure 4, we analyze images from ImageNet-1k validation set that produced the highest neuron magnitudes for TopK and MSAE architectures. In Table 4, we show that more constrained SAEs (TopK ($k \le 128$)) produce a higher number of samples with neurons above 15; however, the percentage of valid neurons is lower than in MSAE and TopK ($k$ = 256) which have significantly less high magnitude samples. This indicates that high-magnitude neurons in highly constrained TopK may presumably learn complex features. Figure 10 presents the top 6 valid highest neuron magnitude images per model, demonstrating that very high magnitudes often correspond to images with almost singular concepts.

### D. Activation Soft-capping

Analysis of MSAE in Figure 4 reveals that despite effective handling of multi-granular sparsity, the model learns to encode concepts using extremely large activation values (> 15). This can lead to more composite rather than atomic features, as it was in the case of TopK ($k \le 128$) revealed in Appendix C.

#### D.1. Definition of Soft-capping

To address this, we introduce activation soft-capping (SC), adapting the logit soft-capping concept from language models (Gemma & DeepMind, 2024). This technique prevents too high activation magnitude and circumvention of sparsity constraints via activation magnitude manipulation:

$$
\hat{z} = \text{softcap} \cdot \tanh(z / \text{softcap}), \quad \hat{x} = W_{\text{dec}}\hat{z} + b_{\text{pre}}, \quad (4)
$$

where softcap hyperparameter controls maximum activation magnitude. Combined with ReLU, this bounds SAE activations to $(0, \text{softcap})$.

#### D.2. Results

In Table 5, we show soft-capping’s impact on MSAE performance across key metrics using the ImageNet-1k training set. Our analysis reveals two key benefits of applying soft-capping on MSAE. First, it consistently improves L⁰ sparsity, with MSAE RW (SC) achieving values of 0.830 and 0.889 for 6144 and 12288 sizes, respectively. Second, while base MSAE UW maintains better FVU and CS scores, the soft-capped MSAE RW significantly reduces the number of dead neurons. With a latent size of 12288, MSAE RW exhibits only 66 dead neurons, compared to 491 in the model with a latent size of 6144. These findings show that soft-capping is particularly beneficial for large-scale SAEs with wider sparse layers, where neuron utilization becomes more challenging. The technique provides a practical approach to reducing dead neurons while maintaining high L⁰ sparsity, with only minimal impact on reconstruction fidelity.

### E. CKNNA Alignment Metric

Introduced in Section 4.4, CKNNA (Centered Kernel Nearest-Neighbor Alignment) measures representation similarity between networks while focusing on local neighborhood structures. Unlike its predecessor CKA (Kornblith et al., 2019), CKNNA refines the alignment computation by considering only *k*-nearest neighbors, making it more sensitive to local geometric relationships. The alignment score between two networks’ representations in our case CLIP embeddings and SAE activation is computed as:

$$
\text{CKNNA}(K, L) = \frac{\text{Align}(K, L)}{\sqrt{\text{HSIC}(K, K) \text{HSIC}(L, L)}},
$$

$$
\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \sum_{i} \sum_{j} (\langle \phi_i, \phi_j \rangle - E_l[\langle \phi_i, \phi_l \rangle])(\langle \psi_i, \psi_j \rangle - E_l[\langle \psi_i, \psi_l \rangle]),
$$

$$
\text{Align}(K, L) = \frac{1}{(n-1)^2} \sum_{i} \sum_{j} \alpha(i, j)(\langle \phi_i, \phi_j \rangle - E_l[\langle \phi_i, \phi_l \rangle])(\langle \psi_i, \psi_j \rangle - E_l[\langle \psi_i, \psi_l \rangle]),
$$

$$
\alpha(i, j; k) = \mathbb{1}[i \neq j \text{ and } \phi_j \in \text{knn}(\phi_i; k) \text{ and } \psi_j \in \text{knn}(\psi_i; k)],
$$

where HSIC measures the global similarity between kernel matrices, and Align introduces the neighborhood constraint through $\alpha(i, j; k)$. The indicator function $\alpha(i, j; k)$ ensures that only pairs of points that are *k*-nearest neighbors in both representation spaces contribute to the alignment score. Here, $\phi_i, \phi_j$ represent CLIP embeddings and $\psi_i, \psi_j$ represent SAE activations for corresponding input data points $i$ and $j$. Following (Yu et al., 2025), we set $k = 10$ as it provides better alignment sensitivity and calculate CKNNA over randomly sampled (batch size) 10,000 representations when evaluating. Higher CKNNA scores indicate stronger similarity between the CLIP and SAE learned representations.

### F. Evaluating MSAE: Additional Results

We extend the results from Section 4 by analyzing multiple expansion rates, SAE variants, input modalities, and CLIP architectures. Unless otherwise specified, experiments use CLIP ViT-L/14 with an expansion rate of 8 on image modality. For text modality evaluations, we use the CC3M validation subset, while image modality evaluations are performed on the ImageNet-1k training subset.

#### F.1. Sparsity–Fidelity Trade-off

Figure 11 presents an extended analysis of sparsity-fidelity trade-offs, including standard deviations and an alternative reconstruction metric tailored for CLIP embeddings (cosine similarity). The results demonstrate MSAE’s superior stability across both modalities, particularly in text representations where only MSAE models show stable and elevated results. Furthermore, we observe that models learned with lower *k* or higher sparsity regularization show high variance on the trained (image) modality, and this instability becomes even more pronounced on the text modality. Due to MSAE’s inherently low variance in the image modality, this similar low variance is preserved in the other modality, leading to consistently high and stable performance across both modalities. Figure 12 further strengthens our findings by showing MSAE’s superiority on a different CLIP architecture (ViT-B/16).

#### F.2. Ablation: Matryoshka at Lower Granularity Levels

Figure 13 extends the analysis from Figure 3 by evaluating four key metrics: reconstruction fidelity (EVR), reconstruction error (CS), alignment (CKNNA), and neuron utilization (NDN). Our expanded comparison reinforces MSAE’s competitive performance against TopK SAE, demonstrating that MSAE (RW) achieves similar or better results across most metrics except for NDN, where the (UW) version of MSAE performs better.

#### F.3. Activation Magnitudes Analysis

We extend the activation magnitude analysis from Figure 4 by including varying versions of model sparsity from each evaluated SAE architecture. Figure 14 shows non-zero and maximum SAE activations for expansion rate 8, revealing that less constrained TopK models exhibit double-curvature distributions similar to MSAE and ReLU. Maximum activation analysis highlights ReLU’s shrinkage effect, while TopK and MSAE maintain distributions closer to normal. These patterns persist at higher expansion rates (16 and 32) as shown in Figure 15.

#### F.4. Progressive Recovery

We extend the analysis from Figure 2 by examining progressive reconstruction performance across additional metrics and modalities. Figure 16 demonstrates performance at expansion rate 8 for reconstruction quality (EVR, CS) and neuron utilization (NDN) across both modalities, while Figure 17 extends this analysis to expansion rates 16 and 32.

#### F.5. Comprehensive Evaluation with CLIP ViT-L/14 and ViT-B/16 Architectures

We present an extensive quantitative comparison of SAE variants across CLIP ViT-L/14 and ViT-B/16 architectures. Our evaluation encompasses ReLU ($\lambda$ = 0.03, 0.01, 0.003, 0.001), TopK ($k$ = 32, 64, 128, 256, 512), BatchTopK ($k$ = 16, 32, 64, 128, 256), and MSAE (RW, UW) models, tested across three expansion rates (8, 16, 32) for both image and text modalities. For text modality and ViT-B/16 architecture, we omit LP (Acc) and LP (KL) metrics based on our findings in Section 4.4 that CS and FVU correlate strongly with linear probing metrics.

##### F.5.1. Results for Image Modality

For image modality, Tables 6–8 present detailed results for ViT-L/14 across expansion rates 8, 16, and 32, while Tables 9–11 show parallel performance metrics for ViT-B/16. These tables extend the analysis from Table 1, providing comprehensive measurements across different metrics and model configurations.

##### F.5.2. Results for Text Modality

For text modality, Tables 12–14 present results for ViT-L/14, while Tables 15–17 show ViT-B/16 performance on CC3M validation text data, enabling cross-modal and cross-architecture comparisons.

### G. Interpreting CLIP with MSAE: Additional Results

In this appendix section, we provide additional analysis supporting Section 5. Section G.1 presents high-magnitude activation samples across modalities from MSAE (RW) with an expansion rate of 8. Section G.2 demonstrates how SAE enhances similarity search with interpretable results. Section G.3 presents statistical gender bias analysis on CelebA dataset, supported by concept manipulation visualizations that reinforce the statistical findings. These analyses strengthen our findings from Section 5 while providing deeper insights into MSAE’s interpretability capabilities.

#### G.1. Concept Visualization Analysis

Figures 18 and 19 showcase six valid concepts through their highest-activating images and texts, confirming concept validity. Conversely, Figure 20 demonstrates two invalid concepts, highlighting the importance of validation methods from Section A.

#### G.2. SAE-Enhanced Similarity Search

Building upon Section 5, we demonstrate how SAE enhances nearest neighbor (NN) search by revealing shared semantic concepts between query and retrieved images. Figure 21 illustrates how SAE uncovers interpretable features that drive CLIP’s similarity assessments. Furthermore, we show that conducting similarity search directly in the SAE activation space produces comparable results to CLIP-based search while providing more semantically meaningful matches.

#### G.3. Gender Bias Analysis in CelebA

We analyze gender biases in a CLIP-based classification model using the CelebA dataset, which forms the foundation for our analysis in Section 5.3. Through statistical analysis of concept magnitude distribution against the model gender predictions in Figure 24, we identify significant gender associations for concepts bearded, blondes, and glasses in the classification model. To verify that these concepts align with the true features in the CelebA dataset, we visualize highest-activation images for each concept in Figure 22. Further concept manipulation experiments on both female (Figure 7) and male (Figure 23) examples confirm and strengthen these statistical findings, providing even greater insight into the relationship between gender classification and the chosen concepts.

### H. Stability Evaluations

To gain a deeper understanding of the stability of learned feature directions for the decoder and encoder across various training seeds, we calculated the stability metric proposed by Paulo & Belrose (2025). Table 18 shows the results for all tested SAEs at an expansion rate of 8. We observe that the stability metric is highly correlated with sparsity. Furthermore, Matryoshka SAE demonstrates a comparable stability trade-off to alternative architectures.