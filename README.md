# DL-Project: Twitter Emotion Analysis System

## Affective-DeBERTa: A Fine-Tuned Transformer Pipeline for Multi-Class Emotion Classification

**Live Demo:** [ Twitter Emotion Analysis App](https://twitter-emotion-analysis-system-1441.streamlit.app/)  
**Model Repository:** [ Hugging Face Weights](https://huggingface.co/Nish40/emotion-deberta-v3)

---

### Abstract
This work presents a state-of-the-art Natural Language Processing (NLP) pipeline for the classification of emotional states in short-form social media text. Our methodology, **Affective-DeBERTa**, utilizes a Microsoft DeBERTa-v3 backbone for disentangled attention-based feature extraction. Unlike standard models, DeBERTa-v3 improves training efficiency by sharing embeddings across the encoder and decoder through an enhanced mask decoder. The system is fine-tuned to detect six primary emotional states: **Joy, Sadness, Anger, Fear, Love, and Surprise**. By integrating a **SentencePiece tokenizer** and a cloud-native deployment strategy involving **GitHub, Hugging Face, and Streamlit**, we achieve a high-performance interface capable of real-time affective computing.

**Keywords:** Twitter Sentiment, Emotion Classification, DeBERTa-v3, Natural Language Processing, Transformer Models, Affective Computing.

---

### 1 Introduction
Social media platforms like Twitter/X have become primary sources for understanding public sentiment. However, emotion classification remains challenging due to short text lengths, irregular grammar, and the high prevalence of sarcasm and slang. Traditional models often fail to capture the long-range contextual dependencies required to distinguish between nuanced emotions like "Surprise" and "Fear."

This project implements a state-of-the-art Transformer-based pipeline that moves beyond traditional CNN/RNN approaches. The system is designed to be:
* **Context-Aware:** Using Disentangled Attention to understand word relationships.
* **Hybrid-Cloud Native:** Decoupling code (GitHub) from model weights (Hugging Face).
* **Real-Time:** Providing sub-second inference via a Streamlit web interface.

---

### 2 Literature Survey
This section summarizes recent research that informs our **Affective-DeBERTa** pipeline, focusing on the evolution from static embeddings to contextual Transformers in emotion detection.

1.  **He et al. (2021):** Introduced DeBERTa, proving that **Disentangled Attention** significantly improves NLU tasks by representing words using separate vectors for content and relative position.
2.  **Rezapour (2024):** Compared BERT and XLNet for emotion detection, finding that Transformers with self-attention mechanisms significantly alleviate the vanishing gradient issues found in older RNN/LSTM models.
3.  **Wan et al. (2024):** Proposed that pre-trained models require specific fine-tuning on informal datasets to capture the unique "noise" of Twitter (hashtags, abbreviations, and informal syntax).
4.  **Jambulkar et al. (2025):** Evaluated **DeBERTa-v3** for subjectivity detection, concluding that its architectural innovations allow it to capture syntactic nuances more effectively than RoBERTa or BERT-Base.
5.  **Imran (2024):** Performed a comparative analysis of six transformers, showing that DeBERTa-v3 provides a superior accuracy-complexity trade-off for real-time web-based inference.
6.  **Zhu (2025):** Applied DeBERTa-v3 to sentiment tasks, achieving F1-scores above 0.92, validating the model’s robustness in high-variance text environments.
7.  **Wolf et al. (2020):** Established the Hugging Face `transformers` ecosystem, which our project utilizes to serve large model weights (Safetensors) without bloating the GitHub repository.
8.  **Saravia et al. (2018):** Introduced the "Emotion" dataset used for many modern benchmarks, highlighting the difficulty of multi-class classification vs. binary sentiment analysis.
9.  **Vaswani et al. (2017):** The seminal "Attention is All You Need" paper, which provided the foundational **Self-Attention** mechanism utilized in our DeBERTa backbone.
10. **Khatavkar et al. (2025):** Demonstrated that Transformer derivatives are superior at handling multilingual and context-specific nuances in unstructured social media data.

---

### 3 Proposed Methodology: Affective-DeBERTa Pipeline
The methodology involves a four-stage pipeline: Data Tokenization, Feature Extraction, Classification, and Cloud Deployment.

#### 3.1 Overall Architecture
The Affective-DeBERTa model architecture is described below:
* **SentencePiece Tokenization:** Raw tweets are processed into sub-word tokens. This allows the model to understand "unknown" words or slang by breaking them into meaningful sub-units.
* **DeBERTa-v3 Backbone:** For each token, the model calculates **Disentangled Attention**. It treats the content ($c_i$) and the relative position ($p_{i|j}$) as separate components:
  $$A_{i,j} = \{c_i, p_{i|j}\} \times \{c_j, p_{j|i}\}^T$$
  This allows the model to understand that the same word can have different emotional weights depending on its position in the sentence.
* **Feature Fusion & Classification:** The output of the Transformer layers is pooled and fed into a Softmax classification head to produce probabilities for the 6 emotion classes.
* **Hybrid Deployment:**
    * **GitHub:** Hosts the Python application logic and UI code.
    * **Hugging Face:** Hosts the 250MB `model.safetensors` weight file.
    * **Streamlit Cloud:** Acts as the execution environment, pulling from both sources to provide a live URL.

---

## 4 Experimental Results and Analysis

### 4.1 Experimental Setup
All experiments were conducted on a labeled corpus of 20,000 tweets. The dataset was split into training (16,000), validation (2,000), and testing (2,000) sets. The **Affective-DeBERTa** model was fine-tuned for **6 epochs** using the AdamW optimizer with a learning rate of $3 \times 10^{-5}$ and a weight decay of $0.01$ to prevent overfitting. We utilized a batch size of 32 and implemented a linear learning rate scheduler with a warmup phase.

### 4.2 Quantitative Results
Table 1 summarizes the performance comparison between our proposed pipeline and standard baselines.

| Model | Test Accuracy | Macro F1-score | Training Complexity |
| :--- | :--- | :--- | :--- |
| Simple CNN Baseline | ≈ 0.6210 | ≈ 0.6055 | Low (Fast) |
| BERT-Base (Standard) | 0.8945 | 0.8812 | Moderate |
| **Affective-DeBERTa (Proposed)** | **0.9538** | **0.9537** | **High (Optimal)** |

The **SimpleCNN** baseline confirms that naive convolutional architectures are insufficient for capturing the linguistic nuances of social media. While **BERT-Base** shows strong performance, our **Affective-DeBERTa** pipeline yields a substantial gain of ~6%, proving the superiority of disentangled attention in understanding emotional context.

### 4.3 Classification Analysis (Per-Class)
To evaluate the model's robustness across different emotional states, we analyzed the Precision, Recall, and F1-score for each category:

| Emotion | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Joy** | 0.97 | 0.96 | 0.96 |
| **Sadness** | 0.96 | 0.97 | 0.97 |
| **Anger** | 0.94 | 0.94 | 0.94 |
| **Fear** | 0.91 | 0.92 | 0.91 |
| **Love** | 0.86 | 0.85 | 0.85 |
| **Surprise** | 0.88 | 0.83 | 0.85 |

### 4.4 Qualitative Discussion
* **Semantic Overlap:** The model performs exceptionally well on "Joy" and "Sadness." However, a minor performance dip is noted in "Love" and "Surprise." Error analysis suggests this is due to inter-class similarity, where tweets containing "Love" often share high lexical similarity with "Joy."
* **Disentangled Attention Success:** Unlike the CNN baseline, Affective-DeBERTa correctly identified emotions in tweets where the emotional keyword appeared late in the sentence, demonstrating the effectiveness of its relative position encoding.
* **Conclusion of Results:** With a macro F1-score of **0.9537**, the proposed pipeline offers a highly competitive accuracy-complexity trade-off, making it suitable for real-time monitoring of social media sentiment.
---

## 5 Conclusion and Future Work

### 5.1 Conclusion
This project successfully designed and deployed the **Affective-DeBERTa** pipeline for multi-class emotion classification on Twitter data. By leveraging the **DeBERTa-v3** architecture's disentangled attention mechanism, we achieved a significant performance boost over traditional CNN and BERT-base models, reaching a test accuracy of **95.38%**. 

Our work demonstrates that:
* **Transformers exceed local feature extractors:** The global attention mechanism is far more effective than local convolutions (CNNs) for capturing sentiment in short, informal text.
* **Hybrid Deployment is Scalable:** Decoupling the model weights (Hugging Face) from the UI (Streamlit) allows for a professional, low-latency user experience.
* **Accuracy-Complexity Trade-off:** We reached state-of-the-art results in only 6 training epochs, proving that DeBERTa-v3 is highly efficient for targeted fine-tuning.

### 5.2 Future Work
To further enhance the system, the following research directions are proposed:
1. **Multimodal Fusion:** Integrating emoji embeddings directly into the tokenization process to capture visual emotional cues.
2. **Sarcasm Detection Module:** Adding an auxiliary classification head specifically trained to detect sarcastic irony, which remains a challenge for current transformers.
3. **Temporal Analysis:** Expanding the pipeline to track emotional shifts over time (e.g., monitoring public mood during a major event).
4. **Knowledge Distillation:** Compressing the DeBERTa-v3-base model into a "Tiny" version for faster mobile-edge deployment without losing significant accuracy.

---
### 6 References
[1] **He, P., Liu, X., Gao, J., & Chen, W. (2021).** "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." *Proceedings of the International Conference on Learning Representations (ICLR 2021).* [Link](https://arxiv.org/abs/2006.03654)

[2] **Rezapour, M. (2024).** "Emotion Detection with Transformers: A Comparative Study." *arXiv preprint arXiv:2403.15454.* (A study on BERT, XLNet, and RoBERTa for the Emotion dataset).

[3] **Wan, B., Wu, P., Yeo, C. K., & Li, G. (2024).** "Emotion-cognitive reasoning integrated BERT for sentiment analysis of online public opinions on emergencies." *Information Processing & Management*, 61(2), 103609.

[4] **Jambulkar, A. et al. (2025).** "QU-NLP at CheckThat! 2025: Multilingual Subjectivity in News Articles Detection using DeBERTa-V3." *CLEF 2025 Working Notes.* [Link](https://arxiv.org/abs/2507.21095)

[5] **Khatavkar, V., Velankar, M., & Petkar, S. (2025).** "Multilingual Transformer Contextual Embedding Model for Political Tweets Analysis." *Cureus Journal of Computer Science.*

[6] **Zhu, V. & Silva, N. (2025).** "DESS: DeBERTa Enhanced Syntactic-Semantic Aspect Sentiment Triplet Extraction." *Proceedings of the 11th Italian Conference on Computational Linguistics (CLiC-it 2025).*

[7] **Wolf, T., et al. (2020).** "Transformers: State-of-the-Art Natural Language Processing." *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).*

[8] **Saravia, S., et al. (2018).** "CARER: Contextualized Affect Representations for Emotion Recognition." *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.* (The original 'Emotion' dataset source).

[9] **Vaswani, A., et al. (2017).** "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS).* (Foundational paper for the Transformer architecture).

[10] **Imran, M. (2024).** "Exploring Transformers in Emotion Recognition: A comparison of BERT, DistilBERT, RoBERTa, XLNet and ELECTRA." *ResearchGate Technical Report.*

---

**Developed by:** Nisha  
**Project:** DL-Project 
