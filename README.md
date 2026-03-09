# ULMFiT vs Naive Bayes — Yelp Sentiment Classification

Binary sentiment classification on Yelp customer reviews using two approaches: fine-tuned AWD_LSTM via the ULMFiT method (fastai) and a traditional TF-IDF + Multinomial Naive Bayes baseline. This project compares deep learning and classical NLP in terms of accuracy, development effort, and practical deployability.

---

## Overview

Sentiment analysis on customer reviews is a core NLP task in business intelligence and customer experience management. This project fine-tunes a pretrained language model on Yelp reviews and benchmarks it against a lightweight classical baseline, demonstrating the real-world trade-offs between the two approaches.

---

## Dataset

**Yelp Polarity** — binary sentiment classification (positive vs negative reviews)

| Property | Detail |
|---|---|
| Source | Hugging Face Datasets (`yelp_polarity`) |
| Original size | 560,000 train / 38,000 test |
| Subset used | 10,000 train (5,000 pos + 5,000 neg) / 4,000 test (2,000 pos + 2,000 neg) |
| Classes | 2 (Positive, Negative) |

**Why subsampled?** Full dataset training exceeded free Google Colab GPU limits. The balanced 10K/4K subset enabled stable training while preserving class balance.

---

## Models

### ULMFiT (AWD_LSTM via fastai)

ULMFiT is a transfer learning approach for NLP that fine-tunes a pretrained language model before adding a task-specific classification head.

**Stage 1 — Language Model Fine-Tuning**
- Base model: AWD_LSTM pretrained on WikiText-103 (103M words)
- Fine-tuned for 1 epoch on Yelp training corpus
- Learning rate: 0.01 | Dropout multiplier: 0.3
- Goal: Adapt pretrained weights to Yelp vocabulary and review style

**Stage 2 — Classifier Fine-Tuning**
- Classification head added on top of saved encoder
- 2 frozen epochs → 2 fully unfrozen epochs (gradual unfreezing)
- Dropout multiplier: 0.5 to reduce overfitting

### Naive Bayes (TF-IDF Baseline)

- TF-IDF vectorization with bigrams, max 50,000 features
- Multinomial Naive Bayes classifier
- No pretraining, no GPU required
- Entire pipeline: < 10 lines of code, completes in seconds

---

## Results

**Overall accuracy:**

| Model | Accuracy |
|---|---|
| ULMFiT (AWD_LSTM) | **94.20%** |
| Naive Bayes (TF-IDF) | 90.20% |

**Per-class performance:**

| Model | Negative Recall | Positive Recall |
|---|---|---|
| ULMFiT | 95% | 93% |
| Naive Bayes | 91% | 90% |

ULMFiT outperforms the baseline by 4 percentage points overall, with the most pronounced advantage in identifying negative sentiment — critical for customer service applications where detecting dissatisfied customers quickly enables faster service recovery.

---

## Development Effort Comparison

| Dimension | ULMFiT | Naive Bayes |
|---|---|---|
| Setup complexity | High (weights, fine-tuning, encoder, classifier) | Low (vectorizer + classifier) |
| Training time | Several hours (GPU required) | Seconds (CPU) |
| Code volume | ~50+ lines | < 10 lines |
| Infrastructure | GPU required | CPU sufficient |
| Reproducibility | Session-sensitive (Colab disconnects) | Fully reproducible |

---

## Recommendation

For production deployment, **ULMFiT is the recommended model**. The 4-point accuracy advantage is meaningful at scale — for a platform processing millions of reviews, that translates to significantly fewer misclassifications. The stronger negative recall (95% vs 91%) is particularly valuable for customer experience management, where identifying dissatisfied customers quickly matters most.

Naive Bayes is appropriate for lightweight prototyping or severely resource-constrained environments, but for a production customer sentiment system, the accuracy and robustness of ULMFiT justifies the additional development effort.

---

## Tech Stack

- Python, fastai, PyTorch
- AWD_LSTM (pretrained on WikiText-103)
- scikit-learn (TF-IDF + Multinomial Naive Bayes)
- Hugging Face Datasets (`yelp_polarity`)
- Matplotlib (accuracy comparison chart, confusion matrices)
- Google Colab (T4 GPU)

---

## Repository Structure

```
ulmfit-yelp-sentiment/
├── notebook.ipynb       # Full training pipeline and evaluation
├── README.md
```

---

## References

- Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *ACL 2018*, 328–339. https://arxiv.org/abs/1801.06146
- Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *NeurIPS*, 28, 649–657. https://arxiv.org/abs/1509.01626
- Thompson, N. C., et al. (2020). The computational limits of deep learning. arXiv. https://arxiv.org/abs/2007.05558
