# Transformer From Scratch (No PyTorch nn)

This project implements a Transformer model completely from scratch using PyTorch tensors only, without using `torch.nn` modules such as `nn.Linear`, `nn.LayerNorm`, or `nn.Transformer`.

The purpose of this project is to understand how modern architectures like OpenAI GPT and Google BERT work internally at a low level.

---

## Overview

This implementation includes:

* Custom Linear layer (manual weight and bias)
* Custom Embedding layer
* Sinusoidal Positional Encoding
* Scaled Dot-Product Attention
* Multi-Head Attention
* Feed Forward Network
* Layer Normalization
* Encoder stack
* Manual training loop (no optimizer API)

---

## Architecture

The Transformer processes input in the following order:

Text → Embedding → Positional Encoding → Multi-Head Attention → Feed Forward → Output

---

## Motivation

High-level libraries abstract away important details of how neural networks operate.

This project focuses on:

* Understanding the math behind Transformers
* Implementing each component manually
* Observing how gradients and parameters behave

---

## Mathematical Formulation

### Linear Layer

y = xW + b

---

### Scaled Dot-Product Attention

Attention(Q, K, V) = softmax((QKᵀ) / √dₖ) V

---

### Positional Encoding

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

---

### Layer Normalization

LayerNorm(x) = (x - μ) / √(σ² + ε) * γ + β

---

## Training

The current version uses randomly generated data for demonstration purposes.
As a result, the loss does not decrease significantly because there is no meaningful pattern to learn.

To make training effective:

* Use real text data
* Implement next-token prediction (language modeling)
* Add causal masking (for GPT-style models)

---

## Future Improvements

* Implement decoder (full Transformer)
* Add causal masking
* Build tokenizer (BPE or WordPiece)
* Train on real datasets
* Implement Adam optimizer from scratch
* Add text generation

---

## Conclusion

This project provides a low-level, transparent implementation of a Transformer, helping build a strong conceptual foundation for understanding large language models.


## References

This implementation is inspired by the following foundational research papers in Transformer-based architectures:

### Core Transformer Paper

* Vaswani, A., et al. (2017). *Attention Is All You Need*
  https://arxiv.org/abs/1706.03762

---

### Encoder-based Models

* Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
  https://arxiv.org/abs/1810.04805

---

### Decoder-based Models (GPT Series)

* Radford, A., et al. (2018). *Improving Language Understanding by Generative Pre-Training*
  https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

* Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2)*
  https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

* Brown, T., et al. (2020). *Language Models are Few-Shot Learners (GPT-3)*
  https://arxiv.org/abs/2005.14165

---

### Additional Reading

* Efficient Transformer Variants and Attention Improvements
  https://arxiv.org/abs/2308.07661

---

These papers provide the theoretical foundation for modern large language models and directly influenced this implementation.
