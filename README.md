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
