# Transformer From Scratch  
Rebuilding "Attention Is All You Need" step by step  

A complete from-scratch implementation of the Transformer architecture (Vaswani et al., 2017) using PyTorch and NumPy. This project aims to provide an educational, minimal, and fully transparent codebase that demystifies the core concepts of attention, multi-head attention, positional encoding, and Transformer blocks — with detailed explanations and visualizations.
# Transformer Encoder Blocks (PyTorch Implementation)

This repository contains a modular and educational PyTorch implementation of the core building blocks of the Transformer Encoder architecture as introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

---

## Architecture Overview

![Transformer arch](https://github.com/user-attachments/assets/f72c0f8c-df8b-446b-8f9b-0831e2ab8cc2) width="600"


*(Image credit: The Illustrated Transformer - Jay Alammar)*

---

## Implemented Components

This code implements the following modules:

### 1️⃣ **Input Embeddings**

- Learnable word embeddings.
- Scaled by √(d_model) for stable gradients.

### 2️⃣ **Positional Encodings**

- Injects positional information using sine/cosine functions.
- Makes the model aware of the sequence order.

### 3️⃣ **Layer Normalization**

- Normalizes inputs across the feature dimension.
- Ensures stable training and improved convergence.

### 4️⃣ **FeedForward Block**

- A simple MLP applied independently to each position.
- Activation: ReLU → Dropout → Linear.

### 5️⃣ **Multi-Head Attention**

- Computes attention using multiple heads in parallel.
- Projects queries, keys, and values, then performs scaled dot-product attention.
- Allows the model to focus on different parts of the sequence simultaneously.

### 6️⃣ **Residual Connection Block**

- Adds skip connections and layer normalization.
- Helps preserve gradients in deep architectures.

### 7️⃣ **Encoder Block**

- Stacks:
    - Multi-Head Self-Attention + Residual Connection.
    - FeedForward + Residual Connection.

### 8️⃣ **Encoder**

- Stack of multiple `EncoderBlock` layers.
- Final Layer Normalization applied.

---

## Transformer Encoder Block Diagram

<img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png" width="500">

*(Image credit: The Illustrated Transformer - Jay Alammar)
