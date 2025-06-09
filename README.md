# Transformer From Scratch

**Rebuilding *"Attention Is All You Need"* step by step**

A complete from-scratch implementation of the Transformer architecture (Vaswani et al., 2017) using **PyTorch** and **NumPy**.
This project aims to provide an **educational, minimal, and fully transparent codebase** that demystifies the core concepts of:

* Scaled Dot-Product Attention
* Multi-Head Attention
* Positional Encoding
* FeedForward Networks
* Encoder & Decoder Blocks
* Full Transformer Model

with **detailed explanations and visualizations**.

---

# Transformer Encoder Blocks (PyTorch Implementation)

This repository contains a modular and educational PyTorch implementation of the **core building blocks of the Transformer Encoder** architecture as introduced in the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

---

## Architecture Overview

![Transformer arch](https://github.com/user-attachments/assets/f72c0f8c-df8b-446b-8f9b-0831e2ab8cc2)

---

## Implemented Components

### 1️⃣ **Input Embeddings**

* Learnable word embeddings.
* Scaled by √(d\_model) for stable gradients.

### 2️⃣ **Positional Encodings**

* Injects positional information using sine/cosine functions.
* Makes the model aware of the sequence order.

<img src="https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png" width="600">

### 3️⃣ **Layer Normalization**

* Normalizes inputs across the feature dimension.
* Ensures stable training and improved convergence.

### 4️⃣ **FeedForward Block**

* A simple MLP applied independently to each position.
* Activation: ReLU → Dropout → Linear.

### 5️⃣ **Multi-Head Attention**

* Computes attention using multiple heads in parallel.
* Projects queries, keys, and values, then performs scaled dot-product attention.
* Allows the model to focus on different parts of the sequence simultaneously.

 <img src="https://jalammar.github.io/images/t/transformer_multi-head-attention-recap.png" width="600">

### 6️⃣ **Residual Connection Block**

* Adds skip connections and layer normalization.
* Helps preserve gradients in deep architectures.


<img src="https://jalammar.github.io/images/t/transformer_decoder_residual_layer_norm.png" width="500">


### 7️⃣ **Encoder Block**

* Stacks:

  * Multi-Head Self-Attention + Residual Connection.
  * FeedForward + Residual Connection.

### 8️⃣ **Encoder**

* Stack of multiple `EncoderBlock` layers.
* Final Layer Normalization applied.

---

## Transformer Encoder Block Diagram

<img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png" width="500">  
*(Image credit: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar)*

---

# Full Transformer Model (Encoder-Decoder Architecture)

In addition to the **Encoder blocks**, this project also implements the full **Transformer Encoder-Decoder stack**, including:

### 9️⃣ **Decoder Block**

* **Masked Multi-Head Self-Attention**:

  * Prevents attending to future tokens.
* **Cross-Attention**:

  * Attends to encoder outputs.
* FeedForward + Residual Connections.

### 🔟 **Decoder**

* Stack of multiple DecoderBlocks.
* Final Layer Normalization applied.

## Transformer Decoder Block Diagram

<img src="https://jalammar.github.io/images/t/transformer_decoder_block.png" width="500">

*(Image credit: The Illustrated Transformer - Jay Alammar)*


### Projection Layer

* Projects decoder outputs to logits over the target vocabulary.
<img src="https://jalammar.github.io/images/t/transformer_logit_softmax.png" width="400">
---

# Build Function

```python
build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer
```

Builds a complete Transformer model with configurable hyperparameters:

* N: number of encoder/decoder blocks
* h: number of attention heads
* d\_model: embedding/hidden size
* d\_ff: size of feedforward layers
* dropout: dropout rate

---

# Example Usage

```python
# Build model
transformer = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    src_seq_len=50,
    tgt_seq_len=50
)

# Dummy input tensors
src = torch.randint(0, 10000, (32, 50))  # (batch_size, src_seq_len)
tgt = torch.randint(0, 10000, (32, 50))  # (batch_size, tgt_seq_len)

# Masks (None here, but should be added for real use)
src_mask = None
tgt_mask = None

# Forward pass
encoder_out = transformer.encode(src, src_mask)
decoder_out = transformer.decode(encoder_out, src_mask, tgt, tgt_mask)
output_logits = transformer.project(decoder_out)

print(output_logits.shape)  # Expected: (batch_size, tgt_seq_len, tgt_vocab_size)
```

---

# Notes & Limitations

✅ **Fully functional** — implements all core ideas of the original Transformer paper.
✅ Works with arbitrary vocab sizes and sequence lengths.
✅ Implements **layer normalization, multi-head attention, residual connections** manually.
✅ All layers initialized with **Xavier Uniform**.

⚠️ For production use:

* Add proper attention masks.
* Add training loop and optimizer.
* Add loss computation (e.g., CrossEntropyLoss).

---

# Summary

This code is designed to help understand and reproduce the original **Transformer architecture** step by step:

✅ Self Attention
✅ Cross Attention
✅ Stacked Encoder & Decoder blocks
✅ Position-wise FFN
✅ Positional Encoding
✅ Layer Normalization
✅ Residual Connections
✅ Final projection to vocab logits

---

# References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017.
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar.

---

# License

MIT License — free to use and modify.
