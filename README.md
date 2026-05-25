# LLM from Scratch — Karpathy Transformer Recreation

> **Disclaimer:** This project is a personal recreation of the transformer-based language model built by Andrej Karpathy in his YouTube tutorial
> [**"Let's build GPT: from scratch, in code, spelled out."**](https://www.youtube.com/watch?v=kCc8FmEb1nY).
>
> This is **my own code**, written step by step while following the video — it is not a clone or fork of Karpathy's repository. The code reflects my own understanding of the material, and includes my own comments and key notes (written in Spanish) that help me reason through how each component is built and why it works the way it does.

---

## What this project is

A character-level language model trained on a corpus of Shakespeare text (`tinyshakespeare.txt`). Once trained, the model generates new text that mimics Shakespeare's style.

The vocabulary is built directly from the characters found in the text — every letter, punctuation mark, and symbol becomes a token. This is the most basic form of tokenization possible and is intentionally kept simple for learning purposes.

---

## Project structure

```
.
├── tinyshakespeare.txt                         # Training corpus
├── bigram_model/
│   ├── bigram_language_model.py                # Simple bigram model (no attention)
│   └── train.py                                # Training loop for the bigram model
├── toy_example_for_self_attention/
│   └── toy_example_self_attention.ipynb        # Notebook exploring self-attention step by step
└── transformer/
    ├── bigram_language_model.py                # Full transformer model (multi-head attention)
    └── train.py                                # Training loop for the transformer
```

---

## Learning progression

The project follows a deliberate progression from the simplest possible model to a full transformer:

### 1. Bigram model (`bigram_model/`)

The starting point. Each token prediction depends only on the immediately preceding token. There is no attention mechanism — just a lookup table (`nn.Embedding`) that maps each token to a distribution over the next token.

- `BigramLanguageModel`: a single embedding table where `logits = embedding[token]`
- `train.py`: character-level tokenization, 90/10 train/val split, AdamW optimizer, loss evaluation every 1000 epochs

### 2. Self-attention toy example (`toy_example_for_self_attention/`)

A Jupyter notebook that walks through the math of self-attention from scratch before any model code is written:

- Bag-of-words (averaging past token vectors) via lower-triangular matrix multiplication
- Building the Q, K, V matrices with `nn.Linear`
- Computing attention weights: `softmax(QK^T / sqrt(head_size))`
- Masking future tokens with `torch.tril` to implement a decoder (autoregressive) style

### 3. Full transformer (`transformer/`)

The complete model, built from the concepts explored in the notebook:

| Component | Class | Role |
|---|---|---|
| Single attention head | `Head` | Computes scaled dot-product attention for one head |
| Multi-head attention | `MultiHeadAttention` | Runs `num_heads` heads in parallel and concatenates their outputs |
| Feed-forward block | `FeedForward` | Two-layer MLP with 4x expansion, applied per token |
| Transformer block | `Block` | Combines multi-head attention + feed-forward with residual connections and LayerNorm |
| Full model | `BigramLanguageModel` | Token embeddings + positional embeddings + stacked blocks + language model head |

---

## Tokenization

The vocabulary is built by collecting every unique character in `tinyshakespeare.txt`:

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }   # character -> integer
itos = { i:ch for i,ch in enumerate(chars) }   # integer -> character
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])
```

Each character is its own token. No subword splitting, no BPE — just raw characters. This keeps the model simple and the vocabulary small (~65 tokens for Shakespeare text), which is ideal for understanding the fundamentals.

---

## Hyperparameters (transformer)

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 32 | Independent sequences processed in parallel |
| `block_size` | 8 | Context length (tokens seen before predicting the next) |
| `n_embd` | 32 | Embedding dimension |
| `num_heads` | 4 | Number of attention heads |
| `num_layers` | 4 | Number of stacked transformer blocks |
| `dropout` | 0.2 | Dropout rate for regularization |
| `lr` | 1e-3 | Learning rate (AdamW) |
| `epochs` | 5001 | Training steps |

---

## How to run

```bash
# Train the bigram model
cd bigram_model
python train.py

# Train the full transformer
cd transformer
python train.py
```

Both scripts print train/val loss at regular intervals and generate a 500-character sample at the end.

**Requirements:** Python 3.x, PyTorch. A CUDA GPU is used automatically if available, otherwise falls back to CPU.

---

## Reference

- Video tutorial: [Let's build GPT: from scratch, in code, spelled out — Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Original paper: [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
