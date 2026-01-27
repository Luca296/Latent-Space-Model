# Latent‑Space Reasoning Stack

This document describes a prototype architecture where a model **thinks in latent "idea space"** instead of directly in token space.
It explains:

- The overall design (encoder → latent → middle model → decoder)
- How each component is wired together
- How to train the system end‑to‑end (within the limits of a single RTX 3060)
- Suggested datasets and practical training recipes

---

## 1. High‑level architecture

### 1.1 Conceptual overview

The stack is split into three main modules:

1. **Encoder (ModernBERT)**
   - Input: text tokens
   - Output: a compact **idea vector** (latent representation)

2. **Middle model (latent reasoning core)**
   - Input: idea vector
   - Output: transformed idea vector (e.g., "summary idea", "answer idea")

3. **Decoder (GPT‑style LM)**
   - Input: transformed idea vector (mapped into its embedding space)
   - Output: text tokens

> **Key design choice:**
> **All reasoning happens in latent space, not in token space.**
> Tokens are only used at the very beginning (encoding) and the very end (decoding).

---

## 2. Components in detail

### 2.1 Encoder: ModernBERT → idea vector

**Goal:** Turn a sequence of token embeddings into a single fixed‑size latent "idea" vector.

**Inputs:**

- Tokenized text: `input_ids`, `attention_mask`
- Model: `ModernBERT` (base or small, frozen or mostly frozen)

**Steps:**

1. **Run ModernBERT**
   - Get last hidden states: shape `[batch_size, seq_len, hidden_dim]`.

2. **Pooling layer** (trainable or simple):
   - **Option A (simple):** mean pooling over non‑padded tokens.
   - **Option B (better):** attention pooling or using the `[CLS]` token plus a small MLP.

3. **Compression head (MLP):**
   - Map from `hidden_dim` → `latent_dim` (e.g., 768 → 256).
   - This is the **idea vector**.

**Output:**

- `z_in`: latent idea vector, shape `[batch_size, latent_dim]`.

---

### 2.2 Middle model: latent reasoning core

**Goal:** Operate purely on idea vectors, transforming one idea into another.

**Design options:**

- **Small MLP:** 2–4 layers, with non‑linearities (GELU/ReLU).
- **Tiny Transformer:** treat the latent as a sequence of length 1 or a small set of latent slots.
- **GRU/LSTM:** if multi‑step latent reasoning is desired later.

For a first prototype, a **2–4 layer MLP** is sufficient.

**Example:**

- Input: `z_in` (256‑dim)
- Middle model: MLP → 256‑dim
- Output: `z_out` (256‑dim)

This `z_out` is the **transformed idea** (e.g., "summary idea").

---

### 2.3 Decoder: GPT‑style LM with latent prefix

**Goal:** Turn `z_out` into text using a pretrained language model (e.g., GPT‑2 small).

GPT‑2 normally expects **token embeddings**, not a single vector. This is bridged with a **prefix adapter**.

#### 2.3.1 Prefix adapter

**Idea:** Convert `z_out` into a sequence of "soft prompt" embeddings that GPT‑2 uses as context.

1. **Expansion MLP:**
   - Input: `z_out` (latent_dim)
   - Output: `prefix_len × lm_hidden_dim` (e.g., 10 × 768)
   - Reshape to `[batch_size, prefix_len, lm_hidden_dim]`.

2. **Use as prefix:**
   - Concatenate prefix embeddings with the embedding of a BOS token.
   - GPT‑2 then generates tokens conditioned on this prefix.

**Output:**

- Generated text sequence.

---

## 3. Data flow and shapes

Assume:

- `ModernBERT hidden_dim = 768`
- `latent_dim = 256`
- `GPT‑2 hidden_dim = 768`
- `prefix_len = 10`

**Forward pass:**

1. **Encoder:**

   - Input: tokens → ModernBERT → `[B, L, 768]`
   - Pooling: `[B, 768]`
   - Compression MLP: `[B, 256] = z_in`

2. **Middle model:**

   - MLP: `[B, 256] → [B, 256] = z_out`

3. **Decoder adapter:**

   - Expansion MLP: `[B, 256] → [B, 10 × 768] → reshape → `[B, 10, 768]`
   - Concatenate with BOS embedding: `[B, 11, 768]`
   - GPT‑2 generates text autoregressively.

---

## 4. Training strategy

### 4.1 Trainable vs. frozen components

To fit on an RTX 3060, the **big models are frozen** and only small heads are trained:

- **Frozen (or mostly frozen):**
  - ModernBERT backbone
  - GPT‑2 backbone

- **Trainable:**
  - Encoder pooling/compression head
  - Middle model (latent reasoning core)
  - Decoder expansion/prefix adapter

Optionally, **LoRA/QLoRA** can be used on ModernBERT and GPT‑2 for light adaptation, but starting with them fully frozen is recommended.

### 4.2 Task choice

For a first prototype, a **supervised text‑to‑text task** is recommended where:

- Input: text A
- Output: text B

Good candidates:

- Summarization (document → summary)
- Dialogue summarization (chat → summary)
- Paraphrasing (sentence → rephrased sentence)
- Question answering (question + context → answer)

Summarization is a clean, intuitive starting point.

### 4.3 Dataset suggestions

Desired characteristics include:

- Not too large
- Easy to load with Hugging Face Datasets or Kaggle
- Already used for summarization

**Good smallish option:**

- **SAMSum** — dialogue summarization dataset (chat logs + human summaries). It's relatively small (~8 MB) and well‑suited for prototyping summarization models.

**Larger options (for scaling later):**

- **CNN/DailyMail** — news articles + highlights, classic summarization benchmark.

For the given hardware and first prototype, **SAMSum** is ideal.

### 4.4 Training objective

The system is trained end‑to‑end (for the trainable parts) using **standard language modeling loss** on the decoder:

- Let `y` be the target output text (e.g., summary).
- Tokenize `y` into `y_ids`.
- Condition GPT‑2 on the latent prefix from `z_out`.
- Compute cross‑entropy loss between GPT‑2's predicted logits and `y_ids`.

**Loss:**

$$
\mathcal{L} = - \sum_{t} \log p(y_t \mid y_{<t}, \text{latent prefix})
$$

Gradients flow back through:

- GPT‑2 input embeddings (if unfrozen or LoRA‑tuned)
- Prefix adapter (expansion MLP)
- Middle model
- Encoder compression head
- (Optionally) ModernBERT, if partially unfrozen

### 4.5 Training loop (conceptual)

#### 4.5.1 Single example flow

1. **Input text:** `x` (e.g., article or dialogue)
2. **Target text:** `y` (e.g., summary)

**Forward:**

1. Tokenize `x` → `input_ids_x`, `attention_mask_x`
2. ModernBERT → hidden states → pooling → compression → `z_in`
3. Middle model → `z_out`
4. Expansion MLP → prefix embeddings
5. GPT‑2:
   - Condition on prefix
   - Teacher forcing with `y_ids`
   - Compute logits
6. Compute cross‑entropy loss vs. `y_ids`

**Backward:**

- `loss.backward()`
- Optimizer step on:
  - compression head
  - middle model
  - expansion head
  - (optionally) LoRA adapters

### 4.6 Pseudocode sketch

```python
# Pseudocode, not exact API
for batch in dataloader:
    x_text = batch["input"]
    y_text = batch["target"]

    # 1. Encode input with ModernBERT
    x_tokens = modernbert_tokenizer(
        x_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        enc_outputs = modernbert(**x_tokens)
    hidden_states = enc_outputs.last_hidden_state  # [B, L, 768]

    # 2. Pool + compress to latent
    pooled = pooling_layer(hidden_states, x_tokens["attention_mask"])  # [B, 768]
    z_in = compression_mlp(pooled)  # [B, 256]

    # 3. Middle model
    z_out = middle_model(z_in)  # [B, 256]

    # 4. Expansion to GPT-2 prefix
    prefix = expansion_mlp(z_out)  # [B, prefix_len * 768]
    prefix = prefix.view(batch_size, prefix_len, gpt2_hidden_dim)  # [B, P, 768]

    # 5. Prepare decoder inputs
    y_tokens = gpt2_tokenizer(
        y_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    # Get token embeddings for y (teacher forcing)
    y_input_embeds = gpt2.transformer.wte(y_tokens["input_ids"])  # [B, T, 768]

    # 6. Concatenate prefix + y embeddings
    input_embeds = torch.cat(
        [prefix, y_input_embeds[:, :-1, :]], dim=1
    )  # shift for LM

    # 7. Run GPT-2 with custom embeddings
    outputs = gpt2(inputs_embeds=input_embeds, attention_mask=None)
    logits = outputs.logits[:, prefix_len:, :]  # align with y_tokens

    # 8. Compute loss
    loss = cross_entropy_loss(logits, y_tokens["input_ids"])

    # 9. Backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 5. Practical considerations

### 5.1 Hardware constraints (RTX 3060, 32 GB RAM)

- Use **ModernBERT base or small**.
- Use **GPT‑2 small or similar‑sized** model.
- Use mixed precision (fp16/bf16) if possible.
- Keep batch sizes small (e.g., 4–8).
- Start with shorter max sequence lengths (e.g., 256 tokens).

### 5.2 Hyperparameter starting points

| Parameter | Value | Reason |
|-----------|-------|--------|
| `latent_dim` | 256 | Compact yet expressive |
| `prefix_len` | 10–20 | Enough context for GPT‑2 |
| `middle model` | 2–4 layers MLP | Simple, fast |
| MLP hidden size | 512 | Non‑linear capacity |
| learning rate | 1e‑4 – 3e‑4 | Stable for heads |
| optimizer | AdamW | Standard |
| epochs | 3–5 | Small dataset |

### 5.3 Evaluation

- **Summarization**: ROUGE (R‑1, R‑2, F1).
- **Paraphrasing**: BLEU, METEOR.
- **Latent behavior**:
  - Interpolate between two latent vectors and decode.
  - Check if similar inputs produce similar latents.
  - Cluster latents and inspect decoded outputs.

The goal of the prototype is not SOTA performance, but to demonstrate:

- The system can encode → reason → decode.
- The middle model actually changes the output in meaningful ways.

---

## 6. How this differs from a normal encoder–decoder

Traditional encoder–decoder models (like T5, BART) also have:

- Encoder → latent representations
- Decoder → text

However:

- They usually don't expose a single compact "idea vector".
- They don't insert a separate, explicit latent reasoning core that operates on a compressed idea.
- They are trained end‑to‑end in token space, not explicitly in a compact latent space.

This stack explicitly:

- Compresses to a single (or small set of) idea vectors.
- Forces reasoning to happen in that latent space.
- Uses a separate decoder that only sees the transformed latent.

This is closer to world models and latent‑space reasoning architectures than to standard seq2seq.

---

## 7. Roadmap for prototype development

**Phase 1 — Minimal working system**

1. Use SAMSum as dataset.
2. Freeze ModernBERT and GPT‑2.
3. Implement:
   - Pooling + compression head
   - 2‑layer MLP middle model
   - Expansion MLP → GPT‑2 prefix
4. Train on a subset (e.g., 10k examples).
5. Check if the model can produce rough summaries.

**Phase 2 — Improve latent quality**

- Experiment with different pooling strategies.
- Larger `latent_dim` (e.g., 512).
- Slightly deeper middle model.
- Optionally add LoRA to ModernBERT and GPT‑2.

**Phase 3 — Explore "idea space"**

- Interpolate between latents and decode.
- Visualize latents with PCA/TSNE.
- Try different tasks (paraphrasing, style transfer).

---

## 8. Mental model to keep in mind

The system being built operates as follows:

```
ModernBERT : "What is this text about?" → idea vector
Middle model : "What should I do with this idea?" → transformed idea
GPT‑2     : "How do I say this idea in language?" → text