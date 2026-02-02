# Middle Model Architecture: A Technical Deep Dive

## Executive Summary

The **middle model** in this latent-space reasoning architecture is a **MiddleTransformer**—a compact transformer-based reasoner that operates on a fixed-length latent sequence. It is **not just a bottleneck**, but a genuine latent reasoning core with multi-head attention, RoPE, layer normalization, and SwiGLU activations. The middle model learns to transform "idea vectors" (semantic representations) from one state to another, enabling the system to perform latent-space "reasoning" before decoding to text.

---

## 1. Architecture Overview

### 1.1 Component Stack

The full data flow is:

```
Text Input (tokens)
    ↓
[LatentEncoder: ModernBERT + Pooling + Compression MLP]
    ↓
Latent Sequence: [B, latent_seq_len=8, latent_dim=256]
    ↓
[MiddleTransformer: 4 transformer blocks with self-attention]
    ↓
Transformed Latent: [B, 8, 256]
    ↓
[PrefixAdapter: Cross-attention from latent to prefix embeddings]
    ↓
Prefix Embeddings: [B, prefix_len=50, gpt2_hidden_dim=640]
    ↓
[GPT-2 Decoder: Gemma-3-270m-it for text generation]
    ↓
Text Output (tokens)
```

### 1.2 Key Configuration Parameters

From [src/config.py](src/config.py):

```python
# Latent space dimensions
latent_dim: int = 256                    # Dimension of each latent "idea" vector
latent_seq_len: int = 8                  # Number of latent vectors per sample (fixed)

# Middle model architecture
middle_transformer_layers: int = 4       # Number of transformer blocks
middle_num_heads: int = 4                # Attention heads per layer
middle_ffn_multiplier: float = 4.0       # FFN hidden dim = latent_dim * 4.0
middle_dropout: float = 0.1              # Dropout rate
middle_use_rope: bool = True             # Rotary positional embeddings
```

---

## 2. Middle Model Architecture (MiddleTransformer)

### 2.1 Class Definition

**File:** [src/models.py](src/models.py), lines 332–374

```python
class MiddleTransformer(nn.Module):
    """Transformer for latent reasoning over sequences."""
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: int = 10000
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([
            TransformerBlock(...)
            for _ in range(num_layers)
        ])
```

### 2.2 What the Middle Model Actually Does

**Input:**
- `z_in`: shape `[B, latent_seq_len=8, latent_dim=256]`
- A sequence of 8 latent vectors, each 256-dimensional

**Processing (per transformer block):**

Each `TransformerBlock` contains:

1. **Layer Norm (RMSNorm)**: `[B, 8, 256] → [B, 8, 256]`
   - Root Mean Square normalization (see [src/models.py](src/models.py), lines 34–43)

2. **Multi-Head Self-Attention**: `[B, 8, 256] → [B, 8, 256]`
   - 4 attention heads
   - **With RoPE** (Rotary Positional Embeddings)
   - Enables each latent vector to attend to all other latent vectors in the sequence
   - Head dimension: 256 / 4 = 64 (even, required for RoPE)

3. **Residual Connection**: `z + attn(norm(z))`

4. **Layer Norm (RMSNorm)**: `[B, 8, 256] → [B, 8, 256]`

5. **SwiGLU FFN**: 
   - Up-project: 256 → 1024 (4× expansion)
   - SiLU gating + linear projection
   - Down-project: 1024 → 256
   - Dropout applied

6. **Residual Connection**: `z + ffn(norm(z))`

**Output:**
- `z_out`: shape `[B, 8, 256]`
- Transformed latent sequence, same shape as input

**Key Insight:** The middle model operates on **a latent sequence** (not a single latent vector). This means:
- Multiple latent vectors (8 of them) can interact with each other via self-attention
- Each vector can attend to all positions in the sequence
- The transformer can learn to reorganize, combine, or selectively emphasize latent features

### 2.3 Detailed Component Breakdown

#### TransformerBlock (lines 309–327)

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_multiplier: float = 4.0, 
                 dropout: float = 0.0, use_rope: bool = True, rope_base: int = 10000):
        super().__init__()
        hidden_dim = int(dim * ffn_multiplier)  # 256 * 4 = 1024
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, use_rope, rope_base)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            SwiGLU(dim, hidden_dim),              # 256 → 1024
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),           # 1024 → 256
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))         # Pre-norm, residual
        x = x + self.ffn(self.norm2(x))          # Pre-norm, residual
        return x
```

#### MultiHeadSelfAttention with RoPE (lines 214–279)

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, 
                 use_rope: bool = True, rope_base: int = 10000):
        super().__init__()
        self.head_dim = dim // num_heads  # 256 / 4 = 64
        self.qkv = nn.Linear(dim, dim * 3, bias=True)  # 256 → 768
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len=8, 256]
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)                    # [B, 8, 768]
        q, k, v = qkv.chunk(3, dim=-1)       # Each [B, 8, 256]
        
        # Reshape to [B, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)  # [B, 8, 4, 64]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE (if enabled)
        if self.use_rope:
            cos, sin = self.rope(seq_len, x.device, x.dtype)  # seq_len=8
            q = apply_rope(q.reshape(...), cos, sin)           # RoPE on 64-dim vectors
            k = apply_rope(k.reshape(...), cos, sin)
        
        # Standard scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)       # Softmax over [8, 8]
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back and project
        return self.out_proj(attn_output.view(...))
```

#### RoPE (Rotary Positional Embeddings)

From [src/models.py](src/models.py), lines 59–86:

- Applied to query and key vectors in 64-dimensional space
- Encodes positional information directly into vector rotations
- Allows the model to learn relative position dependencies

#### RMSNorm (Root Mean Square Normalization)

From [src/models.py](src/models.py), lines 34–43:

```python
class RMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight  # Learnable scale
```

- More efficient than LayerNorm
- Often used in modern transformers (LLaMA, GPT-NeoX, etc.)

#### SwiGLU (Gated Linear Unit)

From [src/models.py](src/models.py), lines 47–54:

```python
class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)  # 256 → 2048 (for 1024 out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)                       # [B, 8, 2048]
        x_gated, x_linear = x_proj.chunk(2, dim=-1) # Each [B, 8, 1024]
        return F.silu(x_gated) * x_linear           # Gated activation
```

- Two-part gating mechanism
- SiLU (swish) activation on one projection, multiplied with the other
- More expressive than simple GELU/ReLU

---

## 3. Complete Data Flow with Tensor Shapes

### 3.1 Full Forward Pass

**Input to middle model:**
```
z_in: torch.Tensor
  Shape: [B=10, latent_seq_len=8, latent_dim=256]
  Dtype: float32 or bfloat16 (depends on config.use_bf16)
```

**Inside MiddleTransformer.forward():**

```
z_out = z_in  # [10, 8, 256]

for i in range(4):  # 4 transformer blocks
    block = self.layers[i]  # TransformerBlock
    
    # Block 1: Pre-norm + attention + residual
    z_norm = RMSNorm(z_out)                         # [10, 8, 256]
    attn_out = MultiHeadSelfAttention(z_norm)       # [10, 8, 256]
    z_out = z_out + attn_out                        # Residual [10, 8, 256]
    
    # Block 2: Pre-norm + FFN + residual
    z_norm = RMSNorm(z_out)                         # [10, 8, 256]
    z_norm_expanded = SwiGLU(z_norm, 1024)         # [10, 8, 1024]
    z_norm_projected = Linear(1024, 256)(z_norm_expanded)  # [10, 8, 256]
    z_out = z_out + z_norm_projected                # Residual [10, 8, 256]

    # z_out now shape [10, 8, 256]

return z_out  # [10, 8, 256]
```

### 3.2 Attention Pattern Details

For each head at each layer:

```
Input: [B=10, seq_len=8, head_dim=64]

Q = W_q @ z                                  # [10, 8, 64]
K = W_k @ z                                  # [10, 8, 64]
V = W_v @ z                                  # [10, 8, 64]

# Apply RoPE
Q_rope = apply_rope(Q, cos, sin)             # [10, 8, 64] with positional info
K_rope = apply_rope(K, cos, sin)             # [10, 8, 64] with positional info

# Attention scores
scores = (Q_rope @ K_rope^T) / sqrt(64)      # [10, 8, 8] attention matrix
probs = softmax(scores, dim=-1)              # [10, 8, 8], each row sums to 1

# Output for this head
output = probs @ V                           # [10, 8, 64]
```

**Key observation:** The attention weights form an `[8, 8]` matrix for each batch item and head. This means:
- Each of the 8 latent vectors learns to attend to all other latent vectors
- The model can learn which latent features are important at each position
- Positional information is encoded via RoPE

---

## 4. What the Middle Model Really Does

### 4.1 Is It Just a Bottleneck?

**No.** The middle model is **not** a passive bottleneck or projection layer. It is an **active transformer reasoning core** with these capabilities:

| Aspect | Verdict |
|--------|---------|
| **Input/output shape match** | ✓ Yes: `[B, 8, 256] → [B, 8, 256]` (identity-like, but transformed) |
| **Contains learnable nonlinearity** | ✓ Yes: SwiGLU activations, 4 transformer blocks |
| **Supports multi-step reasoning** | ✓ Yes: 4 transformer blocks enable sequential, iterative refinement |
| **Has attention mechanism** | ✓ Yes: 4 heads × 4 layers = 16 distinct attention patterns |
| **Encodes positional structure** | ✓ Yes: RoPE on each layer enables relative reasoning |
| **Parameter count** | ~1.6M parameters (non-trivial) |

### 4.2 Conceptual Role

The middle model can be characterized as:

1. **A semantic transformer**: Transforms one semantic representation (input idea) into another (output idea)
   - Not operating on tokens, but on latent semantic features
   
2. **A latent-space reasoner**: Uses attention to perform implicit reasoning
   - The 8 latent vectors might represent different aspects (e.g., subject, verb, object, sentiment, etc.)
   - Attention enables the model to reorganize these features

3. **A learned transformation**: Each training task (e.g., summarization) provides a supervision signal for what output semantics are desired
   - For summarization: `text → summary idea`
   - The middle model learns to transform task-specific input ideas into appropriate output ideas

### 4.3 Capacity Analysis

**Total parameters in MiddleTransformer:**

```
Per TransformerBlock:
  - RMSNorm (2x): 256 * 2 = 512 params
  - Attention: Q, K, V, Out = 256*256*4 + bias = 262,144 + 256 = 262,400 params
  - SwiGLU: proj (256 → 2048) + Linear (2048 → 256) = 256*2048*2 + 2048 + 256*2048 + 256
           ≈ 1.05M params
  - Subtotal per block: ~1.31M params

Total (4 blocks): ~5.24M params
```

This is **non-trivial**—comparable to a small MLP with hidden dim ~2K, but more expressive due to attention.

### 4.4 Compared to Simple MLP Alternative

If we replaced the MiddleTransformer with a simple MLP:

```python
# Simple 2-layer MLP alternative
nn.Sequential(
    nn.Linear(256, 512),
    nn.GELU(),
    nn.Linear(512, 256)
)
# Parameters: 256*512 + 512*256 ≈ 262K (5% of MiddleTransformer)
# Capability: Point-wise transformation, no inter-latent reasoning
```

**Conclusion:** The MiddleTransformer is **~20× more capable** than a simple MLP in terms of parameters and architecture complexity.

---

## 5. Training Procedure

### 5.1 Overall Training Pipeline

The system uses a **two-stage scientific pipeline** (from [src/config.py](src/config.py#L165)):

```python
pipeline_mode: str = "two_stage_scientific"
```

**Stages:**

1. **Preprocessing**: Cache encoder outputs and latent vectors
2. **Pretraining** (Scientific datasets):
   - Middle model pretraining (WikiText, arXiv, English Pretrain)
   - Adapter pretraining (learn to decode latent → text)
3. **Finetuning** (SAMSum dataset):
   - Middle model finetuning (dialogue → summary ideas)
   - Adapter finetuning (summary ideas → summary text)

### 5.2 Loss Function for Middle Model

From [src/train.py](src/train.py#L171):

```python
def compute_latent_loss(z_pred: torch.Tensor, z_target: torch.Tensor, config: Config) -> torch.Tensor:
    mse = F.mse_loss(z_pred, z_target)           # L2 reconstruction loss
    contrastive = compute_contrastive_loss(z_pred, z_target)  # Similarity loss
    return (config.latent_mse_weight * mse) + (config.contrastive_weight * contrastive)
```

**Components:**

1. **MSE Loss** (L2 reconstruction): 
   - Encourages latent vectors to be close to target (identity during pretraining)
   - Weight: `latent_mse_weight = 1.0`

2. **Contrastive Loss** (metric learning):
   - Encourages similar samples to have similar transformed latents
   - Encourages different samples to have different transformed latents
   - Weight: `contrastive_weight = 0.1`

### 5.3 Pretraining vs. Finetuning

**Pretraining (Scientific data):**
- Task: Learn to preserve latent semantics under identity mapping
- Input: `z_in` from scientific texts
- Target: `z_target = z_in` (identity supervision)
- Duration: 3 epochs

**Finetuning (SAMSum dialogue → summary):**
- Task: Learn to transform input ideas to summary ideas
- Input: `source_idea` (latent from dialogue)
- Target: `target_idea` (latent from summary)
- Duration: 2 epochs

### 5.4 Training Configuration

From [src/config.py](src/config.py#L50-66):

```python
# Stage-specific epochs
pretrain_middle_epochs: int = 3
pretrain_adapter_epochs: int = 3
finetune_middle_epochs: int = 2
finetune_adapter_epochs: int = 2

# Learning
learning_rate: float = 2e-5
batch_size: int = 10
gradient_accumulation_steps: int = 6
warmup_steps: int = 200

# Mixed precision
use_fp16: bool = True
use_bf16: bool = True
use_gradient_checkpointing: bool = True
```

---

## 6. Integration with Encoder and Adapter

### 6.1 Data Flow During Training

```
Input text (dialogue)
    ↓
LatentEncoder.forward(input_ids, attention_mask)
    ├─ ModernBERT (frozen)
    ├─ Mean pooling over tokens → [B, 768]
    └─ sequence_proj (trainable) → [B, latent_seq_len=8, latent_dim=256]
    ↓
z_in = [B, 8, 256]
    ↓
MiddleTransformer.forward(z_in)  ← 4 transformer blocks
    ↓
z_out = [B, 8, 256]
    ↓
PrefixAdapter.forward(z_out)
    ├─ Cross-attention: queries [B, prefix_len=50, 640] attend to z_out [B, 8, 256]
    └─ Output: [B, 50, 640] (prefix embeddings for GPT-2)
    ↓
Concatenate with target embeddings
    ↓
GPT-2 decoder (frozen) → logits
    ↓
CrossEntropy loss against target tokens
```

### 6.2 Freezing/Unfreezing During Stages

**Pretraining Middle (Phase 1):**
- ✓ MiddleTransformer: **trainable**
- ✗ LatentEncoder: **frozen**
- ✗ PrefixAdapter: **frozen**
- ✗ GPT-2: **frozen**

From [src/train.py](src/train.py#L1039-1041):

```python
for param in model.middle_model.parameters():
    param.requires_grad = True
print("  - Frozen: middle_model")  # Misleading comment—it should say unfrozen
```

**Finetuning Middle (Phase 3):**
- ✓ MiddleTransformer: **trainable**
- ✓ LatentEncoder (top 2 layers): **trainable** (via `num_encoder_unfrozen_layers=2`)
- ✗ PrefixAdapter: **frozen** (to avoid conflicting gradients)
- ✗ GPT-2: **frozen**

---

## 7. Inference and Inference-Time Behavior

### 7.1 Generate Function

From [src/models.py](src/models.py#L758-828):

```python
def generate(self, input_ids, attention_mask, max_length=128, ...):
    # 1. Encode input to latent
    z_out = self.encode_to_latent(input_ids, attention_mask)
    
    # 2. Check for STOP_LATENT
    if use_stop_latent:
        stop_mask = self.is_stop_latent(z_out, ...)
        # If latent matches STOP_LATENT, stop early
    
    # 3. Get prefix embeddings via adapter
    prefix_embeddings = self.get_prefix_embeddings(z_out)  # [B, 50, 640]
    
    # 4. Autoregressive generation
    for step in range(max_length):
        logits = GPT2(inputs_embeds=prefix_embeddings + generated_so_far)
        next_token = sample(logits[:, -1, :])
        if next_token == eos_token_id:
            break
        prefix_embeddings = cat([prefix_embeddings, next_embedding])
    
    return generated_ids
```

### 7.2 STOP_LATENT Mechanism

From [src/models.py](src/models.py#L550-589):

A fixed, learned latent vector that the middle model can produce to signal "stop generation":

```python
STOP_LATENT: [1, latent_seq_len=8, latent_dim=256]
    initialized as random_normalized vectors

# During inference:
is_eos = (cosine_similarity(z_out, STOP_LATENT) >= 0.99)
# If true, skip decoder and generate EOS token
```

**Rationale:** Allows the middle model to learn to "think" about when to stop before generating text.

---

## 8. Architecture Limitations and Characteristics

### 8.1 Strengths

| Feature | Benefit |
|---------|---------|
| **Compact latent sequence (8 vectors)** | Efficient computation, forces semantic compression |
| **Transformer with attention** | Can learn complex latent-space reasoning patterns |
| **RoPE positional encodings** | Enables relative position reasoning in latent space |
| **Pre-norm residual blocks** | Stable training, better gradient flow |
| **SwiGLU activations** | More expressive than GELU/ReLU |
| **Multi-head attention (4 heads)** | Can learn multiple types of relationships |

### 8.2 Limitations and Potential Weaknesses

| Limitation | Impact |
|------------|--------|
| **Fixed-length latent (8 vectors)** | May struggle with variable-complexity inputs (short text vs. long) |
| **Only 4 transformer layers** | Limited iterative reasoning depth |
| **Single attention pass per layer** | No hierarchical, multi-scale representation |
| **Small head dimension (64)** | Each head operates in relatively low-dimensional space |
| **Contrastive loss weight (0.1)** | MSE loss dominates; contrastive signal is weak |
| **No task-specific adaptation** | Same middle model used for all tasks (identity pretraining) |
| **Identity supervision in pretraining** | Doesn't force task-specific transformation learning until finetuning |

### 8.3 Where the Middle Model Might Be Underpowered

1. **Complex reasoning tasks**: 4 transformer layers may not be enough for multi-step reasoning
   - Typical language models use 12–96 layers
   - But here we're operating in semantic, not token, space—so this may be less critical

2. **Variable-length outputs**: Fixed 8-slot latent sequence assumes all tasks have similar "idea complexity"
   - A 5-word dialogue and a 500-word article both compress to 8 vectors
   - Information bottleneck may be too tight

3. **Task specialization**: No explicit mechanism to specialize the middle model per task
   - SAMSum finetuning and arXiv summarization both use the same model

### 8.4 Where the Middle Model Might Be Overpowered

1. **Simple tasks** (e.g., classification, short answers)
   - 5M parameters is substantial for MNIST-scale tasks
   - May overfit on small domains

2. **Redundancy with PrefixAdapter**
   - Both middle and adapter learn transformations
   - Possible duplication of effort

---

## 9. Comparison to Alternatives

### 9.1 Simple Bottleneck (MLP)

```python
# Alternative: Simple MLP
class SimpleMLP(nn.Module):
    def forward(self, z_in):  # [B, 8, 256]
        z_in_flat = z_in.flatten(1)  # [B, 8*256=2048]
        z_out_flat = self.layers(z_in_flat)  # [B, 2048]
        return z_out_flat.view(...)  # [B, 8, 256]
```

**Differences:**
- ✓ Simpler, fewer parameters
- ✗ No inter-vector attention (each position processed independently)
- ✗ No positional information encoding
- **Verdict:** Middle Model is more powerful

### 9.2 Larger Transformer

```python
# Alternative: Larger middle transformer
class LargerMiddle(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(12)])
```

**Differences:**
- ✓ More reasoning capacity
- ✗ Slower, more memory
- ✗ Harder to train end-to-end with frozen encoder/decoder
- **Verdict:** 4 layers is reasonable trade-off

### 9.3 Autoregressive Latent Transformer

```python
# Alternative: Generate latent tokens autoregressively
class AutoregressiveLatent(nn.Module):
    def forward(self, z_in):  # [B, 8, 256]
        z_out = []
        for i in range(8):
            z_i = self.generate_latent_token(z_in, z_out)  # Attend to previous z_out tokens
            z_out.append(z_i)
        return stack(z_out)  # [B, 8, 256]
```

**Differences:**
- ✓ Could model latent dependencies more explicitly
- ✗ More complex training, slower inference
- ✗ Higher variance in outputs
- **Verdict:** Current parallel approach is simpler, more stable

---

## 10. Summary: What the Middle Model Really Does

### The Core Answer

The **MiddleTransformer** is a **compact, multi-head transformer** that:

1. **Accepts** a fixed-length latent sequence: `[B, 8, 256]`
2. **Applies** 4 transformer blocks, each with:
   - Layer-normalized self-attention (4 heads, RoPE)
   - SwiGLU FFN with 4× expansion
   - Residual connections
3. **Outputs** a transformed latent sequence: `[B, 8, 256]`
4. **Learns** to map task-specific input semantics to target semantics
   - In pretraining: identity preservation on scientific data
   - In finetuning: dialogue ideas → summary ideas

### Conceptual Role

| Perspective | Characterization |
|-------------|-----------------|
| **Computational** | 4-layer transformer with self-attention over 8-position latent sequence |
| **Functional** | Learned transformation from input to output semantic space |
| **Information-theoretic** | Semantic reasoner with ~5M parameters and attention-based gating |
| **Training** | Supervised via MSE and contrastive losses on latent pairs |
| **Inference** | Stateless forward pass; can optionally output STOP_LATENT to halt decoding |

### Is It Just a Bottleneck?

**No.** It is a **genuine latent-space reasoning core** with:
- Nontrivial architecture (transformer, not linear projection)
- Multi-step iterative refinement (4 layers, each layer can attend to all positions)
- Learnable parameters (~5M) enabling complex transformations
- Attention mechanism providing interpretability into latent-space reasoning

**However**, whether this reasoning is actually necessary depends on the task complexity and the information bottleneck. For simple summarization, a simpler middle model might suffice, but the current design provides a foundation for more complex reasoning if needed in future iterations.

---

## Appendix: File References

- [src/models.py](src/models.py): LatentEncoder, MiddleTransformer, PrefixAdapter, LatentSpaceModel
- [src/config.py](src/config.py): Hyperparameter definitions
- [src/train.py](src/train.py): Training loops, loss computation, phase-specific logic
- [README.md](README.md): High-level architecture overview
- [stack.md](stack.md): Detailed stack and training strategy

