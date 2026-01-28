# Hotfix Implementation Guide

## [X] 2. Normalize the Latent Vector Before Expansion
This helps significantly with stability. GPT-2 expects embeddings with a specific norm distribution. Apply the following normalization:

**Code:**
$$z = \frac{z}{\|z\| + 1e-8}$$

---

## [X] 3. Add a LayerNorm Before GPT-2
GPT-2’s internal embeddings are normalized; your prefix should be as well to ensure feature compatibility.

---

## [X] 4. Train the Decoder Adapter Longer
The initial 5 epochs are insufficient for convergence. Adjust your training hyper-parameters:

* **Duration:** 20–30 epochs
* **Batch Size:** Use small batches
* **Learning Rate:** Low ($1 \times 10^{-5}$ to $5 \times 10^{-5}$)

---

## [] 5. Sequential Training Strategy (The "Big Fix")
Training the encoder compression, middle model, and decoder expansion simultaneously creates too much "co-adaptation" chaos. Use a phased approach:

### [X] Phase 1: Align Decoder
**Goal:** Stabilize the latent-to-text mapping.
* **Components:** Expansion MLP + Optional GPT-2 LoRA.
* **Task:** Identity mapping.
* **Outcome:** GPT-2 learns to decode latent vectors derived from real text.

### [X] Phase 2: Train Encoder Compression
**Goal:** Map input to the stabilized latent space.
* **Components:** Compression head + Expansion head + Optional LoRA.

### [] Phase 3: Train Middle Model
**Goal:** Perform transformations.
* **Constraint:** Only begin this phase once the latent space is stable and the decoder is reliable.