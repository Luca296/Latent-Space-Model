# Latent-Space Reasoning Model

A PyTorch prototype for a model that "thinks" in a compact latent "idea space" instead of directly in token space.

## Architecture

The model consists of three main components:

1. **Encoder (ModernBERT)**: Converts text tokens to a compact "idea vector" (latent_dim=256)
2. **Middle Model (MLP)**: Transforms idea vectors through latent reasoning (2-4 layer MLP)
3. **Decoder (GPT-2)**: Converts transformed idea back to text using a prefix adapter

### Data Flow

```
Input Text -> ModernBERT -> Pooling -> Compression MLP -> Idea Vector (256-dim)
                                                          |
                                                    Middle Model (MLP)
                                                          |
                                                    Transformed Idea
                                                          |
                                            Prefix Adapter (Expansion MLP)
                                                          |
                                              GPT-2 Prefix Embeddings
                                                          |
                                                    Generated Text
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on the SAMSum dataset:

```bash
python main.py --mode train
```

With custom hyperparameters:

```bash
python main.py --mode train \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --epochs 5 \
    --max-train-samples 10000
```

### Interactive Inference

Run interactive inference session:

```bash
python main.py --mode inference
```

Or specify a checkpoint:

```bash
python main.py --mode inference --checkpoint checkpoints/best_model.pt
```

### Batch Generation

Generate text from input:

```bash
python main.py --mode generate \
    --input "Your input text here" \
    --checkpoint checkpoints/best_model.pt
```

With generation parameters:

```bash
python main.py --mode generate \
    --input "Your input text here" \
    --checkpoint checkpoints/best_model.pt \
    --max-length 128 \
    --temperature 0.7 \
    --top-p 0.9
```

## Project Structure

```
Latent-Space-Model/
├── main.py              # Entry point with CLI
├── requirements.txt     # Dependencies
├── README.md           # This file
├── stack.md            # Architecture specification
├── src/
│   ├── __init__.py
│   ├── config.py       # Configuration and hyperparameters
│   ├── models.py       # Model architecture
│   ├── data.py         # Data loading (SAMSum)
│   ├── train.py        # Training loop
│   └── inference.py    # Inference and generation
└── checkpoints/        # Saved model checkpoints (created during training)
```

## Configuration

Key hyperparameters (defined in src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| latent_dim | 256 | Dimension of idea vector |
| prefix_len | 10 | Length of GPT-2 prefix |
| modernbert_hidden_dim | 768 | ModernBERT hidden size |
| gpt2_hidden_dim | 768 | GPT-2 hidden size |
| middle_hidden_dim | 512 | Middle model hidden size |
| middle_layers | 3 | Number of middle MLP layers |
| learning_rate | 1e-4 | Learning rate |
| batch_size | 4 | Batch size |
| max_seq_len | 256 | Maximum input sequence length |
| max_target_len | 128 | Maximum target sequence length |

## Model Components

### LatentEncoder

- Uses ModernBERT (answerdotai/ModernBERT-base) as backbone
- Mean pooling over non-padded tokens
- Compression MLP: 768 -> 512 -> 256
- ModernBERT backbone is frozen during training

### MiddleModel

- 3-layer MLP for latent reasoning
- Architecture: 256 -> 512 -> 512 -> 256
- Fully trainable

### PrefixAdapter

- Expands latent vector to GPT-2 prefix embeddings
- Architecture: 256 -> (10 x 768)
- Fully trainable

### LatentSpaceModel

- Combines all components
- GPT-2 (gpt2) backbone is frozen during training
- Only trainable: encoder compression, middle model, prefix adapter

## Training Strategy

- **Frozen backbones**: ModernBERT and GPT-2 are frozen to save memory
- **Trainable components**: Encoder compression MLP, middle model, prefix adapter
- **Loss**: Cross-entropy on decoder outputs
- **Optimizer**: AdamW with weight decay
- **Mixed precision**: FP16 for memory efficiency
- **Gradient accumulation**: Supported for effective larger batch sizes

## Dataset

Uses the SAMSum dataset for dialogue summarization:
- ~14,700 dialogue-summary pairs
- Loaded from Hugging Face Datasets
- Suitable for prototyping on RTX 3060

## Hardware Requirements

- GPU: RTX 3060 (12GB VRAM) or equivalent
- RAM: 16GB+ recommended
- Storage: ~5GB for models and checkpoints

## Features

- ModernBERT encoder with frozen backbone
- MLP-based latent reasoning core
- GPT-2 decoder with prefix adapter
- Mixed precision training (FP16)
- Gradient accumulation support
- Checkpoint saving and loading
- Interactive inference mode
- Batch text generation
- Latent vector interpolation
- Configurable hyperparameters

## Advanced Features

### Latent Interpolation

Explore the latent space by interpolating between two inputs:

```python
from src.inference import LatentSpaceInference
from src.config import Config

config = Config()
inference = LatentSpaceInference("checkpoints/best_model.pt", config)

# Interpolate between two texts
result = inference.interpolate_and_generate(
    input_text_1="First dialogue...",
    input_text_2="Second dialogue...",
    alpha=0.5  # 0.0 = first, 1.0 = second
)
print(result)
```

### Get Latent Vectors

Extract latent representations for analysis:

```python
latent = inference.get_latent_vector("Your text here")
print(f"Latent shape: {latent.shape}")  # torch.Size([256])
```

## References

- stack.md - Detailed architecture specification
- ModernBERT: answerdotai/ModernBERT-base
- GPT-2: gpt2
- SAMSum Dataset: samsum from Hugging Face
