# Configuration Reference

This file documents all configuration parameters from src/config.py.

## Overview

The Config class uses Python dataclasses for type-safe configuration.

```python
from dataclasses import dataclass

@dataclass
class Config:
    # All parameters with defaults
    ...
```

## Model Dimensions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| latent_dim | int | 256 | Dimension of each latent vector |
| latent_seq_len | int | 8 | Number of latent vectors in sequence |
| prefix_len | int | 50 | Length of decoder prefix |
| modernbert_hidden_dim | int | 768 | ModernBERT hidden size |
| gpt2_hidden_dim | int | 640 | Gemma hidden dimension |
| middle_hidden_dim | int | 512 | Middle model FFN dimension |
| middle_layers | int | 4 | Number of middle MLP layers (legacy) |
| middle_transformer_layers | int | 4 | Number of transformer layers |
| middle_num_heads | int | 4 | Attention heads in middle model |
| middle_ffn_multiplier | float | 4.0 | FFN hidden = dim × multiplier |
| middle_dropout | float | 0.1 | Dropout rate |
| middle_use_rope | bool | True | Use RoPE in middle model |
| adapter_num_heads | int | 4 | Attention heads in prefix adapter |
| adapter_dropout | float | 0.1 | Dropout in adapter |
| adapter_use_rope | bool | True | Use RoPE in adapter |
| num_encoder_unfrozen_layers | int | 2 | Top ModernBERT layers to unfreeze |

## Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| learning_rate | float | 2e-5 | Initial learning rate |
| batch_size | int | 30 | Training batch size |
| gradient_accumulation_steps | int | 3 | Steps between optimizer updates |
| max_seq_len | int | 512 | Maximum input sequence length |
| max_target_len | int | 256 | Maximum target sequence length |
| num_workers | int | 4 | DataLoader worker processes |
| prefetch_factor | int | 4 | Batches prefetched per worker |
| num_epochs | int | 50 | Total training epochs |
| warmup_steps | int | 200 | LR warmup steps |
| weight_decay | float | 0.01 | AdamW weight decay |

## Model Names

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| modernbert_model | str | answerdotai/ModernBERT-base | Encoder model |
| gpt2_model | str | google/gemma-3-270m-it | Decoder model |

## Device and Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| device | str | cuda | Device for training |
| use_fp16 | bool | True | Enable FP16 mixed precision |
| use_bf16 | bool | True | Enable BF16 mixed precision |
| use_gradient_checkpointing | bool | True | Reduce memory via checkpointing |
| use_torch_compile | bool | False | Use torch.compile |
| torch_compile_mode | str | reduce-overhead | Compilation mode |
| async_checkpointing | bool | True | Save checkpoints async |

## Attention and Normalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| attn_implementation | str | flash_attention_2 | Attention backend |
| use_rmsnorm | bool | True | Use RMSNorm |
| rmsnorm_eps | float | 1e-6 | RMSNorm epsilon |
| use_prefix_rope | bool | True | RoPE in prefix adapter |
| rope_base | int | 10000 | RoPE base frequency |

## Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| checkpoint_dir | str | checkpoints | Directory for checkpoints |
| save_every | int | 500 | Steps between saves |
| pretrain_best_model_filename | str | best_model_pretrain.pt | Stage 1-2 output |
| finetune_best_model_filename | str | best_model_finetune.pt | Stage 3-4 output |
| handoff_best_model_filename | str | best_model.pt | Final checkpoint |

## Inference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| temperature | float | 0.7 | Sampling temperature |
| max_generation_length | int | 256 | Max output tokens |
| do_sample | bool | True | Use sampling vs greedy |
| top_p | float | 0.9 | Nucleus sampling |
| top_k | int | 50 | Top-k sampling |

## UI

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| use_tui | bool | True | Use Rich terminal UI |

## Latent Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| normalize_latent | bool | True | L2 normalize before decoding |
| use_stop_latent | bool | True | Enable stop latent learning |
| stop_latent_init | str | random_normalized | Initialization method |
| stop_latent_seed | int | 1337 | Random seed |
| stop_latent_cosine_threshold | float | 0.99 | Early stopping threshold |

## Datasets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_name | str | knkarthick/samsum | Main dataset |
| train_split | str | train | Train split name |
| validation_split | str | validation | Validation split name |
| test_split | str | test | Test split name |
| max_train_samples | int | None | Limit train samples |
| wikitext_dataset | str | Hieuman/wikitext-103-filtered | Stage 1 data |
| arxiv_dataset | str | macrocosm/arxiv_abstracts | Stage 1 data |
| english_pretrain_dataset | str | shuyuej/English-Pretraining-Dataset | Stage 1 data |

## Caching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| preprocess_cache_dir | str | cache/preprocessed | Cache location |
| preprocess_format | str | jsonl | Cache format |
| skip_preprocessing_if_cached | bool | True | Reuse existing cache |
| preprocess_batch_size | int | 32 | Batch size for preprocessing |
| preprocess_validation_fraction | float | 0.05 | Val split for pretraining |

### Cache Field Controls

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| cache_store_text | bool | False | Store raw text |
| cache_store_input_ids | bool | False | Store token IDs |
| cache_store_attention_mask | bool | False | Store attention masks |
| cache_store_target_ids | bool | True | Store target IDs |
| cache_store_target_attention_mask | bool | True | Store target masks |
| cache_store_embeddings_int8 | bool | True | Quantize embeddings |
| cache_use_embeddings_for_training | bool | True | Use cached embeddings |
| cache_write_offsets_index | bool | True | Write offset file |

## Pipeline Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| pipeline_mode | str | two_stage_scientific | Training pipeline type |
| run_preprocessing | bool | True | Execute preprocessing |
| run_pretraining | bool | True | Run Stage 1-2 |
| run_finetuning | bool | True | Run Stage 3-4 |
| freeze_encoder_compression_in_pipeline | bool | True | Freeze encoder in pipeline |
| adapter_pretrain_use_middle | bool | False | Use middle in Stage 2 |

## Stage-Specific Epochs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| pretrain_middle_epochs | int | 6 | Stage 1 epochs |
| pretrain_adapter_epochs | int | 4 | Stage 2 epochs |
| finetune_middle_epochs | int | 2 | Stage 3 epochs |
| finetune_adapter_epochs | int | 2 | Stage 4 epochs |

## Stage-Specific Batch Sizes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| pretrain_middle_batch_size | int | None | Stage 1 batch size |
| pretrain_adapter_batch_size | int | None | Stage 2 batch size |
| finetune_middle_batch_size | int | None | Stage 3 batch size |
| finetune_adapter_batch_size | int | None | Stage 4 batch size |

Note: None means use global batch_size

## Stage-Specific Max Samples

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| pretrain_middle_max_samples | int | None | Stage 1 total cap |
| pretrain_adapter_max_samples | int | None | Stage 2 total cap |
| finetune_middle_max_samples | int | None | Stage 3 total cap |
| finetune_adapter_max_samples | int | None | Stage 4 total cap |

## Per-Dataset Caps (Pretraining)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| pretrain_middle_max_samples_wikitext | int | None | WikiText cap |
| pretrain_middle_max_samples_arxiv | int | None | arXiv cap |
| pretrain_middle_max_samples_english | int | None | English pretrain cap |
| pretrain_adapter_max_samples_wikitext | int | 150000 | Stage 2 WikiText cap |
| pretrain_adapter_max_samples_arxiv | int | 150000 | Stage 2 arXiv cap |

## Loss Weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| latent_mse_weight | float | 1.0 | MSE component weight |
| contrastive_weight | float | 0.1 | Contrastive loss weight |

## Diagnostic Modes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| training_phase | str | normal | Training mode |

Options:
- normal: Standard training
- phase1: Decoder alignment only
- phase2: Encoder training
- phase3: Middle model only

## Usage Examples

### Creating Config

```python
from src.config import Config

# Default configuration
config = Config()

# Modify specific values
config.learning_rate = 1e-4
config.batch_size = 16

# Save and load
import json
with open('my_config.json', 'w') as f:
    json.dump(config.__dict__, f)
```

### CLI Override

```bash
python main.py --mode train \
    --learning-rate 5e-5 \
    --batch-size 20 \
    --pretrain-middle-epochs 8
```

## Configuration Tips

1. Reduce batch_size if out of memory
2. Increase gradient_accumulation_steps to maintain effective batch size
3. Adjust warmup_steps based on dataset size
4. Enable use_bf16 if supported (faster than fp16)
5. Use async_checkpointing for better throughput
6. Set max_train_samples for faster experimentation
