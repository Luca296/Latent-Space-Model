"""
Configuration for the latent-space reasoning model.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters for the latent-space reasoning model."""
    
    # Model dimensions
    latent_dim: int = 256
    latent_seq_len: int = 8  # Fixed-length latent sequence length
    prefix_len: int = 50  # Restored to larger default to provide longer prefix context
    modernbert_hidden_dim: int = 768
    gpt2_hidden_dim: int = 640  # Gemma-3-270m hidden dimension
    middle_hidden_dim: int = 512
    middle_layers: int = 4
    middle_transformer_layers: int = 4
    middle_num_heads: int = 4
    middle_ffn_multiplier: float = 4.0
    middle_dropout: float = 0.1
    middle_use_rope: bool = True
    adapter_num_heads: int = 4
    adapter_dropout: float = 0.1
    adapter_use_rope: bool = True
    num_encoder_unfrozen_layers: int = 2  # Number of top ModernBERT layers to unfreeze in Phase 2
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 30
    gradient_accumulation_steps: int = 3
    max_seq_len: int = 512
    max_target_len: int = 256
    num_workers: int = 4  # Number of workers for data loading
    prefetch_factor: int = 4  # Prefetch batches per worker (only when num_workers > 0)
    
    # Model names
    modernbert_model: str = "answerdotai/ModernBERT-base"  # Base is the standard/small option
    gpt2_model: str = "google/gemma-3-270m-it"  # Gemma 3 270M (efficient decoder)
    
    # Training settings
    num_epochs: int = 50  # Increased from 30 for better convergence
    warmup_steps: int = 200  # Increased from 100 for better learning rate schedule
    weight_decay: float = 0.01
    
    # Device settings
    device: str = "cuda"  # Will be set automatically based on availability
    use_fp16: bool = True
    use_bf16: bool = True  # Prefer BF16 when supported for speed/stability
    use_gradient_checkpointing: bool = True  # Reduce memory usage during backward pass
    use_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    async_checkpointing: bool = True

    # Attention + normalization/activation
    attn_implementation: str | None = "flash_attention_2"  # Fallback to SDPA/eager if unavailable
    use_rmsnorm: bool = True
    rmsnorm_eps: float = 1e-6
    use_prefix_rope: bool = True
    rope_base: int = 10000
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500  # Save every N steps
    
    # Inference settings
    temperature: float = 0.7
    max_generation_length: int = 256
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50

    # UI settings
    use_tui: bool = True

    # Latent handling
    normalize_latent: bool = True  # If True, L2-normalize latent vectors before decoding

    # Latent STOP vector handling
    use_stop_latent: bool = True
    stop_latent_init: str = "random_normalized"  # "zero" or "random_normalized"
    stop_latent_seed: int = 1337
    stop_latent_cosine_threshold: float = 0.99
    stop_latent_l2_threshold: float = None
    
    # Dataset
    dataset_name: str = "knkarthick/samsum"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    max_train_samples: int = None  # Use subset for faster prototyping

    # Scientific pretraining datasets
    wikitext_dataset: str = "Hieuman/wikitext-103-filtered"
    arxiv_dataset: str = "macrocosm/arxiv_abstracts"
    english_pretrain_dataset: str = "shuyuej/English-Pretraining-Dataset"
    wikitext_split: str = "train"
    arxiv_split: str = "train"
    english_pretrain_split: str = "train"
    wikitext_max_samples: int = 750000
    arxiv_max_samples: int = 750000
    english_pretrain_max_samples: int = None

    # Preprocessing / caching
    preprocess_cache_dir: str = "cache/preprocessed"
    preprocess_format: str = "jsonl"
    skip_preprocessing_if_cached: bool = True
    preprocess_batch_size: int = 32
    preprocess_wikitext: bool = True
    preprocess_arxiv: bool = True
    preprocess_english_pretrain: bool = True
    preprocess_validation_fraction: float = 0.05

    # Cache field controls (minimal by default)
    cache_store_text: bool = False
    cache_store_input_ids: bool = False
    cache_store_attention_mask: bool = False
    cache_store_target_ids: bool = True
    cache_store_target_attention_mask: bool = True
    cache_store_embeddings_fp16: bool = False
    cache_store_embeddings_int8: bool = True
    cache_store_ideas_fp16: bool = False
    cache_store_ideas_int8: bool = False
    cache_use_embeddings_for_training: bool = True
    cache_write_offsets_index: bool = True

    # Pipeline control
    pipeline_mode: str = "two_stage_scientific"  # "two_stage_scientific" or "legacy"
    run_preprocessing: bool = True
    run_pretraining: bool = True
    run_finetuning: bool = True
    freeze_encoder_compression_in_pipeline: bool = True
    adapter_pretrain_use_middle: bool = False

    # Stage-specific epochs
    pretrain_middle_epochs: int = 6
    pretrain_adapter_epochs: int = 24
    finetune_middle_epochs: int = 2
    finetune_adapter_epochs: int = 2

    # Stage-specific batch sizes (None = use global batch_size)
    pretrain_middle_batch_size: int | None = None
    pretrain_adapter_batch_size: int | None = None
    finetune_middle_batch_size: int | None = None
    finetune_adapter_batch_size: int | None = None

    # Stage-specific max samples (None = use all available)
    pretrain_middle_max_samples: int | None = None
    pretrain_adapter_max_samples: int | None = None
    finetune_middle_max_samples: int | None = None
    finetune_adapter_max_samples: int | None = None

    # Per-dataset caps (override stage-level caps when set)
    pretrain_middle_max_samples_wikitext: int | None = None
    pretrain_middle_max_samples_arxiv: int | None = None
    pretrain_middle_max_samples_english: int | None = None
    pretrain_middle_max_samples_scitldr: int | None = None

    pretrain_adapter_max_samples_wikitext: int | None = 150000
    pretrain_adapter_max_samples_arxiv: int | None = 150000
    pretrain_adapter_max_samples_english: int | None = None
    pretrain_adapter_max_samples_scitldr: int | None = None

    # Combined objective weights (middle model)
    latent_mse_weight: float = 1.0
    contrastive_weight: float = 0.1

    # Stage-specific checkpoints
    pretrain_best_model_filename: str = "best_model_pretrain.pt"
    finetune_best_model_filename: str = "best_model_finetune.pt"
    handoff_best_model_filename: str = "best_model.pt"
    
    # Diagnostic test modes
    # "normal" = Standard training (Dialog -> Summary, all trainable)
    # "phase1" = Align Decoder (Simple Identity, train decoder only)
    # "phase2" = Train Encoder (SAMSum Identity, train encoder+decoder, bypass middle)
    # "phase3" = Train Middle (SAMSum, train middle only, freeze others)
    training_phase: str = "normal"
    
    # Deprecated (kept for backward compatibility)
    test_mode: str = None