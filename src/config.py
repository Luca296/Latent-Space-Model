"""
Configuration for the latent-space reasoning model.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters for the latent-space reasoning model."""
    
    # Model dimensions
    latent_dim: int = 256
    prefix_len: int = 10  # Reduced from 50 for more stable training
    modernbert_hidden_dim: int = 768
    gpt2_hidden_dim: int = 640  # Gemma-3-270m hidden dimension
    middle_hidden_dim: int = 512
    middle_layers: int = 3
    num_encoder_unfrozen_layers: int = 2  # Number of top ModernBERT layers to unfreeze in Phase 2
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    max_seq_len: int = 512
    max_target_len: int = 256
    num_workers: int = 4  # Number of workers for data loading
    
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
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500  # Save every N steps
    
    # Inference settings
    temperature: float = 0.7
    max_generation_length: int = 128
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50

    # UI settings
    use_tui: bool = True
    
    # Dataset
    dataset_name: str = "knkarthick/samsum"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    max_train_samples: int = 10000  # Use subset for faster prototyping
    
    # Diagnostic test modes
    # "normal" = Standard training (Dialog -> Summary, all trainable)
    # "phase1" = Align Decoder (Simple Identity, train decoder only)
    # "phase2" = Train Encoder (SAMSum Identity, train encoder+decoder, bypass middle)
    # "phase3" = Train Middle (SAMSum, train middle only, freeze others)
    training_phase: str = "normal"
    
    # Deprecated (kept for backward compatibility)
    test_mode: str = None