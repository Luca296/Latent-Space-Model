"""
Configuration for the latent-space reasoning model.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters for the latent-space reasoning model."""
    
    # Model dimensions
    latent_dim: int = 256
    prefix_len: int = 10
    modernbert_hidden_dim: int = 768
    gpt2_hidden_dim: int = 768
    middle_hidden_dim: int = 512
    middle_layers: int = 3
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 256
    max_target_len: int = 128
    
    # Model names
    modernbert_model: str = "answerdotai/ModernBERT-base"
    gpt2_model: str = "gpt2"
    
    # Training settings
    num_epochs: int = 5
    warmup_steps: int = 100
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
    
    # Dataset
    dataset_name: str = "samsum"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    max_train_samples: int = 10000  # Use subset for faster prototyping