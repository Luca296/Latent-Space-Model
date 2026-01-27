"""
Training loop for the latent-space reasoning model.

Handles training, validation, checkpointing, and mixed precision training.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path

from src.models import LatentSpaceModel
from src.data import create_dataloaders
from src.config import Config


def compute_loss(logits: torch.Tensor, target_ids: torch.Tensor, 
                 target_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for decoder outputs.
    
    Args:
        logits: [B, T-1, vocab_size] decoder logits
        target_ids: [B, T] target token IDs
        target_attention_mask: [B, T] target attention mask
        
    Returns:
        loss: Scalar loss value
    """
    # Shift target IDs for teacher forcing (align with logits)
    target_ids_shifted = target_ids[:, 1:]  # [B, T-1]
    target_mask_shifted = target_attention_mask[:, 1:]  # [B, T-1]
    
    # Flatten for cross-entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    target_ids_flat = target_ids_shifted.reshape(-1)
    target_mask_flat = target_mask_shifted.reshape(-1)
    
    # Compute loss (ignore padding tokens)
    loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    loss_per_token = loss_fct(logits_flat, target_ids_flat)
    
    # Apply mask to ignore padding
    loss_per_token = loss_per_token * target_mask_flat.float()
    
    # Average over non-padding tokens
    loss = loss_per_token.sum() / target_mask_flat.sum()
    
    return loss


def train_epoch(
    model: LatentSpaceModel,
    dataloader,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: Config,
    epoch: int
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: The latent-space model
        dataloader: Training dataloader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: Device to train on
        config: Configuration
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config.use_fp16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_ids=target_ids,
                target_attention_mask=target_attention_mask
            )
            
            loss = compute_loss(logits, target_ids, target_attention_mask)
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
        
        # Update weights after gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": total_loss / num_batches})
            
            # Save checkpoint
            global_step = epoch * len(dataloader) + step + 1
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, scaler, global_step, epoch, config)
    
    # Handle remaining gradients
    if num_batches % config.gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: LatentSpaceModel,
    dataloader,
    device: torch.device,
    config: Config
) -> float:
    """
    Validate the model.
    
    Args:
        model: The latent-space model
        dataloader: Validation dataloader
        device: Device to validate on
        config: Configuration
        
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)
            
            # Forward pass
            with autocast(enabled=config.use_fp16):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_ids=target_ids,
                    target_attention_mask=target_attention_mask
                )
                
                loss = compute_loss(logits, target_ids, target_attention_mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(
    model: LatentSpaceModel,
    optimizer,
    scaler: GradScaler,
    step: int,
    epoch: int,
    config: Config
):
    """Save model checkpoint."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    
    torch.save({
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(model: LatentSpaceModel, checkpoint_path: str, device: torch.device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint["step"], checkpoint["epoch"]


def train(config: Config):
    """
    Main training function.
    
    Args:
        config: Configuration object
    """
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config,
        train_samples=config.max_train_samples,
        val_samples=1000  # Use 1000 samples for validation
    )
    
    # Initialize model
    print("Initializing model...")
    model = LatentSpaceModel(config).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer (only for trainable parameters)
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.use_fp16)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch
        )
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device, config)
        print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    config = Config()
    train(config)