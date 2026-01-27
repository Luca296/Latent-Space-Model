"""
Training loop for the latent-space reasoning model.

Handles training, validation, checkpointing, and mixed precision training.
Now with a rich TUI!
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime

# Rich imports
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID
)
from rich.text import Text
from rich import box

from src.models import LatentSpaceModel
from src.data import create_dataloaders
from src.config import Config


class TrainingDashboard:
    """Manages the Rich TUI for training."""
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.start_time = time.time()
        self.metrics_history = []
        self.logs = []
        
        # Initialize layout
        self.layout = self.make_layout()
        
        # Progress bars
        self.progress_group = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
        self.epoch_task = self.progress_group.add_task("[yellow]Overall Progress", total=config.num_epochs)
        self.batch_task = self.progress_group.add_task("[cyan]Current Epoch", total=100) # Placeholder total
        
        # Current state
        self.current_epoch = 0
        self.current_step = 0
        self.current_loss = 0.0
        self.avg_loss = 0.0
        self.val_loss = None
        self.best_val_loss = float('inf')

    def make_layout(self) -> Layout:
        """Define the TUI layout."""
        layout = Layout(name="root")
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        return layout

    def generate_header(self) -> Panel:
        """Create the header panel."""
        elapsed = time.time() - self.start_time
        title = Text("Latent-Space Reasoning Model Training", style="bold magenta")
        subtitle = f"Device: {self.config.device} | FP16: {self.config.use_fp16} | Batch: {self.config.batch_size}"
        return Panel(
            Text.from_markup(f"{title}\n[dim]{subtitle}[/dim]"),
            style="white on blue",
            box=box.ROUNDED
        )

    def generate_metrics_table(self) -> Panel:
        """Create the metrics table."""
        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Current Epoch", f"{self.current_epoch + 1}/{self.config.num_epochs}")
        table.add_row("Global Step", str(self.current_step))
        table.add_row("Current Loss", f"{self.current_loss:.4f}")
        table.add_row("Average Loss", f"{self.avg_loss:.4f}")
        
        if self.val_loss is not None:
            val_style = "green" if self.val_loss <= self.best_val_loss else "yellow"
            table.add_row("Validation Loss", f"[{val_style}]{self.val_loss:.4f}[/]")
            table.add_row("Best Val Loss", f"[bold green]{self.best_val_loss:.4f}[/]")
            
        return Panel(
            table,
            title="Training Metrics",
            border_style="yellow",
            box=box.ROUNDED
        )

    def generate_logs_panel(self) -> Panel:
        """Create the logs panel."""
        log_text = "\n".join(self.logs[-8:]) # Show last 8 logs
        return Panel(
            log_text,
            title="Recent Logs",
            border_style="white",
            box=box.ROUNDED
        )
        
    def generate_config_panel(self) -> Panel:
        """Create configuration summary panel."""
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Param", style="dim cyan")
        table.add_column("Value", style="dim white")
        
        table.add_row("Dataset", self.config.dataset_name)
        table.add_row("Base Model", self.config.modernbert_model.split('/')[-1])
        table.add_row("Decoder", self.config.gpt2_model)
        table.add_row("Latent Dim", str(self.config.latent_dim))
        table.add_row("Middle Layers", str(self.config.middle_layers))
        table.add_row("LR", str(self.config.learning_rate))
        
        return Panel(
            table,
            title="Configuration",
            border_style="blue",
            box=box.ROUNDED
        )

    def update_display(self):
        """Update the entire layout."""
        self.layout["header"].update(self.generate_header())
        self.layout["left"].update(
            Panel(
                self.progress_group,
                title="Progress",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2)
            )
        )
        
        # Right side split into Metrics and Config
        right_content = Layout()
        right_content.split_column(
            Layout(self.generate_metrics_table(), ratio=2),
            Layout(self.generate_config_panel(), ratio=1)
        )
        self.layout["right"].update(right_content)
        
        self.layout["footer"].update(self.generate_logs_panel())

    def log(self, message: str):
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        
    def update_metrics(self, loss: float, avg_loss: float, step: int):
        self.current_loss = loss
        self.avg_loss = avg_loss
        self.current_step = step


def compute_loss(logits: torch.Tensor, target_ids: torch.Tensor, 
                 target_attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for decoder outputs."""
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
    epoch: int,
    dashboard: TrainingDashboard
) -> float:
    """Train for one epoch with TUI updates."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    dashboard.current_epoch = epoch
    
    # Reset and update batch progress bar
    dashboard.progress_group.reset(dashboard.batch_task)
    dashboard.progress_group.update(
        dashboard.batch_task, 
        total=len(dataloader),
        description=f"[cyan]Epoch {epoch+1}/{config.num_epochs}"
    )
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
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
        
        current_loss_val = loss.item() * config.gradient_accumulation_steps
        total_loss += current_loss_val
        num_batches += 1
        
        # Update weights after gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update Dashboard
            global_step = epoch * len(dataloader) + step + 1
            avg_loss = total_loss / num_batches
            dashboard.update_metrics(current_loss_val, avg_loss, global_step)
            
            # Save checkpoint
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, scaler, global_step, epoch, config)
                dashboard.log(f"[yellow]Checkpoint saved at step {global_step}[/]")
        
        # Advance progress bar
        dashboard.progress_group.advance(dashboard.batch_task)
    
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
    config: Config,
    dashboard: TrainingDashboard
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    dashboard.log("[blue]Starting validation...[/]")
    
    # Create a temporary progress task for validation
    val_task = dashboard.progress_group.add_task("[magenta]Validating...", total=len(dataloader))
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)
            
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
            dashboard.progress_group.advance(val_task)
            
    # Remove validation task
    dashboard.progress_group.remove_task(val_task)
    
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


def train(config: Config):
    """Main training function."""
    # Initialize Dashboard
    dashboard = TrainingDashboard(config)
    
    with Live(dashboard.layout, refresh_per_second=4, screen=True) as live:
        # Bind the live update to the dashboard
        def update_view():
            dashboard.update_display()
            live.update(dashboard.layout)
            
        dashboard.log("Initializing training setup...")
        update_view()

        # Set device
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        dashboard.log(f"Using device: [bold cyan]{device}[/]")
        update_view()
        
        # Create dataloaders
        dashboard.log("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            config,
            train_samples=config.max_train_samples,
            val_samples=1000
        )
        dashboard.log(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        update_view()
        
        # Initialize model
        dashboard.log("Initializing model (downloading if needed)...")
        update_view() # Force update before potentially long blocking call
        model = LatentSpaceModel(config).to(device)
        
        # Parameter stats
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        dashboard.log(f"Parameters: {total_params:,} (Trainable: {trainable_params:,})")
        
        # Optimizer & Scaler
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params_list, lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = GradScaler(enabled=config.use_fp16)
        
        dashboard.log("[bold green]Starting training loop![/]")
        update_view()
        
        for epoch in range(config.num_epochs):
            # Train
            train_loss = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                config=config,
                epoch=epoch,
                dashboard=dashboard
            )
            
            dashboard.log(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = validate(model, val_loader, device, config, dashboard)
            dashboard.val_loss = val_loss
            dashboard.log(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < dashboard.best_val_loss:
                dashboard.best_val_loss = val_loss
                checkpoint_dir = Path(config.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = checkpoint_dir / "best_model.pt"
                torch.save(model.state_dict(), best_model_path)
                dashboard.log(f"[bold green]New best model saved![/] (Loss: {val_loss:.4f})")
            
            # Update epoch progress
            dashboard.progress_group.advance(dashboard.epoch_task)
            update_view()
            
        dashboard.log("[bold magenta]Training complete![/]")
        # Keep the final screen visible for a few seconds or wait for user input if desired
        time.sleep(5) 
