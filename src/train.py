"""
Training loop for the latent-space reasoning model.

Handles training, validation, checkpointing, and mixed precision training.
Supports both Rich TUI and simple Tqdm output.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import glob

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
    MofNCompleteColumn
)
from rich.text import Text
from rich import box

from src.models import LatentSpaceModel
from src.data import create_dataloaders, create_identity_dataloaders, create_phase1_dataloaders
from src.config import Config


# --- Common Functions ---

def compute_loss(logits: torch.Tensor, target_ids: torch.Tensor, 
                 target_attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for decoder outputs."""
    # We now predict target_ids[:, :-1] using prefix + target_ids[:, :-1]
    # Logit at prefix_len-1 predicts target[0]
    # Logit at prefix_len predicts target[1]
    # So we compare logits with target_ids[:, :-1]
    target_ids_to_predict = target_ids[:, :-1]
    target_mask_to_predict = target_attention_mask[:, :-1]
    
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    target_ids_flat = target_ids_to_predict.reshape(-1)
    target_mask_flat = target_mask_to_predict.reshape(-1)
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    loss_per_token = loss_fct(logits_flat, target_ids_flat)
    loss_per_token = loss_per_token * target_mask_flat.float()
    loss = loss_per_token.sum() / torch.clamp(target_mask_flat.sum(), min=1e-9)
    
    return loss


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
    }, checkpoint_path, _use_new_zipfile_serialization=False)


def cleanup_checkpoints(config: Config, best_model_saved: bool):
    """No-op: keep all checkpoints and best_model.pt."""
    return


# --- TUI Logic ---

class TrainingDashboard:
    """Manages the Rich TUI for training."""
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.start_time = time.time()
        self.metrics_history = []
        self.logs = []
        
        self.layout = self.make_layout()
        
        self.progress_group = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
        self.epoch_task = self.progress_group.add_task("[yellow]Overall Progress", total=config.num_epochs)
        self.batch_task = self.progress_group.add_task("[cyan]Current Epoch", total=100)
        
        self.current_epoch = 0
        self.current_step = 0
        self.current_loss = 0.0
        self.avg_loss = 0.0
        self.val_loss = None
        self.best_val_loss = float('inf')

    def make_layout(self) -> Layout:
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
        title = Text("Latent-Space Reasoning Model Training", style="bold magenta")
        subtitle = f"Device: {self.config.device} | FP16: {self.config.use_fp16} | Batch: {self.config.batch_size}"
        return Panel(Text.from_markup(f"{title}\n[dim]{subtitle}[/dim]"), style="white on blue", box=box.ROUNDED)

    def generate_metrics_table(self) -> Panel:
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
        if self.best_val_loss != float('inf'):
            table.add_row("Best Val Loss", f"[bold green]{self.best_val_loss:.4f}[/]")
        else:
            table.add_row("Best Val Loss", "[dim]N/A[/]")
        return Panel(table, title="Training Metrics", border_style="yellow", box=box.ROUNDED)

    def generate_logs_panel(self) -> Panel:
        log_text = "\n".join(self.logs[-8:])
        return Panel(log_text, title="Recent Logs", border_style="white", box=box.ROUNDED)
        
    def generate_config_panel(self) -> Panel:
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Param", style="dim cyan")
        table.add_column("Value", style="dim white")
        table.add_row("Dataset", self.config.dataset_name)
        table.add_row("Base Model", self.config.modernbert_model.split('/')[-1])
        table.add_row("Decoder", self.config.gpt2_model)
        table.add_row("Latent Dim", str(self.config.latent_dim))
        return Panel(table, title="Configuration", border_style="blue", box=box.ROUNDED)

    def update_display(self):
        self.layout["header"].update(self.generate_header())
        self.layout["left"].update(Panel(self.progress_group, title="Progress", border_style="green", box=box.ROUNDED, padding=(1, 2)))
        right_content = Layout()
        right_content.split_column(Layout(self.generate_metrics_table(), ratio=2), Layout(self.generate_config_panel(), ratio=1))
        self.layout["right"].update(right_content)
        self.layout["footer"].update(self.generate_logs_panel())

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        
    def update_metrics(self, loss: float, avg_loss: float, step: int):
        self.current_loss = loss
        self.avg_loss = avg_loss
        self.current_step = step


def train_epoch_tui(model, dataloader, optimizer, scaler, device, config, epoch, dashboard):
    model.train()
    total_loss = 0.0
    num_batches = 0
    dashboard.current_epoch = epoch
    dashboard.progress_group.reset(dashboard.batch_task)
    dashboard.progress_group.update(dashboard.batch_task, total=len(dataloader), description=f"[cyan]Epoch {epoch+1}/{config.num_epochs}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)
        
        with autocast(enabled=config.use_fp16):
            logits = model(input_ids, attention_mask, target_ids, target_attention_mask)
            loss = compute_loss(logits, target_ids, target_attention_mask)
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step = epoch * len(dataloader) + step + 1
            avg_loss = total_loss / num_batches
            dashboard.update_metrics(loss.item() * config.gradient_accumulation_steps, avg_loss, global_step)
            
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, scaler, global_step, epoch, config)
                dashboard.log(f"[yellow]Checkpoint saved at step {global_step}[/]")
        
        dashboard.progress_group.advance(dashboard.batch_task)
    
    if num_batches % config.gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    return total_loss / num_batches

def validate_tui(model, dataloader, device, config, dashboard):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    dashboard.log("[blue]Starting validation...[/]")
    val_task = dashboard.progress_group.add_task("[magenta]Validating...", total=len(dataloader))
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)
            
            with autocast(enabled=config.use_fp16):
                logits = model(input_ids, attention_mask, target_ids, target_attention_mask)
                loss = compute_loss(logits, target_ids, target_attention_mask)
            
            total_loss += loss.item()
            num_batches += 1
            dashboard.progress_group.advance(val_task)
            
    dashboard.progress_group.remove_task(val_task)
    return total_loss / num_batches

def run_tui_training(config, model, train_loader, val_loader, optimizer, scaler, device) -> bool:
    dashboard = TrainingDashboard(config)
    best_model_saved = False
    # Ensure epoch progress bar matches total epochs
    dashboard.progress_group.reset(dashboard.epoch_task)
    dashboard.progress_group.update(dashboard.epoch_task, total=config.num_epochs, completed=0)
    with Live(dashboard.layout, refresh_per_second=4, screen=True) as live:
        def update_view():
            dashboard.update_display()
            live.update(dashboard.layout)
            
        dashboard.log("Starting training with TUI...")
        update_view()
        
        for epoch in range(config.num_epochs):
            train_loss = train_epoch_tui(model, train_loader, optimizer, scaler, device, config, epoch, dashboard)
            dashboard.log(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
            
            val_loss = validate_tui(model, val_loader, device, config, dashboard)
            dashboard.val_loss = val_loss
            dashboard.log(f"Val Loss: {val_loss:.4f}")
            
            if val_loss < dashboard.best_val_loss:
                dashboard.best_val_loss = val_loss
                checkpoint_dir = Path(config.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config
                }, best_model_path, _use_new_zipfile_serialization=False)
                best_model_saved = True
                dashboard.log(f"[bold green]New best model saved![/] (Loss: {val_loss:.4f})")
                
            dashboard.progress_group.update(dashboard.epoch_task, completed=epoch + 1)
            update_view()
            
        dashboard.log("[bold magenta]Training complete![/]")
        time.sleep(2)
    return best_model_saved


# --- Simple (No-TUI) Logic ---

def train_epoch_simple(model, dataloader, optimizer, scaler, device, config, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)
        
        with autocast(enabled=config.use_fp16):
            logits = model(input_ids, attention_mask, target_ids, target_attention_mask)
            loss = compute_loss(logits, target_ids, target_attention_mask)
            loss = loss / config.gradient_accumulation_steps
            
        scaler.scale(loss).backward()
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": total_loss / num_batches})
            
            global_step = epoch * len(dataloader) + step + 1
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, scaler, global_step, epoch, config)
                
    if num_batches % config.gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    return total_loss / num_batches

def validate_simple(model, dataloader, device, config):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)
            
            with autocast(enabled=config.use_fp16):
                logits = model(input_ids, attention_mask, target_ids, target_attention_mask)
                loss = compute_loss(logits, target_ids, target_attention_mask)
            
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches

def run_simple_training(config, model, train_loader, val_loader, optimizer, scaler, device) -> bool:
    print("Starting training (Simple Mode)...")
    best_val_loss = float("inf")
    best_model_saved = False
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch_simple(model, train_loader, optimizer, scaler, device, config, epoch)
        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f}")
        
        val_loss = validate_simple(model, val_loader, device, config)
        print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config
            }, best_model_path, _use_new_zipfile_serialization=False)
            best_model_saved = True
            print(f"Saved best model with val_loss: {val_loss:.4f}")
            
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    return best_model_saved


# --- Main Entry ---

def train(config: Config):
    """Main training function dispatch."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for diagnostic test modes
    test_mode = getattr(config, 'test_mode', None)
    
    if test_mode == 'bypass_middle':
        print("\n" + "="*60)
        print("TEST B: BYPASS MIDDLE MODEL (z_out = z_in)")
        print("="*60)
        print("If output is still nonsense -> decoder adapter is the problem")
        print("If output improves -> middle model transformation is the issue")
        print("="*60 + "\n")
    elif test_mode == 'identity_task':
        print("\n" + "="*60)
        print("TEST C: IDENTITY TASK - Train decoder adapter only")
        print("="*60)
        print("Freezing: encoder, middle model")
        print("Training: prefix_adapter (expansion MLP) only")
        print("Task: Input='Hello world' -> Output='Hello world'")
        print("If it can't reproduce simple text -> adapter is not aligned")
        print("="*60 + "\n")
    elif test_mode == 'phase1_decoder':
        print("\n" + "="*60)
        print("PHASE 1: ALIGN DECODER (Sequential Training)")
        print("="*60)
        print("Goal: Stabilize latent-to-text mapping")
        print("Freezing: encoder compression MLP, middle model")
        print("Training: prefix_adapter + prefix_layernorm")
        print("Task: Real text reconstruction (SAMSum dialogues)")
        print("="*60 + "\n")
    elif test_mode == 'phase2_encoder':
        print("\n" + "="*60)
        print("PHASE 2: TRAIN ENCODER COMPRESSION (Sequential Training)")
        print("="*60)
        print("Goal: Map input to the stabilized latent space")
        print("Freezing: middle model")
        print("Training: encoder compression + prefix_adapter + prefix_layernorm")
        print("Task: Real text reconstruction (SAMSum dialogues)")
        print("="*60 + "\n")
    
    # Create dataloaders based on test mode
    if test_mode == 'identity_task':
        print("Creating identity task dataloaders...")
        train_loader, val_loader = create_identity_dataloaders(config)
    elif test_mode == 'phase1_decoder' or test_mode == 'phase2_encoder':
        print("Creating Phase 1/2 dataloaders (Identity SAMSum)...")
        train_loader, val_loader = create_phase1_dataloaders(config, train_samples=config.max_train_samples, val_samples=1000)
    else:
        print("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(config, train_samples=config.max_train_samples, val_samples=1000)
    
    print("Initializing model...")
    model = LatentSpaceModel(config).to(device)
    
    # For Test C: Freeze everything except prefix_adapter
    if test_mode == 'identity_task':
        print("\nFreezing components for Test C...")
        
        # Freeze encoder compression MLP
        for param in model.encoder.compression_mlp.parameters():
            param.requires_grad = False
        print("  - Frozen: encoder.compression_mlp")
        
        # Freeze middle model
        for param in model.middle_model.parameters():
            param.requires_grad = False
        print("  - Frozen: middle_model")
        
        # Ensure prefix_adapter is trainable
        for param in model.prefix_adapter.parameters():
            param.requires_grad = True
        print("  - Trainable: prefix_adapter (expansion MLP)")
        print()
    elif test_mode == 'phase1_decoder':
        print("\nFreezing components for Phase 1...")

        # Freeze encoder compression MLP
        for param in model.encoder.compression_mlp.parameters():
            param.requires_grad = False
        print("  - Frozen: encoder.compression_mlp")

        # Freeze middle model
        for param in model.middle_model.parameters():
            param.requires_grad = False
        print("  - Frozen: middle_model")

        # Ensure prefix adapter + LayerNorm are trainable
        for param in model.prefix_adapter.parameters():
            param.requires_grad = True
        for param in model.prefix_layernorm.parameters():
            param.requires_grad = True
        print("  - Trainable: prefix_adapter (expansion MLP)")
        print("  - Trainable: prefix_layernorm")
        print()
    elif test_mode == 'phase2_encoder':
        print("\nConfiguring components for Phase 2...")
        
        # Load Phase 1 Checkpoint
        checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
        if checkpoint_path.exists():
            print(f"Loading Phase 1 checkpoint from {checkpoint_path}...")
            # Use weights_only=False to allow loading the Config object
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Backup Phase 1 model
            backup_path = Path(config.checkpoint_dir) / "best_model_phase1.pt"
            print(f"Backing up Phase 1 model to {backup_path}...")
            torch.save(checkpoint, backup_path)
            
            # strict=False because we might be modifying architecture in future or backward compatibility
            # In this exact flow, keys should match.
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            print("Checkpoint loaded.")
        else:
            print("[WARNING] Phase 1 checkpoint not found! Starting from scratch (not recommended for Phase 2).")

        # Unfreeze encoder compression MLP
        for param in model.encoder.compression_mlp.parameters():
            param.requires_grad = True
        print("  - Trainable: encoder.compression_mlp")

        # Freeze middle model
        for param in model.middle_model.parameters():
            param.requires_grad = False
        print("  - Frozen: middle_model")

        # Ensure prefix adapter + LayerNorm are trainable
        for param in model.prefix_adapter.parameters():
            param.requires_grad = True
        for param in model.prefix_layernorm.parameters():
            param.requires_grad = True
        print("  - Trainable: prefix_adapter (expansion MLP)")
        print("  - Trainable: prefix_layernorm")
        print()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=config.use_fp16)
    
    if config.use_tui:
        best_model_saved = run_tui_training(config, model, train_loader, val_loader, optimizer, scaler, device)
    else:
        best_model_saved = run_simple_training(config, model, train_loader, val_loader, optimizer, scaler, device)

    # Cleanup logic
    cleanup_checkpoints(config, best_model_saved)


if __name__ == "__main__":
    config = Config()
    train(config)