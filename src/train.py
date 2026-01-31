"""
Training loop for the latent-space reasoning model.

Handles training, validation, checkpointing, and mixed precision training.
Supports both Rich TUI and simple Tqdm output.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.config import Config
from src.data import (
    create_dataloaders,
    create_identity_dataloaders,
    create_phase1_dataloaders,
    preprocess_and_cache_datasets,
    create_pretrain_middle_dataloaders,
    create_pretrain_adapter_dataloader,
    create_finetune_middle_dataloader,
    create_finetune_adapter_dataloader,
    create_finetune_adapter_dataloaders
)


# --- Common Functions ---

# Decoder pad token ID (use PyTorch's default ignore index)
DECODER_PAD_TOKEN_ID = -100


def append_stop_latent_to_latent_batch(
    z_in: torch.Tensor,
    z_target: torch.Tensor,
    model: LatentSpaceModel
) -> tuple[torch.Tensor, torch.Tensor]:
    stop_latent = model.get_stop_latent(batch_size=z_in.size(0), device=z_in.device)
    stop_latent = stop_latent.to(dtype=z_in.dtype)
    z_in_aug = torch.cat([z_in, stop_latent], dim=0)
    z_target_aug = torch.cat([z_target, stop_latent], dim=0)
    return z_in_aug, z_target_aug


def append_stop_latent_to_decoder_batch(
    z_in: torch.Tensor,
    target_ids: torch.Tensor,
    target_attention_mask: torch.Tensor,
    model: LatentSpaceModel
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len = target_ids.size()
    stop_latent = model.get_stop_latent(batch_size=batch_size, device=z_in.device)
    stop_latent = stop_latent.to(dtype=z_in.dtype)
    eos_token_id = model.gpt2.config.eos_token_id

    eos_targets = torch.full_like(target_ids, fill_value=eos_token_id)
    eos_attention = torch.zeros_like(target_attention_mask)
    eos_attention[:, 0] = 1

    z_in_aug = torch.cat([z_in, stop_latent], dim=0)
    target_ids_aug = torch.cat([target_ids, eos_targets], dim=0)
    target_attention_aug = torch.cat([target_attention_mask, eos_attention], dim=0)
    return z_in_aug, target_ids_aug, target_attention_aug


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
    
    # Use PyTorch default ignore index (-100) which handles variable vocab sizes
    loss_fct = nn.CrossEntropyLoss(ignore_index=DECODER_PAD_TOKEN_ID, reduction='none')
    loss_per_token = loss_fct(logits_flat, target_ids_flat)
    loss_per_token = loss_per_token * target_mask_flat.float()
    loss = loss_per_token.sum() / torch.clamp(target_mask_flat.sum(), min=1e-9)
    
    return loss


def compute_contrastive_loss(z_pred: torch.Tensor, z_target: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z_pred = F.normalize(z_pred, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    logits = torch.matmul(z_pred, z_target.t()) / temperature
    labels = torch.arange(z_pred.size(0), device=z_pred.device)
    loss_forward = F.cross_entropy(logits, labels)
    loss_backward = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_forward + loss_backward)


def compute_latent_loss(z_pred: torch.Tensor, z_target: torch.Tensor, config: Config) -> torch.Tensor:
    mse = F.mse_loss(z_pred, z_target)
    contrastive = compute_contrastive_loss(z_pred, z_target)
    return (config.latent_mse_weight * mse) + (config.contrastive_weight * contrastive)


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


def find_latest_checkpoint(config: Config):
    """Find the latest checkpoint by step number."""
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by step number (extract from filename)
    checkpoints_with_steps = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.stem.split("_")[-1])
            checkpoints_with_steps.append((step, ckpt))
        except (ValueError, IndexError):
            continue
    
    if not checkpoints_with_steps:
        return None
    
    # Return the checkpoint with the highest step number
    latest_step, latest_path = max(checkpoints_with_steps, key=lambda x: x[0])
    return latest_path


def load_checkpoint(checkpoint_path, model, optimizer, scaler, device, config):
    """Load a checkpoint and return the starting epoch and step."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Use weights_only=False to allow loading the Config object
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print("  - Model state loaded")
    
    # Load optimizer state
    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("  - Optimizer state loaded")
    
    # Load scaler state (for mixed precision training)
    if "scaler_state_dict" in checkpoint and scaler is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("  - Scaler state loaded")
        except Exception as e:
            print(f"  - Warning: Could not load scaler state: {e}")
    
    # Get the step and epoch to resume from
    start_step = checkpoint.get("step", 0)
    start_epoch = checkpoint.get("epoch", 0)
    
    print(f"Resuming from epoch {start_epoch}, step {start_step}")
    
    return start_epoch, start_step


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


def train_epoch_tui(model, dataloader, optimizer, scaler, device, config, epoch, dashboard, start_step=0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    dashboard.current_epoch = epoch
    dashboard.progress_group.reset(dashboard.batch_task)
    dashboard.progress_group.update(dashboard.batch_task, total=len(dataloader), description=f"[cyan]Epoch {epoch+1}/{config.num_epochs}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        # Skip batches if resuming from a checkpoint mid-epoch
        if step < start_step:
            continue
            
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

def run_tui_training(config, model, train_loader, val_loader, optimizer, scaler, device, start_epoch=0, start_step=0) -> bool:
    dashboard = TrainingDashboard(config)
    best_model_saved = False
    # Ensure epoch progress bar matches total epochs
    dashboard.progress_group.reset(dashboard.epoch_task)
    dashboard.progress_group.update(dashboard.epoch_task, total=config.num_epochs, completed=start_epoch)
    with Live(dashboard.layout, refresh_per_second=4, screen=True) as live:
        def update_view():
            dashboard.update_display()
            live.update(dashboard.layout)
            
        dashboard.log("Starting training with TUI...")
        if start_epoch > 0 or start_step > 0:
            dashboard.log(f"[yellow]Resuming from epoch {start_epoch}, step {start_step}[/]")
        update_view()
        
        for epoch in range(start_epoch, config.num_epochs):
            train_loss = train_epoch_tui(model, train_loader, optimizer, scaler, device, config, epoch, dashboard, start_step if epoch == start_epoch else 0)
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
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "step": start_step,
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

def train_epoch_simple(model, dataloader, optimizer, scaler, device, config, epoch, start_step=0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        # Skip batches if resuming from a checkpoint mid-epoch
        if step < start_step:
            progress_bar.update(1)
            continue
            
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

def run_simple_training(config, model, train_loader, val_loader, optimizer, scaler, device, start_epoch=0, start_step=0) -> bool:
    print("Starting training (Simple Mode)...")
    if start_epoch > 0 or start_step > 0:
        print(f"Resuming from epoch {start_epoch}, step {start_step}...")
    best_val_loss = float("inf")
    best_model_saved = False
    
    for epoch in range(start_epoch, config.num_epochs):
        train_loss = train_epoch_simple(model, train_loader, optimizer, scaler, device, config, epoch, start_step if epoch == start_epoch else 0)
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
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "step": start_step,
                "config": config
            }, best_model_path, _use_new_zipfile_serialization=False)
            best_model_saved = True
            print(f"Saved best model with val_loss: {val_loss:.4f}")
            
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    return best_model_saved


# --- Two-Stage Scientific Pipeline ---

def train_middle_epoch(model, dataloader, optimizer, scaler, device, config, epoch, target_key: str = "idea"):
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Middle Epoch {epoch+1}")

    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        z_in = batch[target_key].to(device, dtype=torch.float16)
        z_target = z_in
        with autocast(enabled=config.use_fp16):
            z_out = model.middle_model(z_in)
            loss = compute_latent_loss(z_out, z_target, config)
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": total_loss / num_batches})

    if num_batches % config.gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(1, num_batches)


def train_middle_finetune_epoch(model, dataloader, optimizer, scaler, device, config, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Middle Fine-Tune {epoch+1}")

    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        source_idea = batch["source_idea"].to(device, dtype=torch.float16)
        target_idea = batch["target_idea"].to(device, dtype=torch.float16)
        source_idea, target_idea = append_stop_latent_to_latent_batch(source_idea, target_idea, model)

        with autocast(enabled=config.use_fp16):
            z_out = model.middle_model(source_idea)
            loss = compute_latent_loss(z_out, target_idea, config)
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": total_loss / num_batches})

    if num_batches % config.gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(1, num_batches)


def train_adapter_epoch(model, dataloader, optimizer, scaler, device, config, epoch, use_middle: bool, is_summary: bool = False, use_stop_latent: bool = False):
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Adapter Epoch {epoch+1}")

    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        if is_summary:
            z_in = batch["source_idea"].to(device, dtype=torch.float16)
            target_ids = batch["summary_ids"].to(device)
            target_attention = batch["summary_attention_mask"].to(device)
        else:
            z_in = batch["idea"].to(device, dtype=torch.float16)
            target_ids = batch["target_ids"].to(device)
            target_attention = batch["target_attention_mask"].to(device)

        if use_stop_latent:
            z_in, target_ids, target_attention = append_stop_latent_to_decoder_batch(
                z_in,
                target_ids,
                target_attention,
                model
            )

        with autocast(enabled=config.use_fp16):
            logits = model.forward_from_latent(z_in, target_ids, target_attention, use_middle=use_middle)
            loss = compute_loss(logits, target_ids, target_attention)
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": total_loss / num_batches})

    if num_batches % config.gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(1, num_batches)


def validate_adapter_epoch(model, dataloader, device, config, use_middle: bool) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating (Summary)"):
            z_in = batch["source_idea"].to(device)
            target_ids = batch["summary_ids"].to(device)
            target_attention = batch["summary_attention_mask"].to(device)
            with autocast(enabled=config.use_fp16):
                logits = model.forward_from_latent(z_in, target_ids, target_attention, use_middle=use_middle)
                loss = compute_loss(logits, target_ids, target_attention)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / max(1, num_batches)


# --- Main Entry ---

def train(config: Config):
    """Main training function dispatch."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if getattr(config, "pipeline_mode", "legacy") == "two_stage_scientific":
        cache_paths = {}
        if getattr(config, "run_preprocessing", True):
            print("\n" + "="*60)
            print("PREPROCESSING & CACHING STAGE")
            print("="*60)
            print("Running preprocessing/cache step...")
            print("Note: This runs once and caches results to disk.")
            print("Set preprocess_wikitext/arxiv/english_pretrain=False to skip datasets.")
            print("="*60 + "\n")
            cache_paths = preprocess_and_cache_datasets(config)
        else:
            cache_dir = Path(config.preprocess_cache_dir)
            cache_paths = {
                "wikitext_train": cache_dir / "wikitext_train.jsonl",
                "wikitext_validation": cache_dir / "wikitext_validation.jsonl",
                "arxiv_train": cache_dir / "arxiv_train.jsonl",
                "arxiv_validation": cache_dir / "arxiv_validation.jsonl",
                "english_pretrain_train": cache_dir / "english_pretrain_train.jsonl",
                "english_pretrain_validation": cache_dir / "english_pretrain_validation.jsonl",
                "scitldr_train": cache_dir / "scitldr_train.jsonl",
                "scitldr_validation": cache_dir / "scitldr_validation.jsonl",
                "compression_mlp": cache_dir / "compression_mlp.pt"
            }

        print("Initializing model for two-stage pipeline...")
        model = LatentSpaceModel(config).to(device)

        if cache_paths.get("compression_mlp") and Path(cache_paths["compression_mlp"]).exists():
            model.encoder.compression_mlp.load_state_dict(torch.load(cache_paths["compression_mlp"], map_location=device, weights_only=False))
            print("Loaded cached compression MLP weights.")

        if getattr(config, "freeze_encoder_compression_in_pipeline", True):
            for param in model.encoder.compression_mlp.parameters():
                param.requires_grad = False

        scaler = GradScaler(enabled=config.use_fp16)

        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        handoff_best_path = checkpoint_dir / config.handoff_best_model_filename
        pretrain_best_path = checkpoint_dir / config.pretrain_best_model_filename
        finetune_best_path = checkpoint_dir / config.finetune_best_model_filename

        # Pretraining: Middle model
        if getattr(config, "run_pretraining", True):
            print("\nStage 1: Middle Model Pretraining")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.middle_model.parameters():
                param.requires_grad = True

            train_loader = create_pretrain_middle_dataloaders(config, cache_paths)
            
            optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay)
            for epoch in range(config.pretrain_middle_epochs):
                loss = train_middle_epoch(model, train_loader, optimizer, scaler, device, config, epoch)
                print(f"Middle Pretrain Epoch {epoch+1}: loss={loss:.4f}")

            torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_dir / "pretrain_middle.pt", _use_new_zipfile_serialization=False)

            print("\nStage 2: Adapter Pretraining")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.prefix_adapter.parameters():
                param.requires_grad = True
            for param in model.prefix_layernorm.parameters():
                param.requires_grad = True

            adapter_loader = create_pretrain_adapter_dataloader(config, cache_paths)
            
            optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay)
            for epoch in range(config.pretrain_adapter_epochs):
                loss = train_adapter_epoch(
                    model,
                    adapter_loader,
                    optimizer,
                    scaler,
                    device,
                    config,
                    epoch,
                    use_middle=getattr(config, "adapter_pretrain_use_middle", False),
                    is_summary=False,
                    use_stop_latent=False
                )
                print(f"Adapter Pretrain Epoch {epoch+1}: loss={loss:.4f}")

            torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_dir / "pretrain_adapter.pt", _use_new_zipfile_serialization=False)
            torch.save({"model_state_dict": model.state_dict(), "config": config}, pretrain_best_path, _use_new_zipfile_serialization=False)
            torch.save({"model_state_dict": model.state_dict(), "config": config}, handoff_best_path, _use_new_zipfile_serialization=False)

        # Load pretrain best model before fine-tuning if needed
        if getattr(config, "run_finetuning", True) and (not getattr(config, "run_pretraining", True)):
            if handoff_best_path.exists():
                checkpoint = torch.load(handoff_best_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                print(f"Loaded pretrain best model from {handoff_best_path}.")
            elif pretrain_best_path.exists():
                checkpoint = torch.load(pretrain_best_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                print(f"Loaded pretrain best model from {pretrain_best_path}.")
            else:
                print("[WARNING] Pretrain best model not found. Fine-tuning from current weights.")

        # Fine-tuning: Summarization
        if getattr(config, "run_finetuning", True):
            print("\nStage 3: Middle Model Summarization Fine-Tune")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.middle_model.parameters():
                param.requires_grad = True

            # Temporarily use num_workers=0 to avoid CUDA worker issues with model access
            original_num_workers = config.num_workers
            config.num_workers = 0
            finetune_middle_loader = create_finetune_middle_dataloader(config, cache_paths, model=model)
            config.num_workers = original_num_workers
            
            optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay)
            for epoch in range(config.finetune_middle_epochs):
                loss = train_middle_finetune_epoch(model, finetune_middle_loader, optimizer, scaler, device, config, epoch)
                print(f"Middle Fine-tune Epoch {epoch+1}: loss={loss:.4f}")

            torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_dir / "finetune_middle.pt", _use_new_zipfile_serialization=False)

            print("\nStage 4: Adapter + Decoder Summarization Fine-Tune")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.prefix_adapter.parameters():
                param.requires_grad = True
            for param in model.prefix_layernorm.parameters():
                param.requires_grad = True

            # Temporarily use num_workers=0 to avoid CUDA worker issues with model access
            config.num_workers = 0
            finetune_adapter_loader, finetune_val_loader = create_finetune_adapter_dataloaders(config, cache_paths, model=model)
            config.num_workers = original_num_workers
            
            optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay)
            best_val_loss = float("inf")
            for epoch in range(config.finetune_adapter_epochs):
                loss = train_adapter_epoch(
                    model,
                    finetune_adapter_loader,
                    optimizer,
                    scaler,
                    device,
                    config,
                    epoch,
                    use_middle=True,
                    is_summary=True,
                    use_stop_latent=True
                )
                val_loss = validate_adapter_epoch(model, finetune_val_loader, device, config, use_middle=True)
                print(f"Adapter Fine-tune Epoch {epoch+1}: loss={loss:.4f} | val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"model_state_dict": model.state_dict(), "config": config}, finetune_best_path, _use_new_zipfile_serialization=False)
                    torch.save({"model_state_dict": model.state_dict(), "config": config}, handoff_best_path, _use_new_zipfile_serialization=False)

            torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_dir / "finetune_adapter.pt", _use_new_zipfile_serialization=False)

        print("Two-stage pipeline complete.")
        return
    
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
    elif test_mode == 'phase1_decoder':
        print("Creating Phase 1 dataloaders (Identity SAMSum)...")
        train_loader, val_loader = create_phase1_dataloaders(config, train_samples=config.max_train_samples, val_samples=1000)
    elif test_mode == 'phase2_encoder':
        print("Creating Phase 2 dataloaders (Actual SAMSum summarization)...")
        train_loader, val_loader = create_dataloaders(config, train_samples=config.max_train_samples, val_samples=1000)
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
        
        # Unfreeze top N ModernBERT layers for better gradient flow
        num_unfrozen = getattr(config, 'num_encoder_unfrozen_layers', 2)
        model.encoder.unfreeze_top_layers(num_unfrozen)

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
    # Debug: report trainable param counts for key components to ensure projection/prefix are trainable
    try:
        prefix_params = sum(p.numel() for p in model.prefix_adapter.parameters() if p.requires_grad)
        compression_params = sum(p.numel() for p in model.encoder.compression_mlp.parameters() if p.requires_grad)
        middle_params = sum(p.numel() for p in model.middle_model.parameters() if p.requires_grad)
        print(f"  - Prefix adapter trainable params: {prefix_params:,}")
        print(f"  - Encoder compression trainable params: {compression_params:,}")
        print(f"  - Middle model trainable params: {middle_params:,}")
        if prefix_params == 0:
            print("[WARNING] prefix_adapter has 0 trainable parameters. Check freezing logic.")
    except Exception:
        pass

    optimizer = AdamW(trainable_params_list, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=config.use_fp16)
    
    # Find and load latest checkpoint if it exists
    start_epoch = 0
    start_step = 0
    latest_checkpoint = find_latest_checkpoint(config)
    if latest_checkpoint:
        try:
            start_epoch, start_step = load_checkpoint(latest_checkpoint, model, optimizer, scaler, device, config)
            print()
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
    
    if config.use_tui:
        best_model_saved = run_tui_training(config, model, train_loader, val_loader, optimizer, scaler, device, start_epoch, start_step)
    else:
        best_model_saved = run_simple_training(config, model, train_loader, val_loader, optimizer, scaler, device, start_epoch, start_step)

    # Cleanup logic
    cleanup_checkpoints(config, best_model_saved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent-space model training")
    parser.add_argument("--fine-tune", action="store_true", help="Run only fine-tuning using best_model.pt")
    args = parser.parse_args()

    config = Config()
    if getattr(config, "pipeline_mode", "legacy") == "two_stage_scientific":
        if args.fine_tune:
            config.run_pretraining = False
            config.run_finetuning = True
        else:
            config.run_pretraining = True
            config.run_finetuning = False

    train(config)