"""
Data loading and preprocessing for the latent-space reasoning model.

Handles SAMSum dataset loading, tokenization, and batching.
"""

import json
import os
import threading
import mmap
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)

from src.models import LatentEncoder


class SAMSumDataset(Dataset):
    """Dataset for SAMSum dialogue summarization task."""
    
    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "knkarthick/samsum",
        modernbert_tokenizer_name: str = "answerdotai/ModernBERT-base",
        gpt2_tokenizer_name: str = "gpt2",
        max_seq_len: int = 256,
        max_target_len: int = 128,
        max_samples: int = None
    ):
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_target_len = max_target_len
        
        # Load tokenizers
        self.modernbert_tokenizer = AutoTokenizer.from_pretrained(modernbert_tokenizer_name)
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_tokenizer_name)
        
        # Add pad token to decoder tokenizer if not present
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        # Load SAMSum dataset
        print(f"Loading {dataset_name} {split} split...")
        dataset = load_dataset(dataset_name, split=split)
        
        # Limit samples if specified
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = dataset
        print(f"Loaded {len(self.data)} samples from {split} split")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample."""
        item = self.data[idx]
        
        dialogue = item["dialogue"]
        summary = item["summary"]
        
        # Tokenize input (dialogue) with ModernBERT tokenizer
        input_encoding = self.modernbert_tokenizer(
            dialogue,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target (summary) with GPT-2 tokenizer
        target_encoding = self.gpt2_tokenizer(
            summary,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "target_ids": target_encoding["input_ids"].squeeze(0),
            "target_attention_mask": target_encoding["attention_mask"].squeeze(0),
            "dialogue": dialogue,
            "summary": summary
        }


class SAMSumIdentityDataset(Dataset):
    """Dataset for identity mapping on SAMSum dialogues (dialogue -> dialogue)."""

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "knkarthick/samsum",
        modernbert_tokenizer_name: str = "answerdotai/ModernBERT-base",
        gpt2_tokenizer_name: str = "gpt2",
        max_seq_len: int = 256,
        max_target_len: int = 128,
        max_samples: int = None
    ):
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_target_len = max_target_len

        # Load tokenizers
        self.modernbert_tokenizer = AutoTokenizer.from_pretrained(modernbert_tokenizer_name)
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_tokenizer_name)

        # Add pad token to decoder tokenizer if not present
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

        # Load SAMSum dataset
        print(f"Loading {dataset_name} {split} split for identity mapping...")
        dataset = load_dataset(dataset_name, split=split)

        # Limit samples if specified
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.data = dataset
        print(f"Loaded {len(self.data)} samples from {split} split (identity)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample (dialogue -> dialogue)."""
        item = self.data[idx]

        dialogue = item["dialogue"]

        # Tokenize input (dialogue) with ModernBERT tokenizer
        input_encoding = self.modernbert_tokenizer(
            dialogue,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target (same dialogue) with GPT-2 tokenizer
        target_encoding = self.gpt2_tokenizer(
            dialogue,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "target_ids": target_encoding["input_ids"].squeeze(0),
            "target_attention_mask": target_encoding["attention_mask"].squeeze(0),
            "dialogue": dialogue,
            "summary": dialogue
        }


def collate_fn(batch: list) -> dict:
    """Collate function for DataLoader."""
    # Stack all tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    target_attention_mask = torch.stack([item["target_attention_mask"] for item in batch])
    
    # Keep text for reference
    dialogues = [item["dialogue"] for item in batch]
    summaries = [item["summary"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
        "target_attention_mask": target_attention_mask,
        "dialogues": dialogues,
        "summaries": summaries
    }


def _dataloader_kwargs(config, num_workers_override: int | None = None) -> dict:
    num_workers = config.num_workers if num_workers_override is None else num_workers_override
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True if torch.cuda.is_available() else False
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = getattr(config, "prefetch_factor", 2)
    return kwargs


def create_dataloaders(
    config,
    train_samples: int = None,
    val_samples: int = None
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object
        train_samples: Maximum number of training samples (None for all)
        val_samples: Maximum number of validation samples (None for all)
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SAMSumDataset(
        split=config.train_split,
        dataset_name=config.dataset_name,
        modernbert_tokenizer_name=config.modernbert_model,
        gpt2_tokenizer_name=config.gpt2_model,
        max_seq_len=config.max_seq_len,
        max_target_len=config.max_target_len,
        max_samples=train_samples
    )
    
    val_dataset = SAMSumDataset(
        split=config.validation_split,
        dataset_name=config.dataset_name,
        modernbert_tokenizer_name=config.modernbert_model,
        gpt2_tokenizer_name=config.gpt2_model,
        max_seq_len=config.max_seq_len,
        max_target_len=config.max_target_len,
        max_samples=val_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **_dataloader_kwargs(config)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dataloader_kwargs(config)
    )
    
    return train_loader, val_loader


def create_test_dataloader(config, test_samples: int = None) -> DataLoader:
    """
    Create test dataloader.
    
    Args:
        config: Configuration object
        test_samples: Maximum number of test samples (None for all)
        
    Returns:
        test_loader
    """
    test_dataset = SAMSumDataset(
        split=config.test_split,
        dataset_name=config.dataset_name,
        modernbert_tokenizer_name=config.modernbert_model,
        gpt2_tokenizer_name=config.gpt2_model,
        max_seq_len=config.max_seq_len,
        max_target_len=config.max_target_len,
        max_samples=test_samples
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dataloader_kwargs(config)
    )
    
    return test_loader


class IdentityDataset(Dataset):
    """
    Test C: Simple identity task dataset.
    Input: "Hello world" -> Output: "Hello world"
    
    This tests if the decoder adapter can learn to reproduce input text.
    If it fails on this trivial task, the adapter architecture is flawed.
    """
    
    # Simple sentences for identity task
    IDENTITY_SAMPLES = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "How are you doing today?",
        "This is a simple test sentence",
        "Machine learning is fascinating",
        "I like to code in Python",
        "The weather is nice today",
        "What time is it?",
        "Please pass the salt",
        "Good morning everyone",
        "See you later",
        "Nice to meet you",
        "Have a great day",
        "Thank you very much",
        "You are welcome",
        "I am learning new things",
        "The book is on the table",
        "She walked to the store",
        "He plays guitar well",
        "We are going home",
        "They finished the project",
        "The cat sat on the mat",
        "Birds fly in the sky",
        "Fish swim in water",
        "Trees grow in forests",
        "The sun rises in the east",
        "Stars shine at night",
        "Rain falls from clouds",
        "Snow covers the mountains",
        "Rivers flow to the sea",
    ]
    
    def __init__(
        self,
        modernbert_tokenizer_name: str = "answerdotai/ModernBERT-base",
        gpt2_tokenizer_name: str = "gpt2",
        max_seq_len: int = 64,
        max_target_len: int = 64,
        repeat_samples: int = 100  # Repeat samples to create larger dataset
    ):
        self.max_seq_len = max_seq_len
        self.max_target_len = max_target_len
        
        # Load tokenizers
        self.modernbert_tokenizer = AutoTokenizer.from_pretrained(modernbert_tokenizer_name)
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_tokenizer_name)
        
        # Add pad token to decoder tokenizer if not present
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        # Create repeated samples
        self.samples = self.IDENTITY_SAMPLES * repeat_samples
        print(f"Created identity dataset with {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample - input and output are the same text."""
        text = self.samples[idx]
        
        # Tokenize input with ModernBERT tokenizer
        input_encoding = self.modernbert_tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target (same text) with GPT-2 tokenizer
        target_encoding = self.gpt2_tokenizer(
            text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "target_ids": target_encoding["input_ids"].squeeze(0),
            "target_attention_mask": target_encoding["attention_mask"].squeeze(0),
            "dialogue": text,  # Keep key name for compatibility
            "summary": text    # Same as input for identity task
        }


def create_identity_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for identity task (Test C).
    
    Args:
        config: Configuration object
        
    Returns:
        train_loader, val_loader
    """
    # Create training dataset with many repetitions
    train_dataset = IdentityDataset(
        modernbert_tokenizer_name=config.modernbert_model,
        gpt2_tokenizer_name=config.gpt2_model,
        max_seq_len=config.max_seq_len,
        max_target_len=config.max_target_len,
        repeat_samples=100  # 3000 training samples
    )
    
    # Create smaller validation dataset
    val_dataset = IdentityDataset(
        modernbert_tokenizer_name=config.modernbert_model,
        gpt2_tokenizer_name=config.gpt2_model,
        max_seq_len=config.max_seq_len,
        max_target_len=config.max_target_len,
        repeat_samples=10  # 300 validation samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **_dataloader_kwargs(config)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dataloader_kwargs(config)
    )
    
    return train_loader, val_loader


def create_phase1_dataloaders(
    config,
    train_samples: int = None,
    val_samples: int = None
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Phase 1 decoder alignment.
    Uses dialogue -> dialogue identity mapping on SAMSum.
    """
    train_dataset = SAMSumIdentityDataset(
        split=config.train_split,
        dataset_name=config.dataset_name,
        modernbert_tokenizer_name=config.modernbert_model,
        gpt2_tokenizer_name=config.gpt2_model,
        max_seq_len=config.max_seq_len,
        max_target_len=config.max_target_len,
        max_samples=train_samples
    )

    val_dataset = SAMSumIdentityDataset(
        split=config.validation_split,
        dataset_name=config.dataset_name,
        modernbert_tokenizer_name=config.modernbert_model,
        gpt2_tokenizer_name=config.gpt2_model,
        max_seq_len=config.max_seq_len,
        max_target_len=config.max_target_len,
        max_samples=val_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **_dataloader_kwargs(config)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dataloader_kwargs(config)
    )

    return train_loader, val_loader


# --- Cached Scientific Pretraining / Fine-tuning ---

def _mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
    sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def _to_float16_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().numpy().astype(np.float16).tolist()


def _quantize_int8(tensor: torch.Tensor) -> tuple[list, float]:
    emb = tensor.detach().cpu().numpy().astype(np.float16)
    if emb.size == 0:
        return [], 1.0
    max_abs = float(np.max(np.abs(emb)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    emb_q = np.clip(np.round(emb / scale), -127, 127).astype(np.int8)
    return emb_q.tolist(), float(scale)


def _dequantize_int8(emb_q: list, scale: float) -> np.ndarray:
    if scale is None or scale == 0:
        scale = 1.0
    emb_q_arr = np.array(emb_q, dtype=np.int8)
    return (emb_q_arr.astype(np.float16) * np.float16(scale)).astype(np.float16)


def _extract_field(item: dict, candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        value = item.get(key, None)
        if value:
            return value
    return None


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def preprocess_and_cache_datasets(config) -> dict:
    """
    Preprocess datasets into jsonl cache: tokenized text, embeddings, and idea vectors.
    Returns paths for cached files.
    """
    cache_dir = Path(config.preprocess_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

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

    if config.skip_preprocessing_if_cached:
        required = []
        if getattr(config, "preprocess_wikitext", True):
            required += [cache_paths["wikitext_train"], cache_paths["wikitext_validation"]]
        if getattr(config, "preprocess_arxiv", True):
            required += [cache_paths["arxiv_train"], cache_paths["arxiv_validation"]]
        if getattr(config, "preprocess_english_pretrain", True):
            required += [cache_paths["english_pretrain_train"], cache_paths["english_pretrain_validation"]]
        if getattr(config, "preprocess_scitldr", False) and hasattr(config, "scitldr_dataset"):
            required += [cache_paths["scitldr_train"], cache_paths["scitldr_validation"]]
        required += [cache_paths["compression_mlp"]]
        # NOTE: We intentionally do NOT return early here. Cached files can be partial
        # (e.g., after an interrupted run), so we validate completeness per dataset below.

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Tokenizers
    modernbert_tokenizer = AutoTokenizer.from_pretrained(config.modernbert_model)
    decoder_tokenizer = AutoTokenizer.from_pretrained(config.gpt2_model)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    cache_store_text = bool(getattr(config, "cache_store_text", False))
    cache_store_input_ids = bool(getattr(config, "cache_store_input_ids", False))
    cache_store_attention_mask = bool(getattr(config, "cache_store_attention_mask", False))
    cache_store_target_ids = bool(getattr(config, "cache_store_target_ids", True))
    cache_store_target_attention_mask = bool(getattr(config, "cache_store_target_attention_mask", True))
    cache_store_embeddings_fp16 = bool(getattr(config, "cache_store_embeddings_fp16", False))
    cache_store_embeddings_int8 = bool(getattr(config, "cache_store_embeddings_int8", True))
    cache_store_ideas_fp16 = bool(getattr(config, "cache_store_ideas_fp16", False))
    cache_store_ideas_int8 = bool(getattr(config, "cache_store_ideas_int8", False))

    # Encoder for embeddings + idea vectors
    encoder = LatentEncoder(
        model_name=config.modernbert_model,
        hidden_dim=config.modernbert_hidden_dim,
        latent_dim=config.latent_dim,
        num_unfrozen_layers=0
    ).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Save compression MLP weights for training alignment
    torch.save(encoder.compression_mlp.state_dict(), cache_paths["compression_mlp"], _use_new_zipfile_serialization=False)

    def _split_dataset(dataset, val_fraction: float):
        if val_fraction <= 0 or len(dataset) < 2:
            return dataset, None
        split = dataset.train_test_split(test_size=val_fraction, seed=42, shuffle=True)
        return split["train"], split["test"]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn()
    )
    progress_lock = threading.Lock()
    gpu_lock = threading.Lock()

    def _safe_update(task_id, **kwargs):
        with progress_lock:
            progress.update(task_id, **kwargs)

    def _finish_task(task_id, description: str):
        # Mark task completed; try public API first, fall back to internal structures
        with progress_lock:
            try:
                task = progress.get_task(task_id)
                total = task.total
            except Exception:
                try:
                    total = progress.tasks[task_id].total
                except Exception:
                    total = None
            if total is not None:
                progress.update(task_id, description=description, completed=total)
            else:
                progress.update(task_id, description=description)

    def _prepare_task(output_path: Path, label: str):
        with progress_lock:
            return progress.add_task(f"{label}: {output_path.name}", total=1)

    def _jsonl_line_count(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    def process_single_text_dataset(dataset, output_path: Path, label: str, task_id: int, expected_rows: int):
        if output_path.exists():
            cached_rows = _jsonl_line_count(output_path)
            if expected_rows > 0 and cached_rows >= expected_rows:
                _safe_update(task_id, total=expected_rows, completed=expected_rows, description=f"[green]Cached {output_path.name} (skipped)")
                return
            _safe_update(
                task_id,
                total=expected_rows or len(dataset),
                completed=cached_rows,
                description=f"[yellow]Cached {output_path.name} incomplete ({cached_rows}/{expected_rows}); regenerating"
            )
        batch_size = max(1, int(getattr(config, "preprocess_batch_size", 16)))
        total_samples = len(dataset)
        _safe_update(task_id, total=total_samples, completed=0, description=f"{label}: {output_path.name}")
        with output_path.open("w", encoding="utf-8") as f:
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start : start + batch_size]
                texts = []
                for i in range(len(next(iter(batch.values())))):
                    text = None
                    for key in ["text", "query", "abstract", "content", "article", "body", "title"]:
                        if key in batch and batch[key][i]:
                            text = batch[key][i]
                            break
                    if text:
                        texts.append(text)
                if not texts:
                    # advance by zero samples (no valid texts in this batch)
                    _safe_update(task_id, advance=0)
                    continue

                input_enc = modernbert_tokenizer(
                    texts,
                    max_length=config.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                target_enc = None
                if cache_store_target_ids or cache_store_target_attention_mask:
                    target_enc = decoder_tokenizer(
                        texts,
                        max_length=config.max_target_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )

                input_ids = input_enc["input_ids"].to(device)
                attention_mask = input_enc["attention_mask"].to(device)

                with gpu_lock, torch.inference_mode():
                    outputs = encoder.modernbert(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state
                    pooled = _mean_pooling(hidden_states, attention_mask)
                    ideas = None
                    if cache_store_ideas_fp16 or cache_store_ideas_int8:
                        ideas = encoder.project_to_latent_sequence(pooled)

                for i in range(len(texts)):
                    row = {}
                    if cache_store_text:
                        row["text"] = texts[i]
                    if cache_store_input_ids:
                        row["input_ids"] = input_enc["input_ids"][i].tolist()
                    if cache_store_attention_mask:
                        row["attention_mask"] = input_enc["attention_mask"][i].tolist()
                    if cache_store_target_ids and target_enc is not None:
                        row["target_ids"] = target_enc["input_ids"][i].tolist()
                    if cache_store_target_attention_mask and target_enc is not None:
                        row["target_attention_mask"] = target_enc["attention_mask"][i].tolist()
                    if cache_store_embeddings_fp16:
                        row["embedding"] = _to_float16_list(pooled[i])
                    if cache_store_embeddings_int8:
                        embedding_q, embedding_scale = _quantize_int8(pooled[i])
                        row["embedding_q"] = embedding_q
                        row["embedding_scale"] = embedding_scale
                    if cache_store_ideas_fp16 and ideas is not None:
                        row["idea"] = _to_float16_list(ideas[i])
                    if cache_store_ideas_int8 and ideas is not None:
                        idea_q, idea_scale = _quantize_int8(ideas[i])
                        row["idea_q"] = idea_q
                        row["idea_scale"] = idea_scale
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                # advance progress by number of samples processed in this batch
                _safe_update(task_id, advance=len(texts))

        _finish_task(task_id, f"[green]Cached {output_path.name}")

    def process_scitldr(dataset, output_path: Path, label: str, task_id: int, expected_rows: int):
        if output_path.exists():
            cached_rows = _jsonl_line_count(output_path)
            if expected_rows > 0 and cached_rows >= expected_rows:
                _safe_update(task_id, total=expected_rows, completed=expected_rows, description=f"[green]Cached {output_path.name} (skipped)")
                return
            _safe_update(
                task_id,
                total=expected_rows or len(dataset),
                completed=cached_rows,
                description=f"[yellow]Cached {output_path.name} incomplete ({cached_rows}/{expected_rows}); regenerating"
            )
        batch_size = max(1, int(getattr(config, "preprocess_batch_size", 16)))
        total_samples = len(dataset)
        _safe_update(task_id, total=total_samples, completed=0, description=f"{label}: {output_path.name}")
        with output_path.open("w", encoding="utf-8") as f:
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start : start + batch_size]
                sources, targets = [], []
                if "source" not in batch or "target" not in batch:
                    _safe_update(task_id, advance=0)
                    continue
                for i in range(len(batch["source"])):
                    source = batch["source"][i]
                    target = batch["target"][i]
                    if source and target:
                        sources.append(source)
                        targets.append(target)

                if not sources:
                    _safe_update(task_id, advance=0)
                    continue

                source_enc = modernbert_tokenizer(
                    sources,
                    max_length=config.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                target_enc = modernbert_tokenizer(
                    targets,
                    max_length=config.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                source_target_enc = None
                target_target_enc = None
                if cache_store_target_ids or cache_store_target_attention_mask:
                    source_target_enc = decoder_tokenizer(
                        sources,
                        max_length=config.max_target_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    target_target_enc = decoder_tokenizer(
                        targets,
                        max_length=config.max_target_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )

                source_input_ids = source_enc["input_ids"].to(device)
                source_attention = source_enc["attention_mask"].to(device)
                target_input_ids = target_enc["input_ids"].to(device)
                target_attention = target_enc["attention_mask"].to(device)

                with gpu_lock, torch.inference_mode():
                    source_outputs = encoder.modernbert(input_ids=source_input_ids, attention_mask=source_attention)
                    target_outputs = encoder.modernbert(input_ids=target_input_ids, attention_mask=target_attention)
                    source_pooled = _mean_pooling(source_outputs.last_hidden_state, source_attention)
                    target_pooled = _mean_pooling(target_outputs.last_hidden_state, target_attention)
                    source_ideas = None
                    target_ideas = None
                    if cache_store_ideas_fp16 or cache_store_ideas_int8:
                        source_ideas = encoder.project_to_latent_sequence(source_pooled)
                        target_ideas = encoder.project_to_latent_sequence(target_pooled)

                for i in range(len(sources)):
                    row = {}
                    if cache_store_text:
                        row["source"] = sources[i]
                        row["target"] = targets[i]
                    if cache_store_input_ids:
                        row["source_input_ids"] = source_enc["input_ids"][i].tolist()
                        row["target_input_ids"] = target_enc["input_ids"][i].tolist()
                    if cache_store_attention_mask:
                        row["source_attention_mask"] = source_enc["attention_mask"][i].tolist()
                        row["target_attention_mask"] = target_enc["attention_mask"][i].tolist()
                    if cache_store_target_ids and source_target_enc is not None and target_target_enc is not None:
                        row["source_target_ids"] = source_target_enc["input_ids"][i].tolist()
                        row["target_target_ids"] = target_target_enc["input_ids"][i].tolist()
                    if cache_store_target_attention_mask and source_target_enc is not None and target_target_enc is not None:
                        row["source_target_attention_mask"] = source_target_enc["attention_mask"][i].tolist()
                        row["target_target_attention_mask"] = target_target_enc["attention_mask"][i].tolist()
                    if cache_store_embeddings_fp16:
                        row["source_embedding"] = _to_float16_list(source_pooled[i])
                        row["target_embedding"] = _to_float16_list(target_pooled[i])
                    if cache_store_embeddings_int8:
                        source_embedding_q, source_embedding_scale = _quantize_int8(source_pooled[i])
                        target_embedding_q, target_embedding_scale = _quantize_int8(target_pooled[i])
                        row["source_embedding_q"] = source_embedding_q
                        row["source_embedding_scale"] = source_embedding_scale
                        row["target_embedding_q"] = target_embedding_q
                        row["target_embedding_scale"] = target_embedding_scale
                    if cache_store_ideas_fp16 and source_ideas is not None and target_ideas is not None:
                        row["source_idea"] = _to_float16_list(source_ideas[i])
                        row["target_idea"] = _to_float16_list(target_ideas[i])
                    if cache_store_ideas_int8 and source_ideas is not None and target_ideas is not None:
                        source_idea_q, source_idea_scale = _quantize_int8(source_ideas[i])
                        target_idea_q, target_idea_scale = _quantize_int8(target_ideas[i])
                        row["source_idea_q"] = source_idea_q
                        row["source_idea_scale"] = source_idea_scale
                        row["target_idea_q"] = target_idea_q
                        row["target_idea_scale"] = target_idea_scale
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                # advance by number of valid pairs processed
                _safe_update(task_id, advance=len(sources))

        _finish_task(task_id, f"[green]Cached {output_path.name}")

    def load_train_only(dataset_name: str, split: str, max_samples: Optional[int]) -> Optional[object]:
        try:
            print(f"  Loading {dataset_name} ({split})...")
            dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"  ERROR loading {dataset_name}: {e}")
            print(f"  Skipping {dataset_name}")
            return None

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"  Loaded {len(dataset)} samples")
        return dataset

    val_fraction = float(getattr(config, "preprocess_validation_fraction", 0.05))

    preprocess_scitldr = getattr(config, "preprocess_scitldr", False) and hasattr(config, "scitldr_dataset")

    jobs = []
    task_ids = {}

    if getattr(config, "preprocess_wikitext", True):
        task_ids["wikitext_train"] = _prepare_task(cache_paths["wikitext_train"], "WikiText Train")
        task_ids["wikitext_validation"] = _prepare_task(cache_paths["wikitext_validation"], "WikiText Validation")

        def _job_wikitext():
            dataset = load_train_only(config.wikitext_dataset, config.wikitext_split, config.wikitext_max_samples)
            if dataset is None:
                _safe_update(task_ids["wikitext_train"], total=1, completed=1, description="[red]WikiText load failed")
                _safe_update(task_ids["wikitext_validation"], total=1, completed=1, description="[red]WikiText load failed")
                return
            train_ds, val_ds = _split_dataset(dataset, val_fraction)
            process_single_text_dataset(
                train_ds,
                cache_paths["wikitext_train"],
                "WikiText Train",
                task_ids["wikitext_train"],
                expected_rows=len(train_ds)
            )
            if val_ds is not None:
                process_single_text_dataset(
                    val_ds,
                    cache_paths["wikitext_validation"],
                    "WikiText Validation",
                    task_ids["wikitext_validation"],
                    expected_rows=len(val_ds)
                )
            else:
                _safe_update(task_ids["wikitext_validation"], total=1, completed=1, description="[yellow]No validation split")

        jobs.append(_job_wikitext)

    if getattr(config, "preprocess_arxiv", True):
        task_ids["arxiv_train"] = _prepare_task(cache_paths["arxiv_train"], "arXiv Train")
        task_ids["arxiv_validation"] = _prepare_task(cache_paths["arxiv_validation"], "arXiv Validation")

        def _job_arxiv():
            dataset = load_train_only(config.arxiv_dataset, config.arxiv_split, config.arxiv_max_samples)
            if dataset is None:
                _safe_update(task_ids["arxiv_train"], total=1, completed=1, description="[red]arXiv load failed")
                _safe_update(task_ids["arxiv_validation"], total=1, completed=1, description="[red]arXiv load failed")
                return
            train_ds, val_ds = _split_dataset(dataset, val_fraction)
            process_single_text_dataset(
                train_ds,
                cache_paths["arxiv_train"],
                "arXiv Train",
                task_ids["arxiv_train"],
                expected_rows=len(train_ds)
            )
            if val_ds is not None:
                process_single_text_dataset(
                    val_ds,
                    cache_paths["arxiv_validation"],
                    "arXiv Validation",
                    task_ids["arxiv_validation"],
                    expected_rows=len(val_ds)
                )
            else:
                _safe_update(task_ids["arxiv_validation"], total=1, completed=1, description="[yellow]No validation split")

        jobs.append(_job_arxiv)

    if getattr(config, "preprocess_english_pretrain", True):
        task_ids["english_pretrain_train"] = _prepare_task(cache_paths["english_pretrain_train"], "English Pretrain Train")
        task_ids["english_pretrain_validation"] = _prepare_task(cache_paths["english_pretrain_validation"], "English Pretrain Validation")

        def _job_english_pretrain():
            dataset = load_train_only(
                config.english_pretrain_dataset,
                config.english_pretrain_split,
                config.english_pretrain_max_samples
            )
            if dataset is None:
                _safe_update(task_ids["english_pretrain_train"], total=1, completed=1, description="[red]English pretrain load failed")
                _safe_update(task_ids["english_pretrain_validation"], total=1, completed=1, description="[red]English pretrain load failed")
                return
            train_ds, val_ds = _split_dataset(dataset, val_fraction)
            process_single_text_dataset(
                train_ds,
                cache_paths["english_pretrain_train"],
                "English Pretrain Train",
                task_ids["english_pretrain_train"],
                expected_rows=len(train_ds)
            )
            if val_ds is not None:
                process_single_text_dataset(
                    val_ds,
                    cache_paths["english_pretrain_validation"],
                    "English Pretrain Validation",
                    task_ids["english_pretrain_validation"],
                    expected_rows=len(val_ds)
                )
            else:
                _safe_update(task_ids["english_pretrain_validation"], total=1, completed=1, description="[yellow]No validation split")

        jobs.append(_job_english_pretrain)

    if preprocess_scitldr:
        task_ids["scitldr_train"] = _prepare_task(cache_paths["scitldr_train"], "SciTLDR Train")
        task_ids["scitldr_validation"] = _prepare_task(cache_paths["scitldr_validation"], "SciTLDR Validation")

        def _job_scitldr():
            dataset = load_train_only(config.scitldr_dataset, config.scitldr_train_split, config.scitldr_max_samples)
            if dataset is None:
                _safe_update(task_ids["scitldr_train"], total=1, completed=1, description="[red]SciTLDR load failed")
                _safe_update(task_ids["scitldr_validation"], total=1, completed=1, description="[red]SciTLDR load failed")
                return
            train_ds, val_ds = _split_dataset(dataset, val_fraction)
            process_scitldr(
                train_ds,
                cache_paths["scitldr_train"],
                "SciTLDR Train",
                task_ids["scitldr_train"],
                expected_rows=len(train_ds)
            )
            if val_ds is not None:
                process_scitldr(
                    val_ds,
                    cache_paths["scitldr_validation"],
                    "SciTLDR Validation",
                    task_ids["scitldr_validation"],
                    expected_rows=len(val_ds)
                )
            else:
                _safe_update(task_ids["scitldr_validation"], total=1, completed=1, description="[yellow]No validation split")

        jobs.append(_job_scitldr)

    if jobs:
        max_workers = max(1, (os.cpu_count() or 4) - 2)
        max_workers = min(max_workers, len(jobs))
        with progress:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(job) for job in jobs]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        progress.console.print(f"[red]Preprocessing failed: {exc}")

    return cache_paths


class JsonlIndexedDataset(Dataset):
    """Memory-efficient JSONL reader using mmap with precomputed offsets."""

    def __init__(self, jsonl_path: Path, max_samples: int | None = None, use_index: bool = True):
        self.jsonl_path = jsonl_path
        self._offsets: list[int] = []
        self._lengths: list[int] = []
        self._mmap = None
        self._fp = None
        if use_index and self._load_index(max_samples):
            return
        self._build_index(max_samples)
        if use_index:
            self._save_index()

    def __len__(self) -> int:
        return len(self._offsets)

    def _ensure_mmap(self) -> None:
        if self._mmap is None:
            self._fp = self.jsonl_path.open("rb")
            self._mmap = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)

    def _read_line(self, idx: int) -> str:
        self._ensure_mmap()
        offset = self._offsets[idx]
        length = self._lengths[idx]
        line_bytes = self._mmap[offset: offset + length]
        return line_bytes.decode("utf-8")

    def _index_path(self) -> Path:
        return Path(str(self.jsonl_path) + ".idx.npz")

    def _build_index(self, max_samples: int | None = None) -> None:
        self._offsets = []
        self._lengths = []
        with self.jsonl_path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)
                    self._lengths.append(len(line))
                    if max_samples is not None and len(self._offsets) >= max_samples:
                        break

    def _load_index(self, max_samples: int | None = None) -> bool:
        index_path = self._index_path()
        if not index_path.exists():
            return False
        try:
            stat = self.jsonl_path.stat()
            with np.load(index_path, allow_pickle=False) as data:
                file_size = int(data.get("file_size", -1))
                file_mtime = int(data.get("file_mtime", -1))
                if file_size != stat.st_size or file_mtime != int(stat.st_mtime):
                    return False
                offsets = data["offsets"].astype(np.int64).tolist()
                lengths = data["lengths"].astype(np.int32).tolist()
            if max_samples is not None:
                offsets = offsets[:max_samples]
                lengths = lengths[:max_samples]
            self._offsets = offsets
            self._lengths = lengths
            return True
        except Exception:
            return False

    def _save_index(self) -> None:
        try:
            stat = self.jsonl_path.stat()
            offsets = np.asarray(self._offsets, dtype=np.int64)
            lengths = np.asarray(self._lengths, dtype=np.int32)
            np.savez(
                self._index_path(),
                offsets=offsets,
                lengths=lengths,
                file_size=np.asarray(stat.st_size, dtype=np.int64),
                file_mtime=np.asarray(int(stat.st_mtime), dtype=np.int64)
            )
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmap"] = None
        state["_fp"] = None
        return state

    def __del__(self):
        try:
            if self._mmap is not None:
                self._mmap.close()
        finally:
            if self._fp is not None:
                self._fp.close()


class CachedIdeaDataset(JsonlIndexedDataset):
    def __init__(self, jsonl_path: Path, max_samples: int | None = None, use_index: bool = True):
        super().__init__(jsonl_path=jsonl_path, max_samples=max_samples, use_index=use_index)

    def __getitem__(self, idx: int) -> dict:
        line = self._read_line(idx)
        sample = json.loads(line)
        if "idea_q" in sample and "idea_scale" in sample:
            idea = _dequantize_int8(sample["idea_q"], sample["idea_scale"])
            return {"idea": torch.tensor(idea, dtype=torch.float16)}
        if "idea" in sample:
            return {"idea": torch.tensor(sample["idea"], dtype=torch.float16)}
        if "embedding_q" in sample and "embedding_scale" in sample:
            emb = _dequantize_int8(sample["embedding_q"], sample["embedding_scale"])
            return {"embedding": torch.tensor(emb, dtype=torch.float16)}
        if "embedding" in sample:
            return {"embedding": torch.tensor(sample["embedding"], dtype=torch.float16)}
        raise KeyError("Missing idea/embedding fields in cached sample")


class CachedAdapterDataset(JsonlIndexedDataset):
    def __init__(self, jsonl_path: Path, idea_key: str, target_ids_key: str, target_mask_key: str, max_samples: int | None = None, use_index: bool = True):
        super().__init__(jsonl_path=jsonl_path, max_samples=max_samples, use_index=use_index)
        self.idea_key = idea_key
        self.target_ids_key = target_ids_key
        self.target_mask_key = target_mask_key

    def __getitem__(self, idx: int) -> dict:
        line = self._read_line(idx)
        sample = json.loads(line)
        if f"{self.idea_key}_q" in sample and f"{self.idea_key}_scale" in sample:
            idea = _dequantize_int8(sample[f"{self.idea_key}_q"], sample[f"{self.idea_key}_scale"])
            idea_tensor = torch.tensor(idea, dtype=torch.float16)
            return {
                "idea": idea_tensor,
                "target_ids": torch.tensor(sample[self.target_ids_key], dtype=torch.long),
                "target_attention_mask": torch.tensor(sample[self.target_mask_key], dtype=torch.long)
            }
        if self.idea_key in sample:
            idea_tensor = torch.tensor(sample[self.idea_key], dtype=torch.float16)
            return {
                "idea": idea_tensor,
                "target_ids": torch.tensor(sample[self.target_ids_key], dtype=torch.long),
                "target_attention_mask": torch.tensor(sample[self.target_mask_key], dtype=torch.long)
            }
        embedding_key = self.idea_key.replace("idea", "embedding")
        if f"{embedding_key}_q" in sample and f"{embedding_key}_scale" in sample:
            emb = _dequantize_int8(sample[f"{embedding_key}_q"], sample[f"{embedding_key}_scale"])
            return {
                "embedding": torch.tensor(emb, dtype=torch.float16),
                "target_ids": torch.tensor(sample[self.target_ids_key], dtype=torch.long),
                "target_attention_mask": torch.tensor(sample[self.target_mask_key], dtype=torch.long)
            }
        if embedding_key in sample:
            return {
                "embedding": torch.tensor(sample[embedding_key], dtype=torch.float16),
                "target_ids": torch.tensor(sample[self.target_ids_key], dtype=torch.long),
                "target_attention_mask": torch.tensor(sample[self.target_mask_key], dtype=torch.long)
            }
        raise KeyError("Missing idea/embedding fields in cached sample")


class CachedSciTLDRPairs(JsonlIndexedDataset):
    def __init__(self, jsonl_path: Path, max_samples: int | None = None, use_index: bool = True):
        super().__init__(jsonl_path=jsonl_path, max_samples=max_samples, use_index=use_index)

    def __getitem__(self, idx: int) -> dict:
        line = self._read_line(idx)
        sample = json.loads(line)
        if "source_idea_q" in sample and "source_idea_scale" in sample:
            source_idea = _dequantize_int8(sample["source_idea_q"], sample["source_idea_scale"])
            source_idea_tensor = torch.tensor(source_idea, dtype=torch.float16)
        elif "source_idea" in sample:
            source_idea_tensor = torch.tensor(sample["source_idea"], dtype=torch.float16)
        elif "source_embedding_q" in sample and "source_embedding_scale" in sample:
            source_emb = _dequantize_int8(sample["source_embedding_q"], sample["source_embedding_scale"])
            source_idea_tensor = torch.tensor(source_emb, dtype=torch.float16)
        else:
            source_idea_tensor = torch.tensor(sample["source_embedding"], dtype=torch.float16)
        if "target_idea_q" in sample and "target_idea_scale" in sample:
            target_idea = _dequantize_int8(sample["target_idea_q"], sample["target_idea_scale"])
            target_idea_tensor = torch.tensor(target_idea, dtype=torch.float16)
        elif "target_idea" in sample:
            target_idea_tensor = torch.tensor(sample["target_idea"], dtype=torch.float16)
        elif "target_embedding_q" in sample and "target_embedding_scale" in sample:
            target_emb = _dequantize_int8(sample["target_embedding_q"], sample["target_embedding_scale"])
            target_idea_tensor = torch.tensor(target_emb, dtype=torch.float16)
        else:
            target_idea_tensor = torch.tensor(sample["target_embedding"], dtype=torch.float16)
        return {
            "source_idea": source_idea_tensor,
            "target_idea": target_idea_tensor,
            "summary_ids": torch.tensor(sample["target_target_ids"], dtype=torch.long),
            "summary_attention_mask": torch.tensor(sample["target_target_attention_mask"], dtype=torch.long)
        }


def collate_cached_ideas(batch: list, model = None, config = None) -> dict:
    if "idea" in batch[0]:
        ideas = torch.stack([item["idea"] for item in batch])
        ideas = ideas.to(dtype=torch.float16)

        # Append STOP_LATENT if model is provided and use_stop_latent is enabled
        if model is not None and getattr(config, "use_stop_latent", True):
            stop_latent = model.get_stop_latent(device=ideas.device)
            stop_latent = stop_latent.to(dtype=torch.float16)
            ideas = torch.cat([ideas, stop_latent.unsqueeze(0)], dim=0)

        return {"idea": ideas}

    embeddings = torch.stack([item["embedding"] for item in batch])
    embeddings = embeddings.to(dtype=torch.float16)
    return {"embedding": embeddings}


def collate_cached_adapter(batch: list, model = None, config = None) -> dict:
    target_ids = torch.stack([item["target_ids"] for item in batch])
    target_attention_mask = torch.stack([item["target_attention_mask"] for item in batch])

    if "idea" in batch[0]:
        ideas = torch.stack([item["idea"] for item in batch])
        ideas = ideas.to(dtype=torch.float16)

        # Append STOP_LATENT and EOS if model is provided and use_stop_latent is enabled
        if model is not None and getattr(config, "use_stop_latent", True):
            stop_latent = model.get_stop_latent(device=ideas.device)
            stop_latent = stop_latent.to(dtype=torch.float16)
            eos_token_id = model.gpt2.config.eos_token_id

            # Append STOP_LATENT to ideas (1 sample for STOP_LATENT)
            ideas = torch.cat([ideas, stop_latent.unsqueeze(0)], dim=0)

            # Create target for STOP_LATENT sample: sequence of all EOS tokens
            eos_target = torch.full((1, target_ids.size(1)), fill_value=eos_token_id, dtype=target_ids.dtype, device=target_ids.device)
            target_ids = torch.cat([target_ids, eos_target], dim=0)

            # Create attention for STOP_LATENT sample: all ones (all valid)
            eos_attention = torch.ones((1, target_attention_mask.size(1)), dtype=target_attention_mask.dtype, device=target_attention_mask.device)
            target_attention_mask = torch.cat([target_attention_mask, eos_attention], dim=0)

        return {
            "idea": ideas,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask
        }

    embeddings = torch.stack([item["embedding"] for item in batch])
    embeddings = embeddings.to(dtype=torch.float16)
    return {
        "embedding": embeddings,
        "target_ids": target_ids,
        "target_attention_mask": target_attention_mask
    }


def collate_cached_pairs(batch: list, model = None, config = None) -> dict:
    source_ideas = torch.stack([item["source_idea"] for item in batch])
    source_ideas = source_ideas.to(dtype=torch.float16)
    target_ideas = torch.stack([item["target_idea"] for item in batch])
    target_ideas = target_ideas.to(dtype=torch.float16)
    summary_ids = torch.stack([item["summary_ids"] for item in batch])
    summary_attention_mask = torch.stack([item["summary_attention_mask"] for item in batch])
    
    # Append STOP_LATENT and EOS if model is provided and use_stop_latent is enabled
    if model is not None and getattr(config, "use_stop_latent", True):
        stop_latent = model.get_stop_latent(device=source_ideas.device)
        stop_latent = stop_latent.to(dtype=torch.float16)
        eos_token_id = model.gpt2.config.eos_token_id
        
        # Append STOP_LATENT to both source and target ideas (1 sample)
        source_ideas = torch.cat([source_ideas, stop_latent.unsqueeze(0)], dim=0)
        target_ideas = torch.cat([target_ideas, stop_latent.unsqueeze(0)], dim=0)
        
        # Create target for STOP_LATENT sample: sequence of all EOS tokens
        eos_target = torch.full((1, summary_ids.size(1)), fill_value=eos_token_id, dtype=summary_ids.dtype, device=summary_ids.device)
        summary_ids = torch.cat([summary_ids, eos_target], dim=0)
        
        # Create attention for STOP_LATENT sample: all ones (all valid)
        eos_attention = torch.ones((1, summary_attention_mask.size(1)), dtype=summary_attention_mask.dtype, device=summary_attention_mask.device)
        summary_attention_mask = torch.cat([summary_attention_mask, eos_attention], dim=0)
    
    return {
        "source_idea": source_ideas,
        "target_idea": target_ideas,
        "summary_ids": summary_ids,
        "summary_attention_mask": summary_attention_mask
    }


def create_pretrain_middle_dataloaders(config, cache_paths: dict, model = None) -> DataLoader:
    stage_max = getattr(config, "pretrain_middle_max_samples", None)
    use_index = bool(getattr(config, "cache_write_offsets_index", True))
    def _cap(name: str):
        return getattr(config, f"pretrain_middle_max_samples_{name}", None) or stage_max
    datasets = []
    if cache_paths["wikitext_train"].exists():
        datasets.append(CachedIdeaDataset(cache_paths["wikitext_train"], max_samples=_cap("wikitext"), use_index=use_index))
    if cache_paths["arxiv_train"].exists():
        datasets.append(CachedIdeaDataset(cache_paths["arxiv_train"], max_samples=_cap("arxiv"), use_index=use_index))
    if cache_paths["english_pretrain_train"].exists():
        datasets.append(CachedIdeaDataset(cache_paths["english_pretrain_train"], max_samples=_cap("english"), use_index=use_index))
    if cache_paths["scitldr_train"].exists():
        scitldr = CachedAdapterDataset(
            cache_paths["scitldr_train"],
            idea_key="source_idea",
            target_ids_key="source_target_ids",
            target_mask_key="source_target_attention_mask",
            max_samples=_cap("scitldr"),
            use_index=use_index
        )
        datasets.append(scitldr)
    combined = ConcatDataset(datasets)
    
    # Create collate function with model and config
    collate_fn_partial = partial(collate_cached_ideas, model=model, config=config) if model is not None else collate_cached_ideas
    
    batch_size = getattr(config, "pretrain_middle_batch_size", None) or config.batch_size
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_partial,
        **_dataloader_kwargs(config)
    )


def create_pretrain_adapter_dataloader(config, cache_paths: dict, model = None) -> DataLoader:
    stage_max = getattr(config, "pretrain_adapter_max_samples", None)
    use_index = bool(getattr(config, "cache_write_offsets_index", True))
    def _cap(name: str):
        return getattr(config, f"pretrain_adapter_max_samples_{name}", None) or stage_max
    datasets = []
    if cache_paths["wikitext_train"].exists():
        datasets.append(CachedAdapterDataset(cache_paths["wikitext_train"], "idea", "target_ids", "target_attention_mask", max_samples=_cap("wikitext"), use_index=use_index))
    if cache_paths["arxiv_train"].exists():
        datasets.append(CachedAdapterDataset(cache_paths["arxiv_train"], "idea", "target_ids", "target_attention_mask", max_samples=_cap("arxiv"), use_index=use_index))
    if cache_paths["english_pretrain_train"].exists():
        datasets.append(CachedAdapterDataset(cache_paths["english_pretrain_train"], "idea", "target_ids", "target_attention_mask", max_samples=_cap("english"), use_index=use_index))
    if cache_paths["scitldr_train"].exists():
        datasets.append(CachedAdapterDataset(cache_paths["scitldr_train"], "source_idea", "source_target_ids", "source_target_attention_mask", max_samples=_cap("scitldr"), use_index=use_index))
    combined = ConcatDataset(datasets)
    
    # Create collate function with model and config
    collate_fn_partial = partial(collate_cached_adapter, model=model, config=config) if model is not None else collate_cached_adapter
    
    batch_size = getattr(config, "pretrain_adapter_batch_size", None) or config.batch_size
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_partial,
        **_dataloader_kwargs(config)
    )


class SAMSumIdeaDataset(Dataset):
    """Fine-tuning dataset: loads SAMSum dialogues + summaries, computes ideas on-the-fly."""
    
    def __init__(
        self,
        split: str = "train",
        config = None,
        device = None,
        max_samples: int | None = None
    ):
        self.split = split
        self.config = config
        self.device = device or torch.device("cpu")
        
        # Load SAMSum dataset
        print(f"Loading SAMSum {split} split for fine-tuning...")
        self.dataset = load_dataset("knkarthick/samsum", split=split)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        # Tokenizers
        self.modernbert_tokenizer = AutoTokenizer.from_pretrained(config.modernbert_model)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(config.gpt2_model)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        
        # Encoder for computing ideas
        self.encoder = LatentEncoder(
            model_name=config.modernbert_model,
            hidden_dim=config.modernbert_hidden_dim,
            latent_dim=config.latent_dim,
            num_unfrozen_layers=0,
            attn_implementation=getattr(config, "attn_implementation", None),
            use_gradient_checkpointing=getattr(config, "use_gradient_checkpointing", False)
        ).to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Precompute ideas for all samples
        print(f"Computing ideas for {len(self.dataset)} SAMSum samples...")
        self.ideas = []
        self.summaries = []
        self.target_ids_list = []
        self.target_masks_list = []
        
        for idx, item in enumerate(self.dataset):
            dialogue = item["dialogue"]
            summary = item["summary"]
            
            # Compute idea for dialogue
            dialogue_enc = self.modernbert_tokenizer(
                dialogue,
                max_length=config.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            summary_enc = self.modernbert_tokenizer(
                summary,
                max_length=config.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            dialogue_ids = dialogue_enc["input_ids"].to(self.device)
            dialogue_mask = dialogue_enc["attention_mask"].to(self.device)
            summary_ids = summary_enc["input_ids"].to(self.device)
            summary_mask = summary_enc["attention_mask"].to(self.device)
            
            with torch.no_grad():
                dialogue_output = self.encoder.modernbert(input_ids=dialogue_ids, attention_mask=dialogue_mask)
                summary_output = self.encoder.modernbert(input_ids=summary_ids, attention_mask=summary_mask)
                dialogue_pooled = _mean_pooling(dialogue_output.last_hidden_state, dialogue_mask)
                summary_pooled = _mean_pooling(summary_output.last_hidden_state, summary_mask)
                dialogue_idea = self.encoder.project_to_latent_sequence(dialogue_pooled)
                summary_idea = self.encoder.project_to_latent_sequence(summary_pooled)
            
            # Store ideas
            self.ideas.append((dialogue_idea.cpu(), summary_idea.cpu()))
            
            # Tokenize summary for decoding
            summary_target_enc = self.decoder_tokenizer(
                summary,
                max_length=config.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            self.target_ids_list.append(summary_target_enc["input_ids"].squeeze(0))
            self.target_masks_list.append(summary_target_enc["attention_mask"].squeeze(0))
            
            self.summaries.append(summary)
            
            if (idx + 1) % 100 == 0:
                print(f"  Computed ideas for {idx+1}/{len(self.dataset)} samples")
        
        print(f"Loaded {len(self.dataset)} SAMSum samples for fine-tuning")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        dialogue_idea, summary_idea = self.ideas[idx]
        return {
            "source_idea": dialogue_idea.squeeze(0) if dialogue_idea.dim() > 1 else dialogue_idea,
            "target_idea": summary_idea.squeeze(0) if summary_idea.dim() > 1 else summary_idea,
            "summary_ids": self.target_ids_list[idx],
            "summary_attention_mask": self.target_masks_list[idx],
            "summary": self.summaries[idx]
        }


def create_finetune_middle_dataloader(config, cache_paths: dict = None, model = None) -> DataLoader:
    """Create dataloader for middle model fine-tuning with SAMSum."""
    max_samples = getattr(config, "finetune_middle_max_samples", None)
    dataset = SAMSumIdeaDataset(
        split="train",
        config=config,
        device=torch.device(config.device if torch.cuda.is_available() else "cpu"),
        max_samples=max_samples
    )
    
    # Create collate function with model and config
    collate_fn_partial = partial(collate_cached_pairs, model=model, config=config) if model is not None else collate_cached_pairs
    
    batch_size = getattr(config, "finetune_middle_batch_size", None) or config.batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_partial,
        **_dataloader_kwargs(config, num_workers_override=0)
    )


def create_finetune_adapter_dataloader(config, cache_paths: dict = None, model = None) -> DataLoader:
    """Create dataloader for adapter fine-tuning with SAMSum."""
    max_samples = getattr(config, "finetune_adapter_max_samples", None)
    dataset = SAMSumIdeaDataset(
        split="train",
        config=config,
        device=torch.device(config.device if torch.cuda.is_available() else "cpu"),
        max_samples=max_samples
    )
    
    # Create collate function with model and config
    collate_fn_partial = partial(collate_cached_pairs, model=model, config=config) if model is not None else collate_cached_pairs
    
    batch_size = getattr(config, "finetune_adapter_batch_size", None) or config.batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_partial,
        **_dataloader_kwargs(config, num_workers_override=0)
    )


def create_finetune_adapter_dataloaders(config, cache_paths: dict = None, model = None) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for adapter fine-tuning with SAMSum."""
    max_samples = getattr(config, "finetune_adapter_max_samples", None)
    train_dataset = SAMSumIdeaDataset(
        split="train",
        config=config,
        device=torch.device(config.device if torch.cuda.is_available() else "cpu"),
        max_samples=max_samples
    )
    val_dataset = SAMSumIdeaDataset(
        split="validation",
        config=config,
        device=torch.device(config.device if torch.cuda.is_available() else "cpu"),
        max_samples=max_samples
    )
    
    # Create collate function with model and config
    collate_fn_partial = partial(collate_cached_pairs, model=model, config=config) if model is not None else collate_cached_pairs
    
    batch_size = getattr(config, "finetune_adapter_batch_size", None) or config.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_partial,
        **_dataloader_kwargs(config, num_workers_override=0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_partial,
        **_dataloader_kwargs(config, num_workers_override=0)
    )
    return train_loader, val_loader