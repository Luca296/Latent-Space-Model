"""
Data loading and preprocessing for the latent-space reasoning model.

Handles SAMSum dataset loading, tokenization, and batching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class SAMSumDataset(Dataset):
    """Dataset for SAMSum dialogue summarization task."""
    
    def __init__(
        self,
        split: str = "train",
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
        
        # Add pad token to GPT-2 if not present
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        # Load SAMSum dataset
        print(f"Loading SAMSum {split} split...")
        dataset = load_dataset("samsum", split=split)
        
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
        modernbert_tokenizer_name=config.modernbert_model,
        gpt2_tokenizer_name=config.gpt2_model,
        max_seq_len=config.max_seq_len,
        max_target_len=config.max_target_len,
        max_samples=train_samples
    )
    
    val_dataset = SAMSumDataset(
        split=config.validation_split,
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
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
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
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader