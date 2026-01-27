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
        
        # Add pad token to GPT-2 if not present
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

        # Add pad token to GPT-2 if not present
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
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
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
        
        # Add pad token to GPT-2 if not present
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