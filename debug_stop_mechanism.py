"""
Debug script to verify STOP_LATENT mechanism is correctly wired.

Loads a single training sample and checks:
1. Latent sequences end with STOP_LATENT
2. Decoder target sequences end with EOS
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import Config
from src.models import LatentSpaceModel
from src.data import create_pretrain_middle_dataloaders, create_pretrain_adapter_dataloader, preprocess_and_cache_datasets


def check_latent_stop(model, dataloader, device, config):
    """Check that latent sequences end with STOP_LATENT."""
    print("\n" + "="*60)
    print("CHECKING MIDDLE-MODEL LATENT SEQUENCES")
    print("="*60)
    
    batch = next(iter(dataloader))
    z_batch = batch["idea"].to(device)  # [B, latent_dim]
    
    # Get STOP_LATENT for comparison
    stop_latent = model.get_stop_latent(device=device)  # [latent_dim]
    
    print(f"Latent sequence length: {z_batch.size(0)}")
    print(f"Latent dimension: {z_batch.size(1)}")
    
    # The last latent vector in the batch should be STOP_LATENT (appended by append_stop_latent_to_latent_batch)
    last_latent = z_batch[-1]  # [latent_dim]
    
    # Check using cosine similarity
    cosine_sim = F.cosine_similarity(last_latent.unsqueeze(0), stop_latent.unsqueeze(0), dim=-1)
    is_stop = cosine_sim.item() > 0.99  # High threshold for exact match
    
    print(f"Last latent vector cosine similarity to STOP_LATENT: {cosine_sim.item():.6f}")
    print(f"Ends with STOP_LATENT: {is_stop}")
    
    return is_stop


def check_decoder_eos(model, dataloader, device, config):
    """Check that decoder target sequences end with EOS."""
    print("\n" + "="*60)
    print("CHECKING ADAPTER/DECODER TARGET SEQUENCES")
    print("="*60)
    
    batch = next(iter(dataloader))
    target_ids = batch["target_ids"].to(device)  # [B, seq_len]
    target_mask = batch["target_attention_mask"].to(device)  # [B, seq_len]
    
    eos_token_id = model.gpt2.config.eos_token_id
    
    print(f"Target token sequence length: {target_ids.size(1)}")
    print(f"Batch size: {target_ids.size(0)}")
    
    # The last token in each sequence should be EOS (appended by append_stop_latent_to_decoder_batch)
    last_tokens = target_ids[:, -1]  # [B]
    
    # Check if all end with EOS
    all_eos = (last_tokens == eos_token_id).all().item()
    
    # Also check the last non-padded token
    valid_lengths = target_mask.sum(dim=1)
    first_sample_valid_len = valid_lengths[0].item()
    first_sample_last_valid_idx = first_sample_valid_len - 1
    first_sample_last_token = target_ids[0, first_sample_last_valid_idx].item()
    
    print(f"EOS token ID: {eos_token_id}")
    print(f"Last token in first sample (at index {first_sample_last_valid_idx}): {first_sample_last_token}")
    print(f"All sequences end with EOS: {all_eos}")
    
    return all_eos


def main():
    print("\n" + "="*60)
    print("STOP_LATENT MECHANISM DEBUG SCRIPT")
    print("="*60)
    
    # Load config
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Use STOP_LATENT: {getattr(config, 'use_stop_latent', True)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = LatentSpaceModel(config).to(device)
    model.eval()
    
    # Check if we need to run preprocessing
    cache_dir = Path(config.preprocess_cache_dir)
    cache_paths = {
        "wikitext_train": cache_dir / "wikitext_train.jsonl",
        "arxiv_train": cache_dir / "arxiv_train.jsonl",
        "english_pretrain_train": cache_dir / "english_pretrain_train.jsonl",
        "scitldr_train": cache_dir / "scitldr_train.jsonl",
        "compression_mlp": cache_dir / "compression_mlp.pt"
    }
    
    # Check if cache exists
    cache_exists = all(
        cache_paths[key].exists() 
        for key in ["wikitext_train", "arxiv_train", "english_pretrain_train", "compression_mlp"]
    )
    
    if not cache_exists:
        print("\nCache not found. Running preprocessing...")
        cache_paths = preprocess_and_cache_datasets(config)
    else:
        print("\nUsing existing cache...")
    
    # Create dataloaders for middle model
    print("\nCreating pretrain_middle dataloader...")
    try:
        # Temporarily override num_workers for debug
        original_num_workers = config.num_workers
        config.num_workers = 0
        middle_loader = create_pretrain_middle_dataloaders(config, cache_paths, model=model)
        config.num_workers = original_num_workers
        middle_ok = check_latent_stop(model, middle_loader, device, config)
    except Exception as e:
        print(f"Error loading middle model data: {e}")
        import traceback
        traceback.print_exc()
        middle_ok = False
    
    # Create dataloaders for adapter
    print("\nCreating pretrain_adapter dataloader...")
    try:
        original_num_workers = config.num_workers
        config.num_workers = 0
        adapter_loader = create_pretrain_adapter_dataloader(config, cache_paths, model=model)
        config.num_workers = original_num_workers
        adapter_ok = check_decoder_eos(model, adapter_loader, device, config)
    except Exception as e:
        print(f"Error loading adapter data: {e}")
        import traceback
        traceback.print_exc()
        adapter_ok = False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Middle-model STOP_LATENT check: {'✓ PASS' if middle_ok else '✗ FAIL'}")
    print(f"Adapter-decoder EOS check: {'✓ PASS' if adapter_ok else '✗ FAIL'}")
    
    if middle_ok and adapter_ok:
        print("\n✓ STOP mechanism is correctly integrated!")
    else:
        print("\n✗ STOP mechanism has issues. Review the checks above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
