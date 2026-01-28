"""
Main entry point for the latent-space reasoning model.

Supports training and inference modes via command-line arguments.
"""

import argparse
import sys

# Suppress broken torchvision if necessary
try:
    import torchvision
    # If it imported, check if it's actually working by accessing a member
    _ = torchvision.ops.nms
except (ImportError, RuntimeError, AttributeError):
    import sys
    # Setting to None makes future 'import torchvision' raise ImportError
    sys.modules["torchvision"] = None
    sys.modules["torchvision.ops"] = None
    sys.modules["torchvision.transforms"] = None
    print("Warning: torchvision is broken or missing. Masking it to allow execution.")

from pathlib import Path

from src.config import Config
from src.train import train
from src.inference import interactive_inference, LatentSpaceInference


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Latent-Space Reasoning Model - Train or Inference"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference", "generate"],
        default="train",
        help="Mode to run: train, inference (interactive), or generate (batch)"
    )
    
    # Model configuration
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (required for inference/generate)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input text for generation (used with --mode generate)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for generated text (used with --mode generate)"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum number of training samples"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling threshold"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision training"
    )

    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the Rich TUI for training"
    )
    
    parser.add_argument(
        "--website",
        action="store_true",
        help="Launch web interface for training control instead of direct training"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web interface (default: 5000)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web interface (default: 127.0.0.1)"
    )
    
    # Diagnostic test modes
    parser.add_argument(
        "--test-mode",
        type=str,
        choices=["bypass_middle", "identity_task", "phase1_decoder", "phase2_encoder"],
        default=None,
        help="Diagnostic test mode: bypass_middle (Test B), identity_task (Test C), phase1_decoder (Phase 1), or phase2_encoder (Phase 2)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create config
    config = Config()
    
    # Override config with command-line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.device is not None:
        config.device = args.device
    if args.no_fp16:
        config.use_fp16 = False
    if args.no_tui:
        config.use_tui = False
    
    # Override generation parameters
    if args.max_length is not None:
        config.max_generation_length = args.max_length
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.no_sample:
        config.do_sample = False
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.top_k is not None:
        config.top_k = args.top_k
    
    # Diagnostic test mode
    if args.test_mode is not None:
        config.test_mode = args.test_mode
    
    # Run based on mode
    if args.website:
        # Launch web interface instead of direct training
        from src.web import run_web_server
        run_web_server(host=args.host, port=args.port)
    elif args.mode == "train":
        print("="*50)
        print("Latent-Space Reasoning Model - Training")
        print("="*50)
        print(f"\nConfiguration:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Max train samples: {config.max_train_samples}")
        print(f"  Device: {config.device}")
        print(f"  Mixed precision: {config.use_fp16}")
        print(f"  Latent dim: {config.latent_dim}")
        print(f"  Prefix length: {config.prefix_len}")
        if config.test_mode:
            print(f"  Test mode: {config.test_mode}")
        print("="*50 + "\n")
        
        train(config)
    
    elif args.mode == "inference":
        # Determine checkpoint path
        if args.checkpoint is None:
            checkpoint_dir = Path(config.checkpoint_dir)
            if checkpoint_dir.exists():
                best_model_path = checkpoint_dir / "best_model.pt"
                if best_model_path.exists():
                    checkpoint_path = str(best_model_path)
                else:
                    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
                    if checkpoints:
                        checkpoint_path = str(checkpoints[-1])
                    else:
                        print("Error: No checkpoint found. Please train the model first.")
                        return
            else:
                print("Error: No checkpoint directory found. Please train the model first.")
                return
        else:
            checkpoint_path = args.checkpoint
        
        print("="*50)
        print("Latent-Space Reasoning Model - Interactive Inference")
        print("="*50)
        print(f"\nLoading checkpoint: {checkpoint_path}")
        print("="*50 + "\n")
        
        interactive_inference(checkpoint_path, config)
    
    elif args.mode == "generate":
        # Check for required arguments
        if args.checkpoint is None:
            print("Error: --checkpoint is required for generation mode")
            return
        
        if args.input is None:
            print("Error: --input is required for generation mode")
            return
        
        # Initialize inference model
        import torch
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        inference = LatentSpaceInference(args.checkpoint, config, device)
        
        # Generate output
        print(f"\nGenerating from: {args.input}")
        print("-"*50)
        
        output = inference.generate(
            input_text=args.input,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=not args.no_sample,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        print(f"\nGenerated output:\n{output}")
        
        # Save to file if specified
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()