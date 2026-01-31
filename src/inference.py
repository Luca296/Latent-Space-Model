"""
Inference and generation for the latent-space reasoning model.

Handles text generation from input using the trained model.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path

from src.models import LatentSpaceModel
from src.config import Config


class LatentSpaceInference:
    """Inference wrapper for the latent-space reasoning model."""
    
    def __init__(self, model_path: str, config: Config, device: torch.device = None):
        """
        Initialize inference model.
        
        Args:
            model_path: Path to saved model checkpoint
            config: Configuration object
            device: Device to run inference on (auto-detected if None)
        """
        self.config = config
        
        # Set device
        if device is None:
            device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Initialize model
        print("Loading model...")
        self.model = LatentSpaceModel(config).to(device)
        
        # Load checkpoint (trusted local file)
        torch.serialization.add_safe_globals([Config])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint if "model_state_dict" not in checkpoint else checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
        # Load tokenizers
        print("Loading tokenizers...")
        self.modernbert_tokenizer = AutoTokenizer.from_pretrained(config.modernbert_model)
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(config.gpt2_model)
        
        # Add pad token to GPT-2 if not present
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        print("Inference model ready!")
    
    def generate(
        self,
        input_text: str,
        max_length: int = None,
        temperature: float = None,
        do_sample: bool = None,
        top_p: float = None,
        top_k: int = None
    ) -> str:
        """
        Generate text from input.
        
        Args:
            input_text: Input text to encode
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            
        Returns:
            Generated text
        """
        # Use config defaults if not specified
        if max_length is None:
            max_length = self.config.max_generation_length
        if temperature is None:
            temperature = self.config.temperature
        if do_sample is None:
            do_sample = self.config.do_sample
        if top_p is None:
            top_p = self.config.top_p
        if top_k is None:
            top_k = self.config.top_k
        
        # Tokenize input
        input_encoding = self.modernbert_tokenizer(
            input_text,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = input_encoding["input_ids"].to(self.device)
        attention_mask = input_encoding["attention_mask"].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=self.gpt2_tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.gpt2_tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def generate_batch(
        self,
        input_texts: list[str],
        max_length: int = None,
        temperature: float = None,
        do_sample: bool = None,
        top_p: float = None,
        top_k: int = None
    ) -> list[str]:
        """
        Generate text from multiple inputs.
        
        Args:
            input_texts: List of input texts
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        for input_text in input_texts:
            generated_text = self.generate(
                input_text=input_text,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def get_latent_vector(self, input_text: str) -> torch.Tensor:
        """
        Get the latent vector for an input text.
        
        Args:
            input_text: Input text to encode
            
        Returns:
            Latent vector [latent_dim]
        """
        # Tokenize input
        input_encoding = self.modernbert_tokenizer(
            input_text,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = input_encoding["input_ids"].to(self.device)
        attention_mask = input_encoding["attention_mask"].to(self.device)
        
        # Get latent vector
        with torch.no_grad():
            z_out = self.model.encode_to_latent(input_ids, attention_mask)
        
        return z_out.squeeze(0).cpu()
    
    def interpolate_and_generate(
        self,
        input_text_1: str,
        input_text_2: str,
        alpha: float = 0.5,
        max_length: int = None,
        temperature: float = None,
        do_sample: bool = None
    ) -> str:
        """
        Interpolate between two latent vectors and generate text.
        
        Args:
            input_text_1: First input text
            input_text_2: Second input text
            alpha: Interpolation factor (0.0 = text_1, 1.0 = text_2)
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text from interpolated latent
        """
        # Get latent vectors
        z1 = self.get_latent_vector(input_text_1).unsqueeze(0).to(self.device)
        z2 = self.get_latent_vector(input_text_2).unsqueeze(0).to(self.device)
        
        # Interpolate
        z_interp = (1 - alpha) * z1 + alpha * z2

        # Stop early if interpolated latent matches STOP_LATENT
        if getattr(self.config, "use_stop_latent", True):
            cosine_threshold = getattr(self.config, "stop_latent_cosine_threshold", None)
            l2_threshold = getattr(self.config, "stop_latent_l2_threshold", None)
            stop_mask = self.model.is_stop_latent(z_interp, cosine_threshold=cosine_threshold, l2_threshold=l2_threshold)
            if stop_mask.all():
                return ""
        
        # Get prefix embeddings
        with torch.no_grad():
            prefix_embeddings = self.model.get_prefix_embeddings(z_interp)
        
        # Generate from prefix
        batch_size = 1
        device = self.device
        
        # Use config defaults if not specified
        if max_length is None:
            max_length = self.config.max_generation_length
        if temperature is None:
            temperature = self.config.temperature
        if do_sample is None:
            do_sample = self.config.do_sample
        
        # Initialize generation with prefix
        current_embeds = prefix_embeddings
        current_length = self.config.prefix_len
        
        generated_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        eos_token_id = self.model.gpt2.config.eos_token_id
        
        for i in range(max_length):
            # Run GPT-2 on current embeddings
            with torch.no_grad():
                outputs = self.model.gpt2(inputs_embeds=current_embeds)
                logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Sample or take argmax
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            generated_ids[:, i] = next_token
            
            # Check for EOS
            if next_token.item() == eos_token_id:
                generated_ids = generated_ids[:, :i+1]
                break
            
            # Get embedding for next token and append
            next_token_embed = self.model.gpt2_embeddings(next_token).unsqueeze(1)
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
        
        # Decode
        generated_text = self.gpt2_tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return generated_text


def interactive_inference(model_path: str, config: Config):
    """
    Run interactive inference session.
    
    Args:
        model_path: Path to saved model checkpoint
        config: Configuration object
    """
    # Initialize inference model
    inference = LatentSpaceInference(model_path, config)
    
    print("\n" + "="*50)
    print("Latent-Space Reasoning Model - Interactive Inference")
    print("="*50)
    print("\nCommands:")
    print("  'exit' - Quit the session")
    print("  'interp' - Interpolate between two inputs")
    print("\nEnter input text to generate output:")
    print("-"*50)
    
    while True:
        user_input = input("\nInput: ").strip()
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "interp":
            print("\n--- Interpolation Mode ---")
            text1 = input("Enter first text: ").strip()
            text2 = input("Enter second text: ").strip()
            alpha_str = input("Enter alpha (0.0-1.0, default 0.5): ").strip()
            alpha = float(alpha_str) if alpha_str else 0.5
            
            result = inference.interpolate_and_generate(text1, text2, alpha)
            print(f"\nInterpolated output (alpha={alpha}):")
            print(result)
            continue
        
        if not user_input:
            continue
        
        # Generate output
        print("\nGenerating...")
        output = inference.generate(user_input)
        print(f"\nOutput: {output}")


if __name__ == "__main__":
    config = Config()
    
    # Check if checkpoint exists
    checkpoint_dir = Path(config.checkpoint_dir)
    if checkpoint_dir.exists():
        # Gather all .pt files in the checkpoint dir and let user select
        pt_files = sorted(list(checkpoint_dir.glob("*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pt_files:
            print("No .pt checkpoints found in the checkpoint directory. Please train the model first.")
            exit(1)

        print("Available checkpoints:")
        for i, p in enumerate(pt_files, start=1):
            mtime = p.stat().st_mtime
            print(f"  [{i}] {p.name}  —  {p}")

        # Prompt user to select a model by number
        try:
            sel = input(f"Select model to load [1-{len(pt_files)}] (default 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled. Exiting.")
            exit(1)

        if sel == "":
            sel_idx = 1
        else:
            try:
                sel_idx = int(sel)
            except ValueError:
                print("Invalid selection. Using default (1).")
                sel_idx = 1

        sel_idx = max(1, min(len(pt_files), sel_idx))
        model_path = str(pt_files[sel_idx - 1])
    else:
        print("No checkpoint directory found. Please train the model first.")
        exit(1)
    
    # Run interactive inference
    interactive_inference(model_path, config)