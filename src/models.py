"""
Model architecture for latent-space reasoning.

Components:
- LatentEncoder: ModernBERT + pooling + compression MLP
- MiddleModel: MLP for latent reasoning
- PrefixAdapter: Expands latent to GPT-2 prefix embeddings
- LatentSpaceModel: Combined model
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM


class LatentEncoder(nn.Module):
    """Encoder that converts text tokens to a compact idea vector."""
    
    def __init__(self, model_name: str, hidden_dim: int = 768, latent_dim: int = 256, 
                 num_unfrozen_layers: int = 0):
        super().__init__()
        self.modernbert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_unfrozen_layers = num_unfrozen_layers
        
        # Compression MLP: hidden_dim -> 512 -> latent_dim
        self.compression_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )
        
        # Freeze ModernBERT backbone by default
        for param in self.modernbert.parameters():
            param.requires_grad = False
    
    def unfreeze_top_layers(self, num_layers: int):
        """Unfreeze the top N transformer layers for fine-tuning."""
        if num_layers <= 0:
            return
        
        # ModernBERT uses 'encoder.layer' structure
        if hasattr(self.modernbert, 'encoder') and hasattr(self.modernbert.encoder, 'layer'):
            total_layers = len(self.modernbert.encoder.layer)
            layers_to_unfreeze = list(self.modernbert.encoder.layer)[-num_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"  - Unfroze top {num_layers} ModernBERT layers (layers {total_layers - num_layers} to {total_layers - 1})")
        # Also try alternative structure for some models
        elif hasattr(self.modernbert, 'layers'):
            total_layers = len(self.modernbert.layers)
            layers_to_unfreeze = list(self.modernbert.layers)[-num_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"  - Unfroze top {num_layers} ModernBERT layers (layers {total_layers - num_layers} to {total_layers - 1})")
    
    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over non-padded tokens."""
        # hidden_states: [B, L, D]
        # attention_mask: [B, L]
        
        # Expand mask to match hidden_states dimensions
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum of hidden states weighted by mask
        sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
        
        # Number of non-padding tokens per sequence
        sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        
        # Mean pooling
        pooled = sum_embeddings / sum_mask
        
        return pooled
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            
        Returns:
            z_in: [B, latent_dim] idea vector
        """
        # Check if any modernbert parameters require grad (unfrozen layers)
        has_unfrozen_layers = any(p.requires_grad for p in self.modernbert.parameters())
        
        if has_unfrozen_layers:
            # Allow gradients to flow through unfrozen layers
            outputs = self.modernbert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, L, hidden_dim]
        else:
            # No gradient for fully frozen backbone
            with torch.no_grad():
                outputs = self.modernbert(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # [B, L, hidden_dim]
        
        # Pool hidden states
        pooled = self.mean_pooling(hidden_states, attention_mask)  # [B, hidden_dim]
        
        # Compress to latent space
        z_in = self.compression_mlp(pooled)  # [B, latent_dim]
        
        return z_in


class MiddleModel(nn.Module):
    """MLP for latent reasoning in idea space."""
    
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(latent_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, latent_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.GELU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the middle model.
        
        Args:
            z_in: [B, latent_dim] input idea vector
            
        Returns:
            z_out: [B, latent_dim] transformed idea vector
        """
        z_out = self.mlp(z_in)
        return z_out


class PrefixAdapter(nn.Module):
    """Expands latent vector to GPT-2 prefix embeddings."""
    
    def __init__(self, latent_dim: int = 256, prefix_len: int = 10, gpt2_hidden_dim: int = 768):
        super().__init__()
        self.latent_dim = latent_dim
        self.prefix_len = prefix_len
        self.gpt2_hidden_dim = gpt2_hidden_dim
        
        # Deeper expansion MLP with no final activation
        # This allows the output to match GPT-2's embedding distribution
        output_dim = prefix_len * gpt2_hidden_dim
        self.expansion_mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim)  # No activation at the end!
        )
    
    def forward(self, z_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the prefix adapter.
        
        Args:
            z_out: [B, latent_dim] transformed idea vector
            
        Returns:
            prefix_embeddings: [B, prefix_len, gpt2_hidden_dim] prefix embeddings
        """
        batch_size = z_out.size(0)
        
        # Direct expansion without L2 normalization (preserves magnitude information)
        expanded = self.expansion_mlp(z_out)  # [B, prefix_len * gpt2_hidden_dim]
        
        # Reshape to prefix embeddings
        prefix_embeddings = expanded.view(batch_size, self.prefix_len, self.gpt2_hidden_dim)
        
        return prefix_embeddings


class LatentSpaceModel(nn.Module):
    """Complete latent-space reasoning model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Determine if we should bypass the middle model
        self.training_phase = getattr(config, 'training_phase', 'normal')
        self.test_mode = getattr(config, 'test_mode', None)
        
        self.bypass_middle = (
            self.test_mode == 'bypass_middle' or 
            self.test_mode == 'phase1_decoder' or 
            self.test_mode == 'phase2_encoder' or
            self.training_phase in ['phase1', 'phase2']
        )
        
        # Initialize components
        self.encoder = LatentEncoder(
            model_name=config.modernbert_model,
            hidden_dim=config.modernbert_hidden_dim,
            latent_dim=config.latent_dim,
            num_unfrozen_layers=getattr(config, 'num_encoder_unfrozen_layers', 0)
        )
        
        self.middle_model = MiddleModel(
            latent_dim=config.latent_dim,
            hidden_dim=config.middle_hidden_dim,
            num_layers=config.middle_layers
        )
        
        self.prefix_adapter = PrefixAdapter(
            latent_dim=config.latent_dim,
            prefix_len=config.prefix_len,
            gpt2_hidden_dim=config.gpt2_hidden_dim
        )
        # LayerNorm to match GPT-2 embedding normalization
        self.prefix_layernorm = nn.LayerNorm(config.gpt2_hidden_dim)
        
        # Load GPT-2 for decoding
        self.gpt2 = AutoModelForCausalLM.from_pretrained(config.gpt2_model)
        
        # Freeze GPT-2 backbone
        for param in self.gpt2.parameters():
            param.requires_grad = False
        
        # Get decoder token embeddings for teacher forcing
        # Handle both GPT-2 style (transformer.wte) and Gemma style (model.embed_tokens)
        if hasattr(self.gpt2, 'transformer') and hasattr(self.gpt2.transformer, 'wte'):
            self.gpt2_embeddings = self.gpt2.transformer.wte  # GPT-2 style
        elif hasattr(self.gpt2, 'model') and hasattr(self.gpt2.model, 'embed_tokens'):
            self.gpt2_embeddings = self.gpt2.model.embed_tokens  # Gemma style
        else:
            raise RuntimeError("Could not find embedding layer in decoder model. Check model architecture.")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            input_ids: [B, L] encoder input token IDs
            attention_mask: [B, L] encoder attention mask
            target_ids: [B, T] target token IDs
            target_attention_mask: [B, T] target attention mask
            
        Returns:
            logits: [B, T, vocab_size] decoder logits
        """
        # Encode input to latent space
        z_in = self.encoder(input_ids, attention_mask)  # [B, latent_dim]
        
        # Transform latent with middle model (or bypass for Phases 1 & 2)
        if self.bypass_middle:
            z_out = z_in  # Skip middle model
        else:
            z_out = self.middle_model(z_in)  # [B, latent_dim]
        
        # Expand to prefix embeddings
        prefix_embeddings = self.prefix_adapter(z_out)  # [B, prefix_len, gpt2_hidden_dim]
        prefix_embeddings = self.prefix_layernorm(prefix_embeddings)
        
        # Prepare target input for teacher forcing
        batch_size = target_ids.size(0)
        # We use target_ids[:, :-1] as input to predict target_ids[:, 1:]
        # BUT we also want the prefix to predict target_ids[:, 0]
        target_input_ids = target_ids[:, :-1]  # [B, T-1]
        target_embeddings = self.gpt2_embeddings(target_input_ids)  # [B, T-1, gpt2_hidden_dim]
        
        # Concatenate prefix and target embeddings
        input_embeds = torch.cat([prefix_embeddings, target_embeddings], dim=1)  # [B, prefix_len + T-1, gpt2_hidden_dim]
        
        # Prepare attention mask for GPT-2
        prefix_attention = torch.ones(batch_size, self.config.prefix_len, device=input_ids.device)
        target_attention = target_attention_mask[:, :-1]
        combined_attention = torch.cat([prefix_attention, target_attention], dim=1)
        
        # Run GPT-2
        outputs = self.gpt2(
            inputs_embeds=input_embeds,
            attention_mask=combined_attention
        )
        
        # LOGIT SLICING FIX:
        # Index prefix_len - 1 is the last token of the prefix, it predicts target[0]
        # Index prefix_len is the first token of the target, it predicts target[1]
        # ... and so on.
        # We want to return logits that correspond to target_ids[:, 0 : T-1]
        logits = outputs.logits[:, self.config.prefix_len - 1 : -1, :]  # [B, T-1, vocab_size]
        
        return logits
    
    def encode_to_latent(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (for inference)."""
        z_in = self.encoder(input_ids, attention_mask)
        
        # Re-check bypass in case it changed (though usually constant for model lifetime)
        if self.bypass_middle:
            z_out = z_in
        else:
            z_out = self.middle_model(z_in)
        return z_out
    
    def get_prefix_embeddings(self, z_out: torch.Tensor) -> torch.Tensor:
        """Get prefix embeddings from latent vector (for inference)."""
        prefix_embeddings = self.prefix_adapter(z_out)
        return self.prefix_layernorm(prefix_embeddings)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        Generate text from input using the latent prefix.
        
        Args:
            input_ids: [B, L] encoder input token IDs
            attention_mask: [B, L] encoder attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            eos_token_id: End of sequence token ID
            
        Returns:
            generated_ids: [B, max_length] generated token IDs
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Encode input to latent space
        z_out = self.encode_to_latent(input_ids, attention_mask)
        
        # Get prefix embeddings
        prefix_embeddings = self.get_prefix_embeddings(z_out)  # [B, prefix_len, gpt2_hidden_dim]
        
        # Start with BOS token
        if eos_token_id is None:
            eos_token_id = self.gpt2.config.eos_token_id
        
        # Initialize generation with prefix
        current_embeds = prefix_embeddings
        current_length = self.config.prefix_len
        
        generated_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        
        for i in range(max_length):
            # Run GPT-2 on current embeddings
            with torch.no_grad():
                outputs = self.gpt2(inputs_embeds=current_embeds)
                logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample or take argmax
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            generated_ids[:, i] = next_token
            
            # Check for EOS
            if (next_token == eos_token_id).all():
                generated_ids = generated_ids[:, :i+1]
                break
            
            # Get embedding for next token and append
            next_token_embed = self.gpt2_embeddings(next_token).unsqueeze(1)  # [B, 1, gpt2_hidden_dim]
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
        
        return generated_ids