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
    
    def __init__(self, model_name: str, hidden_dim: int = 768, latent_dim: int = 256):
        super().__init__()
        self.modernbert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Compression MLP: hidden_dim -> 512 -> latent_dim
        self.compression_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )
        
        # Freeze ModernBERT backbone
        for param in self.modernbert.parameters():
            param.requires_grad = False
    
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
        # Get ModernBERT hidden states (no gradient for frozen backbone)
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
        
        # Expansion MLP: latent_dim -> prefix_len * gpt2_hidden_dim
        self.expansion_mlp = nn.Sequential(
            nn.Linear(latent_dim, prefix_len * gpt2_hidden_dim),
            nn.GELU()
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
        
        # Expand to prefix dimensions
        expanded = self.expansion_mlp(z_out)  # [B, prefix_len * gpt2_hidden_dim]
        
        # Reshape to prefix embeddings
        prefix_embeddings = expanded.view(batch_size, self.prefix_len, self.gpt2_hidden_dim)
        
        return prefix_embeddings


class LatentSpaceModel(nn.Module):
    """Complete latent-space reasoning model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.encoder = LatentEncoder(
            model_name=config.modernbert_model,
            hidden_dim=config.modernbert_hidden_dim,
            latent_dim=config.latent_dim
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
        
        # Load GPT-2 for decoding
        self.gpt2 = AutoModelForCausalLM.from_pretrained(config.gpt2_model)
        
        # Freeze GPT-2 backbone
        for param in self.gpt2.parameters():
            param.requires_grad = False
        
        # Get GPT-2 token embeddings for teacher forcing
        self.gpt2_embeddings = self.gpt2.transformer.wte
    
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
        
        # Transform latent with middle model
        z_out = self.middle_model(z_in)  # [B, latent_dim]
        
        # Expand to prefix embeddings
        prefix_embeddings = self.prefix_adapter(z_out)  # [B, prefix_len, gpt2_hidden_dim]
        
        # Get target token embeddings (teacher forcing)
        batch_size = target_ids.size(0)
        target_input_ids = target_ids[:, :-1]  # Shift for teacher forcing
        target_embeddings = self.gpt2_embeddings(target_input_ids)  # [B, T-1, gpt2_hidden_dim]
        
        # Concatenate prefix and target embeddings
        input_embeds = torch.cat([prefix_embeddings, target_embeddings], dim=1)  # [B, prefix_len + T-1, gpt2_hidden_dim]
        
        # Prepare attention mask for GPT-2
        prefix_attention = torch.ones(batch_size, self.config.prefix_len, device=input_ids.device)
        target_attention = target_attention_mask[:, :-1]
        combined_attention = torch.cat([prefix_attention, target_attention], dim=1)
        
        # Run GPT-2. Even though GPT-2 is frozen, we need gradients to flow 
        # back through it to the prefix embeddings.
        outputs = self.gpt2(
            inputs_embeds=input_embeds,
            attention_mask=combined_attention
        )
        
        # Get logits for target positions (skip prefix)
        logits = outputs.logits[:, self.config.prefix_len:, :]  # [B, T-1, vocab_size]
        
        return logits
    
    def encode_to_latent(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (for inference)."""
        z_in = self.encoder(input_ids, attention_mask)
        z_out = self.middle_model(z_in)
        return z_out
    
    def get_prefix_embeddings(self, z_out: torch.Tensor) -> torch.Tensor:
        """Get prefix embeddings from latent vector (for inference)."""
        return self.prefix_adapter(z_out)
    
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