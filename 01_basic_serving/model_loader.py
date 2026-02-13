"""
Model Loading Module

This module handles loading and managing LLM models.
Key concepts:
- Model initialization (loading weights)
- Tokenizer setup
- Device management (CPU/GPU)
- Generation configuration
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LLMModel:
    """
    Wrapper class for LLM model and tokenizer.
    
    This class encapsulates:
    1. Model loading and initialization
    2. Tokenization (text ↔ tokens)
    3. Text generation
    4. Device management
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the model.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2", "distilgpt2")
            device: "cpu" or "cuda" (auto-detected if None)
            torch_dtype: Model precision (float32, float16, etc.)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        logger.info(f"Loading model: {model_name} on device: {device}")
        
        # Load tokenizer
        # Tokenizer converts text to token IDs and vice versa
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist (needed for batching)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        # This downloads weights if not cached, then loads into memory
        model_kwargs = {}
        if torch_dtype:
            model_kwargs["torch_dtype"] = torch_dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move model to device (CPU or GPU)
        self.model.to(self.device)
        
        # Set to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        logger.info(f"Model loaded successfully. Parameters: {self._count_parameters():,}")
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        # Tokenize and convert to PyTorch tensors
        # return_tensors="pt" means return PyTorch tensors
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # Move tensors to the same device as the model
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from a prompt.
        
        This is the core inference function. It:
        1. Tokenizes the prompt
        2. Runs the model forward pass (autoregressive generation)
        3. Decodes tokens back to text
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of different sequences to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            
        Returns:
            Generated text string
        """
        # Tokenize input
        inputs = self.tokenize(prompt)
        
        # Generation configuration
        generation_config = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,  # Use EOS as padding
        }
        
        # Generate text
        # This runs the autoregressive generation loop internally
        with torch.no_grad():  # Disable gradient computation (faster, less memory)
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode tokens back to text
        # Skip special tokens to get clean text
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove the original prompt from output (if it's included)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": self._count_parameters(),
            "vocab_size": len(self.tokenizer),
            "max_position_embeddings": getattr(
                self.model.config,
                "max_position_embeddings",
                "unknown"
            )
        }

