"""
Model utilities for PrismRAG
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with standard configuration.
    
    Args:
        model_name: Name or path of the model
        device: Device to load model on
        torch_dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (model, tokenizer)
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code
    )
    
    logger.info(f"Model loaded successfully on device: {model.device}")
    
    return model, tokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1000,
    temperature: float = 1.0,
    top_p: float = 0.9,
    do_sample: bool = True,
    num_return_sequences: int = 1
) -> List[str]:
    """
    Generate text using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to return
        
    Returns:
        List of generated texts
    """
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4000
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode outputs
    generated_texts = []
    for output in outputs:
        # Remove input tokens from output
        generated_tokens = output[inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text.strip())
    
    return generated_texts


def calculate_model_size(model: AutoModelForCausalLM) -> Dict[str, int]:
    """
    Calculate model parameter counts.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def get_model_memory_usage(model: AutoModelForCausalLM) -> Dict[str, float]:
    """
    Get model memory usage information.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with memory usage in GB
    """
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # Get memory info
    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    return {
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "max_memory_allocated_gb": max_memory_allocated
    }


def create_generation_config(
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 1000,
    do_sample: bool = True,
    repetition_penalty: float = 1.1
) -> GenerationConfig:
    """
    Create a generation configuration.
    
    Args:
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_new_tokens: Maximum new tokens to generate
        do_sample: Whether to use sampling
        repetition_penalty: Repetition penalty
        
    Returns:
        GenerationConfig object
    """
    
    return GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=None,  # Will be set by tokenizer
        eos_token_id=None   # Will be set by tokenizer
    )