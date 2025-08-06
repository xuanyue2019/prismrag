"""
Data collator for PrismRAG training
"""

from typing import Dict, List, Any
import torch
from transformers import PreTrainedTokenizer


class PrismRAGDataCollator:
    """
    Data collator for PrismRAG training.
    
    Handles padding and batching of training samples with proper
    attention masks and label handling.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm: bool = False,
        pad_to_multiple_of: int = None
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries from dataset
            
        Returns:
            Batched and padded tensors
        """
        
        # Extract sequences
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # Pad sequences
        batch = self._pad_sequences(input_ids, attention_masks, labels)
        
        return batch
    
    def _pad_sequences(
        self,
        input_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        labels: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Pad sequences to the same length"""
        
        # Find maximum length in batch
        max_length = max(len(seq) for seq in input_ids)
        
        # Apply padding multiple constraint if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        batch_size = len(input_ids)
        
        # Initialize padded tensors
        padded_input_ids = torch.full(
            (batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        padded_attention_masks = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long
        )
        padded_labels = torch.full(
            (batch_size, max_length),
            -100,  # Ignore index for loss calculation
            dtype=torch.long
        )
        
        # Fill in the actual sequences
        for i, (input_id, attention_mask, label) in enumerate(zip(input_ids, attention_masks, labels)):
            seq_len = len(input_id)
            
            padded_input_ids[i, :seq_len] = input_id
            padded_attention_masks[i, :seq_len] = attention_mask
            padded_labels[i, :seq_len] = label
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels
        }


class PrismRAGEvalDataCollator:
    """
    Data collator for PrismRAG evaluation.
    
    Used during inference/evaluation, handles padding without labels.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pad_to_multiple_of: int = None
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of evaluation features.
        
        Args:
            features: List of feature dictionaries from eval dataset
            
        Returns:
            Batched tensors and metadata
        """
        
        # Extract sequences
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]
        
        # Pad sequences
        batch = self._pad_sequences(input_ids, attention_masks)
        
        # Add metadata
        batch["instructions"] = [feature["instruction"] for feature in features]
        batch["expected_responses"] = [feature["expected_response"] for feature in features]
        batch["sample_types"] = [feature["sample_type"] for feature in features]
        
        return batch
    
    def _pad_sequences(
        self,
        input_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Pad sequences to the same length"""
        
        # Find maximum length in batch
        max_length = max(len(seq) for seq in input_ids)
        
        # Apply padding multiple constraint if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        batch_size = len(input_ids)
        
        # Initialize padded tensors
        padded_input_ids = torch.full(
            (batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        padded_attention_masks = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long
        )
        
        # Fill in the actual sequences
        for i, (input_id, attention_mask) in enumerate(zip(input_ids, attention_masks)):
            seq_len = len(input_id)
            
            padded_input_ids[i, :seq_len] = input_id
            padded_attention_masks[i, :seq_len] = attention_mask
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks
        }