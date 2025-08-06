"""
Dataset classes for PrismRAG training
"""

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class PrismRAGDataset(Dataset):
    """
    Dataset class for PrismRAG training data.
    
    Handles both distractor resilience and strategic CoT samples,
    with proper tokenization and formatting for instruction tuning.
    """
    
    def __init__(
        self,
        samples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        instruction_template: str = "### 指令:\n{instruction}\n\n### 回答:\n{response}"
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        
        self.logger = logging.getLogger(__name__)
        
        # Preprocess samples
        self._preprocess_samples()
    
    def _preprocess_samples(self):
        """Preprocess and tokenize samples"""
        
        self.processed_samples = []
        
        for i, sample in enumerate(self.samples):
            try:
                processed = self._process_sample(sample)
                if processed:
                    self.processed_samples.append(processed)
            except Exception as e:
                self.logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        self.logger.info(f"Successfully processed {len(self.processed_samples)}/{len(self.samples)} samples")
    
    def _process_sample(self, sample: Dict) -> Optional[Dict]:
        """Process a single sample"""
        
        # Format the conversation
        conversation = self.instruction_template.format(
            instruction=sample["instruction"],
            response=sample["response"]
        )
        
        # Tokenize
        tokenized = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Create labels (same as input_ids, but with instruction part masked)
        labels = tokenized["input_ids"].copy()
        
        # Find the start of the response (after "### 回答:\n")
        response_start_text = "### 回答:\n"
        response_start_tokens = self.tokenizer.encode(response_start_text, add_special_tokens=False)
        
        # Find where response starts in the tokenized sequence
        response_start_idx = self._find_subsequence(tokenized["input_ids"], response_start_tokens)
        
        if response_start_idx is not None:
            # Mask instruction part (set to -100 so it's ignored in loss calculation)
            response_start_idx += len(response_start_tokens)
            labels[:response_start_idx] = [-100] * response_start_idx
        else:
            self.logger.warning("Could not find response start in tokenized sequence")
            return None
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "sample_type": sample["sample_type"]
        }
    
    def _find_subsequence(self, sequence: List[int], subsequence: List[int]) -> Optional[int]:
        """Find the starting index of a subsequence in a sequence"""
        
        if not subsequence:
            return 0
        
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i + len(subsequence)] == subsequence:
                return i
        
        return None
    
    def __len__(self) -> int:
        return len(self.processed_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.processed_samples[idx]
        
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long)
        }
    
    def get_sample_type_distribution(self) -> Dict[str, int]:
        """Get distribution of sample types"""
        
        distribution = {}
        for sample in self.processed_samples:
            sample_type = sample["sample_type"]
            distribution[sample_type] = distribution.get(sample_type, 0) + 1
        
        return distribution


class PrismRAGEvalDataset(Dataset):
    """
    Dataset class for PrismRAG evaluation.
    
    Used for generating responses during evaluation,
    without labels for loss calculation.
    """
    
    def __init__(
        self,
        samples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.logger = logging.getLogger(__name__)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Tokenize instruction only (for generation)
        instruction = sample["instruction"]
        tokenized = self.tokenizer(
            instruction,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "instruction": instruction,
            "expected_response": sample.get("response", ""),
            "sample_type": sample.get("sample_type", "unknown")
        }