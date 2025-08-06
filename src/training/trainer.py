"""
PrismRAG Trainer

This module implements the training logic for PrismRAG, combining
distractor resilience and strategic CoT training data.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from .dataset import PrismRAGDataset
from .data_collator import PrismRAGDataCollator


class PrismRAGTrainer:
    """
    Trainer for PrismRAG model fine-tuning.
    
    Handles both distractor resilience and strategic CoT training,
    with support for parameter-efficient fine-tuning using LoRA.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70b-instruct",
        output_dir: str = "models/prismrag",
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Default LoRA configuration
        self.lora_config = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
        
        # Initialize model and tokenizer
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        
        self.logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Apply LoRA if specified
        if self.use_lora:
            self.logger.info("Applying LoRA configuration")
            lora_config = LoraConfig(**self.lora_config)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def prepare_training_data(
        self,
        distractor_samples: List[Dict],
        strategic_cot_samples: List[Dict],
        max_length: int = 4096
    ) -> PrismRAGDataset:
        """
        Prepare training dataset from distractor and strategic CoT samples.
        
        Args:
            distractor_samples: List of distractor training samples
            strategic_cot_samples: List of strategic CoT training samples
            max_length: Maximum sequence length
            
        Returns:
            PrismRAGDataset for training
        """
        
        # Convert samples to training format
        training_samples = []
        
        # Process distractor samples
        for sample in distractor_samples:
            training_sample = self._format_distractor_sample(sample)
            if training_sample:
                training_samples.append(training_sample)
        
        # Process strategic CoT samples
        for sample in strategic_cot_samples:
            training_sample = self._format_strategic_cot_sample(sample)
            if training_sample:
                training_samples.append(training_sample)
        
        self.logger.info(f"Prepared {len(training_samples)} training samples")
        
        return PrismRAGDataset(
            samples=training_samples,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
    
    def _format_distractor_sample(self, sample: Dict) -> Optional[Dict]:
        """Format distractor sample for training"""
        
        try:
            # Combine golden passage and distractor passage as references
            references = f"参考资料 1:\n{sample['golden_passage']}\n\n参考资料 2:\n{sample['distractor_passage']}"
            
            # Build instruction
            instruction = f"""对于此任务，您需要回答一个问题。请提供事实准确、直接和清晰的回复。为了确保考虑正确的事实，您应该始终将您的回答建立在下面提供的参考资料的基础上。如果参考资料与问题无关或没有提供正确的信息，您可以用道歉代替编造事实。

## 参考资料：
{references}

## 问题：
{sample['question']}"""

            # Response is the ground truth answer
            response = sample['answer']
            
            return {
                "instruction": instruction,
                "response": response,
                "sample_type": "distractor_resilience"
            }
            
        except KeyError as e:
            self.logger.warning(f"Missing key in distractor sample: {e}")
            return None
    
    def _format_strategic_cot_sample(self, sample: Dict) -> Optional[Dict]:
        """Format strategic CoT sample for training"""
        
        try:
            # Combine references
            references_text = "\n\n".join([
                f"参考资料 {i+1}:\n{ref}" 
                for i, ref in enumerate(sample['references'])
            ])
            
            # Build instruction
            instruction = f"""对于此任务，您需要回答一个问题。请提供事实准确、直接和清晰的回复。为了确保考虑正确的事实，您应该始终将您的回答建立在下面提供的参考文献的基础上。

## 参考资料：
{references_text}

## 问题：
{sample['question']}

在回答问题之前，请退一步仔细思考回答问题的最佳策略。为您可以采取的推理步骤生成一个大纲，以找到最佳答案。然后，使用大纲逐步思考。"""

            # Response includes strategy, reasoning, and answer
            response = f"""## 策略：
{sample['strategy']}

## 推理：
{sample['reasoning']}

## 答案：
{sample['answer']}"""

            return {
                "instruction": instruction,
                "response": response,
                "sample_type": "strategic_cot"
            }
            
        except KeyError as e:
            self.logger.warning(f"Missing key in strategic CoT sample: {e}")
            return None
    
    def train(
        self,
        train_dataset: PrismRAGDataset,
        eval_dataset: Optional[PrismRAGDataset] = None,
        training_args: Optional[TrainingArguments] = None,
        use_wandb: bool = True,
        wandb_project: str = "prismrag"
    ):
        """
        Train the PrismRAG model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Training arguments (optional)
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        
        # Initialize W&B if specified
        if use_wandb:
            wandb.init(project=wandb_project, name="prismrag-training")
        
        # Default training arguments
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=1e-5,
                weight_decay=0.01,
                warmup_steps=100,
                logging_steps=100,
                save_steps=500,
                eval_steps=500,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False,
                report_to="wandb" if use_wandb else None,
                dataloader_pin_memory=False,
                gradient_checkpointing=True,
                fp16=True,
                remove_unused_columns=False
            )
        
        # Data collator
        data_collator = PrismRAGDataCollator(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        self.logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        self.logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Finish W&B run
        if use_wandb:
            wandb.finish()
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        
        os.makedirs(save_path, exist_ok=True)
        
        if self.use_lora:
            # Save LoRA weights
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a trained model"""
        
        if self.use_lora:
            # Load LoRA weights
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, load_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16,
                device_map=self.device
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        self.logger.info(f"Model loaded from {load_path}")
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model parameter counts"""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }