#!/usr/bin/env python3
"""
Train PrismRAG model

This script trains the PrismRAG model using the generated training data,
combining distractor resilience and strategic CoT samples.
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List

import torch
import yaml
from transformers import TrainingArguments

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training import PrismRAGTrainer, PrismRAGDataset


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_training_data(data_dir: str) -> tuple[List[Dict], List[Dict]]:
    """Load distractor and strategic CoT training data"""
    
    # Load distractor data
    distractor_path = os.path.join(data_dir, "distractor_training_data.json")
    distractor_samples = []
    if os.path.exists(distractor_path):
        with open(distractor_path, 'r', encoding='utf-8') as f:
            distractor_samples = json.load(f)
        logging.info(f"Loaded {len(distractor_samples)} distractor samples")
    else:
        logging.warning(f"Distractor data not found at {distractor_path}")
    
    # Load strategic CoT data
    cot_path = os.path.join(data_dir, "strategic_cot_training_data.json")
    cot_samples = []
    if os.path.exists(cot_path):
        with open(cot_path, 'r', encoding='utf-8') as f:
            cot_samples = json.load(f)
        logging.info(f"Loaded {len(cot_samples)} strategic CoT samples")
    else:
        logging.warning(f"Strategic CoT data not found at {cot_path}")
    
    return distractor_samples, cot_samples


def create_training_arguments(config: Dict, output_dir: str) -> TrainingArguments:
    """Create training arguments from config"""
    
    training_config = config["training"]
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_steps=training_config["warmup_steps"],
        max_grad_norm=training_config["max_grad_norm"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if config["logging"]["use_wandb"] else None,
        run_name=f"prismrag-{config['model']['base_model'].split('/')[-1]}",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        fp16=True,
        remove_unused_columns=False,
        seed=42
    )


def split_data(samples: List[Dict], train_ratio: float = 0.9) -> tuple[List[Dict], List[Dict]]:
    """Split data into train and validation sets"""
    
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    
    return samples[:split_idx], samples[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Train PrismRAG model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/prismrag",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training data
    distractor_samples, cot_samples = load_training_data(args.data_dir)
    
    if not distractor_samples and not cot_samples:
        logging.error("No training data found!")
        return
    
    # Initialize trainer
    logging.info("Initializing PrismRAG trainer...")
    trainer = PrismRAGTrainer(
        model_name=config["model"]["base_model"],
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        device="auto"
    )
    
    # Log model information
    model_info = trainer.get_model_size()
    logging.info(f"Model parameters: {model_info}")
    
    # Prepare training dataset
    logging.info("Preparing training dataset...")
    train_dataset = trainer.prepare_training_data(
        distractor_samples=distractor_samples,
        strategic_cot_samples=cot_samples,
        max_length=config["model"]["max_length"]
    )
    
    # Log dataset information
    sample_distribution = train_dataset.get_sample_type_distribution()
    logging.info(f"Training dataset distribution: {sample_distribution}")
    
    # Split into train and validation
    all_samples = train_dataset.processed_samples
    train_samples, val_samples = split_data(all_samples, train_ratio=0.9)
    
    # Create datasets
    train_dataset = PrismRAGDataset(
        samples=[{"instruction": s["instruction"], "response": s["response"], "sample_type": s["sample_type"]} 
                for s in train_samples],
        tokenizer=trainer.tokenizer,
        max_length=config["model"]["max_length"]
    )
    
    val_dataset = None
    if val_samples:
        val_dataset = PrismRAGDataset(
            samples=[{"instruction": s["instruction"], "response": s["response"], "sample_type": s["sample_type"]} 
                    for s in val_samples],
            tokenizer=trainer.tokenizer,
            max_length=config["model"]["max_length"]
        )
    
    logging.info(f"Training samples: {len(train_samples)}")
    logging.info(f"Validation samples: {len(val_samples) if val_samples else 0}")
    
    # Create training arguments
    training_args = create_training_arguments(config, args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = args.resume_from_checkpoint
        logging.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    
    # Start training
    logging.info("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args,
        use_wandb=config["logging"]["use_wandb"],
        wandb_project=config["logging"]["project_name"]
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    
    # Save training configuration
    config_save_path = os.path.join(args.output_dir, "training_config.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Save training summary
    training_summary = {
        "model_name": config["model"]["base_model"],
        "use_lora": args.use_lora,
        "total_training_samples": len(train_samples),
        "total_validation_samples": len(val_samples) if val_samples else 0,
        "sample_distribution": sample_distribution,
        "model_parameters": model_info,
        "training_args": {
            "num_epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps
        }
    }
    
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    
    logging.info("Training completed successfully!")
    logging.info(f"Model saved to: {final_model_path}")
    logging.info(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()