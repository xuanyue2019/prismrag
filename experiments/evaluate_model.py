#!/usr/bin/env python3
"""
Evaluate PrismRAG model

This script evaluates the trained PrismRAG model on various RAG QA benchmarks
and generates comprehensive evaluation reports.
"""

import argparse
import json
import logging
import os
from typing import Dict, List

import yaml

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import PrismRAGEvaluator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PrismRAG model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained PrismRAG model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="List of benchmarks to evaluate (default: all from config)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate per benchmark (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for faster inference"
    )
    parser.add_argument(
        "--evaluate-robustness",
        action="store_true",
        help="Evaluate robustness to varying reference counts"
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
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine benchmarks to evaluate
    benchmarks = args.benchmarks or config["evaluation"]["benchmarks"]
    
    logging.info(f"Evaluating model: {args.model_path}")
    logging.info(f"Benchmarks: {benchmarks}")
    logging.info(f"Samples per benchmark: {args.num_samples or 'all'}")
    
    # Initialize evaluator
    evaluator = PrismRAGEvaluator(
        model_path=args.model_path,
        device="auto",
        use_vllm=args.use_vllm
    )
    
    # Standard benchmark evaluation
    logging.info("Starting benchmark evaluation...")
    results = evaluator.evaluate_multiple_benchmarks(
        benchmark_names=benchmarks,
        num_samples_per_benchmark=args.num_samples,
        batch_size=args.batch_size
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    evaluator.save_results(results, results_path)
    
    # Robustness evaluation
    if args.evaluate_robustness:
        logging.info("Starting robustness evaluation...")
        
        robustness_results = evaluator.evaluate_robustness(
            benchmark_name="crag",
            reference_counts=[1, 5, 10, 20, 50],
            num_samples=args.num_samples or 100
        )
        
        # Print robustness summary
        print("\n" + "="*80)
        print("Robustness Evaluation (CRAG)")
        print("="*80)
        print(f"{'Refs':<6} {'Factuality':<12} {'Accuracy':<10} {'Hallucination':<13} {'Missing':<8}")
        print("-"*50)
        
        for ref_count, result in robustness_results.items():
            print(f"{ref_count:<6} {result.factuality_score:<12.3f} {result.accuracy:<10.3f} "
                  f"{result.hallucination_rate:<13.3f} {result.missing_rate:<8.3f}")
        
        # Save robustness results
        robustness_path = os.path.join(args.output_dir, "robustness_results.json")
        evaluator.save_results(robustness_results, robustness_path)
    
    # Generate evaluation report
    report = {
        "model_path": args.model_path,
        "evaluation_config": {
            "benchmarks": benchmarks,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "use_vllm": args.use_vllm
        },
        "benchmark_results": {
            name: {
                "factuality_score": result.factuality_score,
                "accuracy": result.accuracy,
                "hallucination_rate": result.hallucination_rate,
                "missing_rate": result.missing_rate,
                "num_samples": result.num_samples
            }
            for name, result in results.items()
        }
    }
    
    if args.evaluate_robustness:
        report["robustness_results"] = {
            str(ref_count): {
                "factuality_score": result.factuality_score,
                "accuracy": result.accuracy,
                "hallucination_rate": result.hallucination_rate,
                "missing_rate": result.missing_rate,
                "num_samples": result.num_samples
            }
            for ref_count, result in robustness_results.items()
        }
    
    # Calculate overall metrics
    if results:
        total_factuality = sum(r.factuality_score * r.num_samples for r in results.values())
        total_samples = sum(r.num_samples for r in results.values())
        avg_factuality = total_factuality / total_samples if total_samples > 0 else 0
        
        report["overall_metrics"] = {
            "average_factuality_score": avg_factuality,
            "total_samples_evaluated": total_samples,
            "num_benchmarks": len(results)
        }
    
    # Save evaluation report
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Evaluation completed!")
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info(f"Evaluation report: {report_path}")
    
    # Print final summary
    if results:
        print(f"\nFinal Results:")
        print(f"Average Factuality Score: {avg_factuality:.3f}")
        print(f"Total Samples Evaluated: {total_samples}")
        print(f"Benchmarks Evaluated: {len(results)}")


if __name__ == "__main__":
    main()