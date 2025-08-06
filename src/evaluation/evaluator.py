"""
PrismRAG Evaluator

This module implements comprehensive evaluation for PrismRAG models,
including factuality scoring, robustness testing, and benchmark evaluation.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from .metrics import FactualityMetrics, RAGMetrics
from .benchmarks import BenchmarkLoader


@dataclass
class EvaluationResult:
    """Data class for evaluation results"""
    benchmark: str
    accuracy: float
    hallucination_rate: float
    missing_rate: float
    factuality_score: float
    num_samples: int
    detailed_results: List[Dict]


class PrismRAGEvaluator:
    """
    Comprehensive evaluator for PrismRAG models.
    
    Supports evaluation on multiple RAG QA benchmarks with
    factuality scoring and robustness analysis.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_vllm: bool = False
    ):
        self.model_path = model_path
        self.device = device
        self.use_vllm = use_vllm
        
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize metrics and benchmark loader
        self.factuality_metrics = FactualityMetrics()
        self.rag_metrics = RAGMetrics()
        self.benchmark_loader = BenchmarkLoader()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        
        self.logger.info(f"Loading model from: {self.model_path}")
        
        if self.use_vllm:
            # Use vLLM for faster inference
            try:
                from vllm import LLM, SamplingParams
                self.model = LLM(
                    model=self.model_path,
                    dtype="float16",
                    trust_remote_code=True
                )
                self.sampling_params = SamplingParams(
                    temperature=1.0,
                    top_p=0.9,
                    max_tokens=1000
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except ImportError:
                self.logger.warning("vLLM not available, falling back to transformers")
                self.use_vllm = False
        
        if not self.use_vllm:
            # Use transformers
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        num_samples: Optional[int] = None,
        batch_size: int = 8
    ) -> EvaluationResult:
        """
        Evaluate on a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark to evaluate
            num_samples: Number of samples to evaluate (None for all)
            batch_size: Batch size for inference
            
        Returns:
            EvaluationResult with detailed metrics
        """
        
        self.logger.info(f"Evaluating on benchmark: {benchmark_name}")
        
        # Load benchmark data
        benchmark_data = self.benchmark_loader.load_benchmark(benchmark_name)
        
        if num_samples:
            benchmark_data = benchmark_data[:num_samples]
        
        # Generate responses
        responses = self._generate_responses(benchmark_data, batch_size)
        
        # Evaluate responses
        detailed_results = []
        for i, (sample, response) in enumerate(zip(benchmark_data, responses)):
            result = self._evaluate_single_response(
                question=sample["question"],
                references=sample["references"],
                ground_truth=sample["answer"],
                generated_response=response,
                sample_id=i
            )
            detailed_results.append(result)
        
        # Aggregate results
        accuracy = sum(1 for r in detailed_results if r["category"] == "accurate") / len(detailed_results)
        hallucination_rate = sum(1 for r in detailed_results if r["category"] == "hallucination") / len(detailed_results)
        missing_rate = sum(1 for r in detailed_results if r["category"] == "missing") / len(detailed_results)
        factuality_score = accuracy - hallucination_rate
        
        return EvaluationResult(
            benchmark=benchmark_name,
            accuracy=accuracy,
            hallucination_rate=hallucination_rate,
            missing_rate=missing_rate,
            factuality_score=factuality_score,
            num_samples=len(detailed_results),
            detailed_results=detailed_results
        )
    
    def evaluate_multiple_benchmarks(
        self,
        benchmark_names: List[str],
        num_samples_per_benchmark: Optional[int] = None,
        batch_size: int = 8
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate on multiple benchmarks.
        
        Args:
            benchmark_names: List of benchmark names
            num_samples_per_benchmark: Number of samples per benchmark
            batch_size: Batch size for inference
            
        Returns:
            Dictionary mapping benchmark names to results
        """
        
        results = {}
        
        for benchmark_name in benchmark_names:
            try:
                result = self.evaluate_benchmark(
                    benchmark_name=benchmark_name,
                    num_samples=num_samples_per_benchmark,
                    batch_size=batch_size
                )
                results[benchmark_name] = result
                
                self.logger.info(
                    f"{benchmark_name}: Factuality={result.factuality_score:.3f}, "
                    f"Accuracy={result.accuracy:.3f}, Hallucination={result.hallucination_rate:.3f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error evaluating {benchmark_name}: {e}")
                continue
        
        return results
    
    def evaluate_robustness(
        self,
        benchmark_name: str = "crag",
        reference_counts: List[int] = [1, 5, 10, 20, 50],
        num_samples: int = 100
    ) -> Dict[int, EvaluationResult]:
        """
        Evaluate robustness to varying numbers of reference documents.
        
        Args:
            benchmark_name: Benchmark to use for robustness testing
            reference_counts: List of reference document counts to test
            num_samples: Number of samples to evaluate per setting
            
        Returns:
            Dictionary mapping reference counts to results
        """
        
        self.logger.info(f"Evaluating robustness on {benchmark_name}")
        
        results = {}
        
        for ref_count in reference_counts:
            self.logger.info(f"Testing with {ref_count} references")
            
            # Load benchmark with specific reference count
            benchmark_data = self.benchmark_loader.load_benchmark(
                benchmark_name,
                max_references=ref_count
            )
            
            if num_samples:
                benchmark_data = benchmark_data[:num_samples]
            
            # Evaluate
            result = self.evaluate_benchmark(
                benchmark_name=f"{benchmark_name}_{ref_count}refs",
                num_samples=len(benchmark_data)
            )
            
            results[ref_count] = result
        
        return results
    
    def _generate_responses(
        self,
        samples: List[Dict],
        batch_size: int = 8
    ) -> List[str]:
        """Generate responses for a list of samples"""
        
        responses = []
        
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating responses"):
            batch = samples[i:i + batch_size]
            
            if self.use_vllm:
                batch_responses = self._generate_batch_vllm(batch)
            else:
                batch_responses = self._generate_batch_transformers(batch)
            
            responses.extend(batch_responses)
        
        return responses
    
    def _generate_batch_vllm(self, batch: List[Dict]) -> List[str]:
        """Generate responses using vLLM"""
        
        prompts = []
        for sample in batch:
            prompt = self._build_inference_prompt(
                question=sample["question"],
                references=sample["references"]
            )
            prompts.append(prompt)
        
        outputs = self.model.generate(prompts, self.sampling_params)
        
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            responses.append(response)
        
        return responses
    
    def _generate_batch_transformers(self, batch: List[Dict]) -> List[str]:
        """Generate responses using transformers"""
        
        responses = []
        
        for sample in batch:
            prompt = self._build_inference_prompt(
                question=sample["question"],
                references=sample["references"]
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3000
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=1.0,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            responses.append(response)
        
        return responses
    
    def _build_inference_prompt(
        self,
        question: str,
        references: List[str]
    ) -> str:
        """Build prompt for inference"""
        
        references_text = "\n\n".join([
            f"参考资料 {i+1}:\n{ref}" 
            for i, ref in enumerate(references)
        ])
        
        prompt = f"""对于此任务，您需要回答一个问题。请提供事实准确、直接和清晰的回复。不要问任何澄清问题或要求其他信息。为了确保考虑正确的事实，您应该始终将您的回答建立在下面提供的参考资料的基础上。如果参考资料与问题无关或没有提供正确的信息，您可以用道歉代替编造事实。

## 参考资料：
{references_text}

## 问题：
{question}

现在让我们逐步思考：

步骤 1：总结问题问的是什么，以及回答问题所需的具体关键信息是什么。

步骤 2：逐一分析提供的参考资料。确定可用于回答问题的相关信息。密切注意与问题相关的实体、名称、时间、地点、事件和关键词。

步骤 3：基于步骤 1 和步骤 2，您必须提供一个直接回答问题并完全基于所提供的参考资料的答案。

现在，用几句话提供您的推理步骤，然后在新的一行中提供问题的最终答案，以"## 答案:"开头。"""

        return prompt
    
    def _evaluate_single_response(
        self,
        question: str,
        references: List[str],
        ground_truth: str,
        generated_response: str,
        sample_id: int
    ) -> Dict:
        """Evaluate a single response"""
        
        # Extract final answer from response
        final_answer = self._extract_final_answer(generated_response)
        
        # Categorize response
        category = self.factuality_metrics.categorize_response(
            question=question,
            ground_truth=ground_truth,
            generated_response=final_answer,
            references=references
        )
        
        # Calculate additional metrics
        metrics = self.rag_metrics.calculate_metrics(
            ground_truth=ground_truth,
            generated_response=final_answer
        )
        
        return {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "generated_response": generated_response,
            "final_answer": final_answer,
            "category": category,
            "metrics": metrics
        }
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from a response"""
        
        # Look for "## 答案:" marker
        answer_marker = "## 答案:"
        if answer_marker in response:
            answer_part = response.split(answer_marker)[-1].strip()
            return answer_part
        
        # Fallback: return the last paragraph
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1]
        
        return response.strip()
    
    def save_results(
        self,
        results: Union[EvaluationResult, Dict[str, EvaluationResult]],
        filepath: str
    ):
        """Save evaluation results to file"""
        
        if isinstance(results, EvaluationResult):
            # Single result
            data = {
                "benchmark": results.benchmark,
                "accuracy": results.accuracy,
                "hallucination_rate": results.hallucination_rate,
                "missing_rate": results.missing_rate,
                "factuality_score": results.factuality_score,
                "num_samples": results.num_samples,
                "detailed_results": results.detailed_results
            }
        else:
            # Multiple results
            data = {}
            for benchmark_name, result in results.items():
                data[benchmark_name] = {
                    "accuracy": result.accuracy,
                    "hallucination_rate": result.hallucination_rate,
                    "missing_rate": result.missing_rate,
                    "factuality_score": result.factuality_score,
                    "num_samples": result.num_samples
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def print_summary(
        self,
        results: Dict[str, EvaluationResult]
    ):
        """Print a summary of evaluation results"""
        
        print("\n" + "="*80)
        print("PrismRAG Evaluation Summary")
        print("="*80)
        
        print(f"{'Benchmark':<15} {'Factuality':<12} {'Accuracy':<10} {'Hallucination':<13} {'Missing':<8} {'Samples':<8}")
        print("-"*80)
        
        total_factuality = 0
        total_samples = 0
        
        for benchmark_name, result in results.items():
            print(f"{benchmark_name:<15} {result.factuality_score:<12.3f} {result.accuracy:<10.3f} "
                  f"{result.hallucination_rate:<13.3f} {result.missing_rate:<8.3f} {result.num_samples:<8}")
            
            total_factuality += result.factuality_score * result.num_samples
            total_samples += result.num_samples
        
        print("-"*80)
        avg_factuality = total_factuality / total_samples if total_samples > 0 else 0
        print(f"{'Average':<15} {avg_factuality:<12.3f} {'':<10} {'':<13} {'':<8} {total_samples:<8}")
        print("="*80)