"""
Evaluation metrics for PrismRAG
"""

import logging
import re
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from bert_score import score as bert_score


class FactualityMetrics:
    """
    Metrics for evaluating factuality of RAG responses.
    
    Implements the factuality scoring approach from the PrismRAG paper,
    categorizing responses as accurate, hallucination, or missing.
    """
    
    def __init__(
        self,
        evaluator_model: str = "meta-llama/Llama-3.1-70b-instruct",
        device: str = "auto"
    ):
        self.evaluator_model = evaluator_model
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Load evaluator model for factuality assessment
        self.tokenizer = AutoTokenizer.from_pretrained(evaluator_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            evaluator_model,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def categorize_response(
        self,
        question: str,
        ground_truth: str,
        generated_response: str,
        references: List[str]
    ) -> str:
        """
        Categorize a response as accurate, hallucination, or missing.
        
        Args:
            question: The original question
            ground_truth: The ground truth answer
            generated_response: The model's response
            references: Reference documents used
            
        Returns:
            Category: "accurate", "hallucination", or "missing"
        """
        
        # Check if response is missing/refusal
        if self._is_missing_response(generated_response):
            return "missing"
        
        # Use LLM evaluator to assess factuality
        factuality_score = self._evaluate_factuality_with_llm(
            question=question,
            ground_truth=ground_truth,
            generated_response=generated_response,
            references=references
        )
        
        # Categorize based on score
        if factuality_score >= 3.5:
            return "accurate"
        elif factuality_score <= 2.0:
            return "hallucination"
        else:
            # Borderline cases - use additional heuristics
            return self._resolve_borderline_case(
                question, ground_truth, generated_response, references
            )
    
    def _is_missing_response(self, response: str) -> bool:
        """Check if response is missing or a refusal"""
        
        response_lower = response.lower().strip()
        
        # Common refusal patterns
        refusal_patterns = [
            "我不知道",
            "不知道",
            "无法回答",
            "没有足够的信息",
            "抱歉",
            "sorry",
            "i don't know",
            "cannot answer",
            "insufficient information"
        ]
        
        # Check if response is too short
        if len(response.strip()) < 10:
            return True
        
        # Check for refusal patterns
        for pattern in refusal_patterns:
            if pattern in response_lower:
                return True
        
        return False
    
    def _evaluate_factuality_with_llm(
        self,
        question: str,
        ground_truth: str,
        generated_response: str,
        references: List[str]
    ) -> float:
        """Use LLM to evaluate factuality"""
        
        references_text = "\n\n".join([
            f"参考资料 {i+1}:\n{ref}" 
            for i, ref in enumerate(references)
        ])
        
        prompt = f"""您需要评估一个问答系统生成的答案的事实准确性。请根据提供的参考资料和标准答案，对生成的答案进行评分。

评分标准（1-4分）：
- 4分：答案完全准确，与标准答案一致，基于参考资料
- 3分：答案大部分准确，与标准答案基本一致
- 2分：答案部分准确，但包含一些错误信息
- 1分：答案包含重大错误或误导性信息

## 参考资料：
{references_text}

## 问题：
{question}

## 标准答案：
{ground_truth}

## 生成的答案：
{generated_response}

请先分析答案的准确性，然后给出1-4的评分。

## 分析：
[您的分析]

## 评分：
[1-4的数字]"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Extract score
            score_match = re.search(r'评分：\s*(\d+)', response)
            if score_match:
                return float(score_match.group(1))
            
            # Fallback: look for numbers in the response
            numbers = re.findall(r'\b([1-4])\b', response)
            if numbers:
                return float(numbers[-1])  # Take the last number found
            
            return 2.0  # Default neutral score
            
        except Exception as e:
            self.logger.error(f"Error in LLM factuality evaluation: {e}")
            return 2.0
    
    def _resolve_borderline_case(
        self,
        question: str,
        ground_truth: str,
        generated_response: str,
        references: List[str]
    ) -> str:
        """Resolve borderline cases using additional heuristics"""
        
        # Simple heuristic: check for obvious hallucination indicators
        response_lower = generated_response.lower()
        
        # Check for uncertainty expressions (lean towards accurate)
        uncertainty_patterns = ["可能", "也许", "大概", "似乎", "probably", "might", "possibly"]
        if any(pattern in response_lower for pattern in uncertainty_patterns):
            return "accurate"
        
        # Check for definitive false statements (lean towards hallucination)
        false_indicators = ["绝对", "肯定", "一定", "definitely", "certainly", "absolutely"]
        if any(pattern in response_lower for pattern in false_indicators):
            # Additional check needed - this is just a heuristic
            pass
        
        # Default to accurate for borderline cases
        return "accurate"
    
    def calculate_factuality_score(
        self,
        categories: List[str]
    ) -> Dict[str, float]:
        """
        Calculate factuality metrics from categorized responses.
        
        Args:
            categories: List of response categories
            
        Returns:
            Dictionary with accuracy, hallucination rate, missing rate, and factuality score
        """
        
        total = len(categories)
        if total == 0:
            return {
                "accuracy": 0.0,
                "hallucination_rate": 0.0,
                "missing_rate": 0.0,
                "factuality_score": 0.0
            }
        
        accurate_count = categories.count("accurate")
        hallucination_count = categories.count("hallucination")
        missing_count = categories.count("missing")
        
        accuracy = accurate_count / total
        hallucination_rate = hallucination_count / total
        missing_rate = missing_count / total
        factuality_score = accuracy - hallucination_rate
        
        return {
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "missing_rate": missing_rate,
            "factuality_score": factuality_score
        }


class RAGMetrics:
    """
    Additional metrics for RAG evaluation.
    
    Includes ROUGE, BERT-Score, and other text similarity metrics.
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(
        self,
        ground_truth: str,
        generated_response: str
    ) -> Dict[str, float]:
        """
        Calculate various text similarity metrics.
        
        Args:
            ground_truth: Reference answer
            generated_response: Generated answer
            
        Returns:
            Dictionary with various metrics
        """
        
        metrics = {}
        
        # ROUGE scores
        try:
            rouge_scores = self.rouge_scorer.score(ground_truth, generated_response)
            metrics.update({
                "rouge1_f": rouge_scores['rouge1'].fmeasure,
                "rouge2_f": rouge_scores['rouge2'].fmeasure,
                "rougeL_f": rouge_scores['rougeL'].fmeasure
            })
        except Exception as e:
            self.logger.warning(f"Error calculating ROUGE scores: {e}")
            metrics.update({
                "rouge1_f": 0.0,
                "rouge2_f": 0.0,
                "rougeL_f": 0.0
            })
        
        # BERT Score
        try:
            P, R, F1 = bert_score([generated_response], [ground_truth], lang="zh")
            metrics["bert_score_f1"] = F1.item()
        except Exception as e:
            self.logger.warning(f"Error calculating BERT score: {e}")
            metrics["bert_score_f1"] = 0.0
        
        # Length-based metrics
        metrics.update({
            "response_length": len(generated_response.split()),
            "length_ratio": len(generated_response.split()) / max(len(ground_truth.split()), 1)
        })
        
        return metrics
    
    def calculate_aggregate_metrics(
        self,
        individual_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across multiple samples"""
        
        if not individual_metrics:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in individual_metrics:
            all_keys.update(metrics.keys())
        
        # Calculate averages
        aggregate = {}
        for key in all_keys:
            values = [m.get(key, 0.0) for m in individual_metrics]
            aggregate[f"avg_{key}"] = sum(values) / len(values)
        
        return aggregate