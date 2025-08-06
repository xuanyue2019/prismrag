"""
Evaluators for data generation quality assessment
"""

import json
import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class DistractorEvaluator:
    """
    Evaluator for distractor quality assessment.
    
    Assesses the quality of generated distractors based on relevance,
    distraction effectiveness, and format similarity.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70b-instruct",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate_distractor(
        self,
        question: str,
        answer: str,
        golden_passage: str,
        distractor_passage: str,
        open_ended_question: str,
        distractor_answer: str,
        user_time: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict:
        """
        Evaluate the quality of a generated distractor.
        
        Args:
            question: Original question
            answer: Ground truth answer
            golden_passage: Original passage
            distractor_passage: Generated distractor passage
            open_ended_question: Modified open-ended question
            distractor_answer: Answer from distractor passage
            user_time: User time context
            location: User location context
            
        Returns:
            Dictionary with evaluation scores and feedback
        """
        
        prompt = self._build_evaluation_prompt(
            question=question,
            answer=answer,
            golden_passage=golden_passage,
            distractor_passage=distractor_passage,
            open_ended_question=open_ended_question,
            distractor_answer=distractor_answer,
            user_time=user_time,
            location=location
        )
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            evaluation = self._parse_evaluation_response(response)
            
            return evaluation or {
                "relevance_score": 1,
                "distraction_score": 1,
                "format_score": 1,
                "overall_score": 1,
                "feedback": "Failed to parse evaluation"
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating distractor: {e}")
            return {
                "relevance_score": 1,
                "distraction_score": 1,
                "format_score": 1,
                "overall_score": 1,
                "feedback": f"Evaluation error: {e}"
            }
    
    def _build_evaluation_prompt(
        self,
        question: str,
        answer: str,
        golden_passage: str,
        distractor_passage: str,
        open_ended_question: str,
        distractor_answer: str,
        user_time: Optional[str] = None,
        location: Optional[str] = None
    ) -> str:
        """Build evaluation prompt"""
        
        return f"""你是一个在语言学方面具有专业知识的智能助手。始终遵循提供的指令，并以有效的json格式生成输出，不包含任何额外信息。

### 用户
给定一个问题、答案、文章、位置、用户时间、干扰文章和干扰文章的答案：

在1到5的范围内打分：

1. 相关性分数：衡量干扰文章与给定问题、答案、文章、位置、用户时间的相关程度。
2. 分心程度：评估文章在提供相关检索噪声方面的质量。
3. 格式：评估干扰文章与原始文章在文本长度和格式上的相似性。

以Json格式输出，包含以下字段：'relevance-score'、'distraction-score'、'format-score'、'thought-process'。

## 问题：{question}
## 答案：{answer}
## 用户时间：{user_time or "未提供"}
## 地点：{location or "未提供"}
## 段落：{golden_passage}
## 开放式问题：{open_ended_question}
## 干扰段落：{distractor_passage}
## 干扰段落的答案：{distractor_answer}

### 助手："""
    
    def _parse_evaluation_response(self, response: str) -> Optional[Dict]:
        """Parse evaluation response"""
        
        try:
            # Find JSON content
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = response[start_idx:end_idx]
            evaluation = json.loads(json_str)
            
            # Calculate overall score
            scores = [
                evaluation.get("relevance-score", 1),
                evaluation.get("distraction-score", 1),
                evaluation.get("format-score", 1)
            ]
            evaluation["overall_score"] = sum(scores) / len(scores)
            evaluation["feedback"] = evaluation.get("thought-process", "")
            
            return evaluation
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse evaluation response: {e}")
            return None


class CoTEvaluator:
    """
    Evaluator for Chain-of-Thought quality assessment.
    
    Assesses the quality of generated reasoning chains and answers.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70b-instruct",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate_reasoning(
        self,
        question: str,
        references: List[str],
        strategy: str,
        reasoning: str,
        answer: str
    ) -> int:
        """
        Evaluate the quality of reasoning.
        
        Args:
            question: The question being answered
            references: Reference documents
            strategy: Generated strategy
            reasoning: Generated reasoning chain
            answer: Generated answer
            
        Returns:
            Score from 1-4
        """
        
        prompt = self._build_reasoning_evaluation_prompt(
            question=question,
            references=references,
            strategy=strategy,
            reasoning=reasoning,
            answer=answer
        )
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Extract score
            import re
            score_match = re.search(r'分数：\s*(\d+)', response)
            if score_match:
                return int(score_match.group(1))
            
            # Fallback
            numbers = re.findall(r'\b([1-4])\b', response)
            if numbers:
                return int(numbers[-1])
            
            return 1  # Default low score
            
        except Exception as e:
            self.logger.error(f"Error evaluating reasoning: {e}")
            return 1
    
    def evaluate_answer(
        self,
        question: str,
        ground_truth: str,
        candidate_answer: str
    ) -> int:
        """
        Evaluate the quality of an answer.
        
        Args:
            question: The question
            ground_truth: Ground truth answer
            candidate_answer: Generated answer
            
        Returns:
            Score from 1-4
        """
        
        prompt = self._build_answer_evaluation_prompt(
            question=question,
            ground_truth=ground_truth,
            candidate_answer=candidate_answer
        )
        
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
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Extract score
            import re
            score_match = re.search(r'分数：\s*(\d+)', response)
            if score_match:
                return int(score_match.group(1))
            
            # Fallback
            numbers = re.findall(r'\b([1-4])\b', response)
            if numbers:
                return int(numbers[-1])
            
            return 1  # Default low score
            
        except Exception as e:
            self.logger.error(f"Error evaluating answer: {e}")
            return 1
    
    def _build_reasoning_evaluation_prompt(
        self,
        question: str,
        references: List[str],
        strategy: str,
        reasoning: str,
        answer: str
    ) -> str:
        """Build reasoning evaluation prompt"""
        
        references_text = "\n\n".join([f"参考资料 {i+1}:\n{ref}" for i, ref in enumerate(references)])
        
        return f"""对于此任务，您将获得一个问题、一组参考资料以及一个推理和一个问题的答案。您的任务是评估推理过程。

使用以下指南来评估推理过程，并为所提供的推理分配1到4之间的分数：

- 分数1：推理不正确。它与问题和参考资料无关，或者它只是对它们的简单重复，没有任何思考过程。
- 分数2：推理与问题和参考资料相关，但对回答问题并没有真正帮助。尚不清楚推理如何引出答案。
- 分数3：推理与问题和参考资料相关，并且有助于回答问题。它可能部分有助于回答问题，但在推理中存在差距，并且问题的某些部分未得到解决。
- 分数4：推理与问题和参考资料相关，并且有助于回答问题。它提供了一个清晰而完整的思考过程，可以引出答案。

## 参考资料：
{references_text}

## 问题：
{question}

## 策略：
{strategy}

## 推理：
{reasoning}

## 答案：
{answer}

首先，根据上述标准，用几句话解释您对推理的评估，然后在以"## 分数："开头的新行中提供最终分数，后跟分数的值。"""
    
    def _build_answer_evaluation_prompt(
        self,
        question: str,
        ground_truth: str,
        candidate_answer: str
    ) -> str:
        """Build answer evaluation prompt"""
        
        return f"""对于此任务，您将获得一个问题，一个参考答案和一个候选答案。您的任务是评估候选答案是否完全回答了问题，同时与参考答案中提供的信息一致。

使用以下准则来评估推理过程，并为候选答案分配1到4分的评分：

- 分数1：候选答案未提供任何信息来回答问题。它是问题的简单重复，拒绝回答，或提供不相关的信息。
- 分数2：候选答案与参考答案不一致，并且存在主要的矛盾点。
- 分数3：候选答案部分回答了问题，并且与参考答案一致。它可能有一些小的矛盾点，但大部分是一致的。
- 分数4：候选答案完全回答了问题，并且与参考答案一致。

## 问题：
{question}

## 参考答案：
{ground_truth}

## 候选答案：
{candidate_answer}

首先，根据上述标准，用几句话解释您对候选答案的评估，然后在以"## 分数："开头的新行中，后跟分数。"""