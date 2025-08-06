"""
Strategic Chain-of-Thought Generator for PrismRAG

This module implements the strategic CoT generation mechanism that creates
dynamic reasoning strategies and detailed chain-of-thought processes.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class StrategicCoTSample:
    """Data class for strategic CoT samples"""
    question: str
    references: List[str]
    strategy: str
    reasoning: str
    answer: str
    user_context: Optional[str] = None


class StrategicCoTGenerator:
    """
    Generates strategic chain-of-thought reasoning for training data.
    
    Based on the PrismRAG paper's approach of first generating a high-level
    strategy outline, then following that strategy to produce detailed CoT
    reasoning and final answers.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70b-instruct",
        device: str = "auto",
        max_iterations: int = 10,
        random_attempts: int = 6,
        quality_threshold: int = 4
    ):
        self.model_name = model_name
        self.device = device
        self.max_iterations = max_iterations
        self.random_attempts = random_attempts
        self.quality_threshold = quality_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_strategic_cot(
        self,
        question: str,
        references: List[str],
        ground_truth_answer: str,
        user_context: Optional[str] = None
    ) -> Optional[StrategicCoTSample]:
        """
        Generate a strategic CoT sample for the given question and references.
        
        Args:
            question: The question to answer
            references: List of reference documents
            ground_truth_answer: The correct answer for evaluation
            user_context: Additional user context (time, location, etc.)
            
        Returns:
            StrategicCoTSample if successful, None otherwise
        """
        
        for iteration in range(self.max_iterations):
            try:
                # Generate strategy and reasoning
                if iteration < self.random_attempts:
                    # Use random generation for first attempts
                    cot_data = self._generate_cot_random(
                        question=question,
                        references=references,
                        user_context=user_context
                    )
                else:
                    # Use critique-based revision for later attempts
                    cot_data = self._generate_cot_with_critique(
                        question=question,
                        references=references,
                        user_context=user_context,
                        previous_attempt=getattr(self, '_last_attempt', None),
                        critique=getattr(self, '_last_critique', None)
                    )
                
                if not cot_data:
                    continue
                
                # Evaluate reasoning quality
                reasoning_score = self._evaluate_reasoning(
                    question=question,
                    references=references,
                    strategy=cot_data["strategy"],
                    reasoning=cot_data["reasoning"],
                    answer=cot_data["answer"]
                )
                
                # Evaluate answer quality
                answer_score = self._evaluate_answer(
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    candidate_answer=cot_data["answer"]
                )
                
                # Check if both scores meet threshold
                if reasoning_score >= self.quality_threshold and answer_score >= self.quality_threshold:
                    return StrategicCoTSample(
                        question=question,
                        references=references,
                        strategy=cot_data["strategy"],
                        reasoning=cot_data["reasoning"],
                        answer=cot_data["answer"],
                        user_context=user_context
                    )
                
                # Store for potential critique-based revision
                self._last_attempt = cot_data
                self._last_critique = f"Reasoning score: {reasoning_score}, Answer score: {answer_score}"
                
            except Exception as e:
                self.logger.warning(f"Error in iteration {iteration}: {e}")
                continue
        
        self.logger.warning(f"Failed to generate quality CoT after {self.max_iterations} iterations")
        return None
    
    def _generate_cot_random(
        self,
        question: str,
        references: List[str],
        user_context: Optional[str] = None
    ) -> Optional[Dict]:
        """Generate CoT using random sampling"""
        
        prompt = self._build_strategic_cot_prompt(
            question=question,
            references=references,
            user_context=user_context
        )
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    temperature=1.0,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return self._parse_strategic_cot_response(response)
            
        except Exception as e:
            self.logger.error(f"Error generating random CoT: {e}")
            return None
    
    def _generate_cot_with_critique(
        self,
        question: str,
        references: List[str],
        user_context: Optional[str],
        previous_attempt: Optional[Dict],
        critique: Optional[str]
    ) -> Optional[Dict]:
        """Generate CoT using critique-based revision"""
        
        if not previous_attempt or not critique:
            return self._generate_cot_random(question, references, user_context)
        
        prompt = self._build_critique_revision_prompt(
            question=question,
            references=references,
            user_context=user_context,
            previous_reasoning=previous_attempt.get("reasoning", ""),
            previous_answer=previous_attempt.get("answer", ""),
            critique=critique
        )
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return self._parse_strategic_cot_response(response)
            
        except Exception as e:
            self.logger.error(f"Error generating critique-based CoT: {e}")
            return None
    
    def _build_strategic_cot_prompt(
        self,
        question: str,
        references: List[str],
        user_context: Optional[str] = None
    ) -> str:
        """Build prompt for strategic CoT generation"""
        
        references_text = "\n\n".join([f"参考资料 {i+1}:\n{ref}" for i, ref in enumerate(references)])
        
        prompt = f"""对于此任务，您需要回答一个问题。请提供事实准确、直接和清晰的回复。请勿询问任何澄清问题或索取其他信息。为了确保考虑正确的事实，您应该始终将您的回答建立在下面提供的参考文献的基础上。如果参考文献与问题无关或未提供正确的信息，您可以回复道歉而不是编造事实。

## 参考资料：
{references_text}

## 问题：
{question}"""

        if user_context:
            prompt += f"\n\n## 用户上下文：\n{user_context}"

        prompt += """

在回答问题之前，请退一步仔细思考回答问题的最佳策略。为您可以采取的推理步骤生成一个大纲，以找到最佳答案。然后，使用大纲逐步思考。

使用以下模板来策略推理步骤，逐步推理，并提供最终答案：

## 策略：
- 步骤1：***第1步的指令***
- 步骤2：***第2步的指令***
...
- 步骤n：***步骤N的指令***

## 推理：
- 步骤1：***对应于策略中的步骤1的推理***
- 步骤2：***对应于策略中的步骤2的推理***
...
- 步骤n：***对应于策略中的步骤N的推理***

## 答案：
问题的最终答案"""

        return prompt
    
    def _build_critique_revision_prompt(
        self,
        question: str,
        references: List[str],
        user_context: Optional[str],
        previous_reasoning: str,
        previous_answer: str,
        critique: str
    ) -> str:
        """Build prompt for critique-based revision"""
        
        references_text = "\n\n".join([f"参考资料 {i+1}:\n{ref}" for i, ref in enumerate(references)])
        
        prompt = f"""您的推理和答案可以进一步改进。以下是您之前推理和答案的批评。您应该使用此批评来改进您的推理和答案。

## 批评：
{critique}

## 参考资料：
{references_text}

## 问题：
{question}"""

        if user_context:
            prompt += f"\n\n## 用户上下文：\n{user_context}"

        prompt += f"""

## 之前的推理：
{previous_reasoning}

## 之前的答案：
{previous_answer}

现在让我们一步一步地思考，同时考虑上面提供的批评：

## 策略：
- 步骤1：***改进后的第1步指令***
- 步骤2：***改进后的第2步指令***
...

## 推理：
- 步骤1：***改进后的推理步骤1***
- 步骤2：***改进后的推理步骤2***
...

## 答案：
改进后的最终答案"""

        return prompt
    
    def _evaluate_reasoning(
        self,
        question: str,
        references: List[str],
        strategy: str,
        reasoning: str,
        answer: str
    ) -> int:
        """Evaluate the quality of reasoning"""
        
        references_text = "\n\n".join([f"参考资料 {i+1}:\n{ref}" for i, ref in enumerate(references)])
        
        prompt = f"""对于此任务，您将获得一个问题、一组参考资料以及一个推理和一个问题的答案。您的任务是评估推理过程。

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
            
            # Extract score from response
            score_line = [line for line in response.split('\n') if '分数：' in line or 'Score:' in line]
            if score_line:
                score_text = score_line[0].split('：')[-1].split(':')[-1].strip()
                try:
                    return int(score_text)
                except ValueError:
                    pass
            
            return 1  # Default low score if parsing fails
            
        except Exception as e:
            self.logger.error(f"Error evaluating reasoning: {e}")
            return 1
    
    def _evaluate_answer(
        self,
        question: str,
        ground_truth_answer: str,
        candidate_answer: str
    ) -> int:
        """Evaluate the quality of the answer"""
        
        prompt = f"""对于此任务，您将获得一个问题，一个参考答案和一个候选答案。您的任务是评估候选答案是否完全回答了问题，同时与参考答案中提供的信息一致。

使用以下准则来评估推理过程，并为候选答案分配1到4分的评分：

- 分数1：候选答案未提供任何信息来回答问题。它是问题的简单重复，拒绝回答，或提供不相关的信息。
- 分数2：候选答案与参考答案不一致，并且存在主要的矛盾点。
- 分数3：候选答案部分回答了问题，并且与参考答案一致。它可能有一些小的矛盾点，但大部分是一致的。
- 分数4：候选答案完全回答了问题，并且与参考答案一致。

## 问题：
{question}

## 参考答案：
{ground_truth_answer}

## 候选答案：
{candidate_answer}

首先，根据上述标准，用几句话解释您对候选答案的评估，然后在以"## 分数："开头的新行中，后跟分数。"""

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
            
            # Extract score from response
            score_line = [line for line in response.split('\n') if '分数：' in line or 'Score:' in line]
            if score_line:
                score_text = score_line[0].split('：')[-1].split(':')[-1].strip()
                try:
                    return int(score_text)
                except ValueError:
                    pass
            
            return 1  # Default low score if parsing fails
            
        except Exception as e:
            self.logger.error(f"Error evaluating answer: {e}")
            return 1
    
    def _parse_strategic_cot_response(self, response: str) -> Optional[Dict]:
        """Parse strategic CoT response"""
        try:
            # Extract strategy section
            strategy_start = response.find("## 策略：")
            reasoning_start = response.find("## 推理：")
            answer_start = response.find("## 答案：")
            
            if strategy_start == -1 or reasoning_start == -1 or answer_start == -1:
                return None
            
            strategy = response[strategy_start + len("## 策略："):reasoning_start].strip()
            reasoning = response[reasoning_start + len("## 推理："):answer_start].strip()
            answer = response[answer_start + len("## 答案："):].strip()
            
            return {
                "strategy": strategy,
                "reasoning": reasoning,
                "answer": answer
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse strategic CoT response: {e}")
            return None
    
    def generate_batch(
        self,
        samples: List[Tuple[str, List[str], str]],
        user_context: Optional[str] = None
    ) -> List[StrategicCoTSample]:
        """Generate strategic CoT for a batch of samples"""
        
        results = []
        for question, references, ground_truth in samples:
            cot_sample = self.generate_strategic_cot(
                question=question,
                references=references,
                ground_truth_answer=ground_truth,
                user_context=user_context
            )
            if cot_sample:
                results.append(cot_sample)
        
        return results