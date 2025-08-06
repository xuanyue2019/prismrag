"""
Distractor Generator for PrismRAG

This module implements the distractor generation mechanism that creates
synthetic distractors by modifying named entities, locations, and temporal
information in golden passages.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DistractorSample:
    """Data class for distractor samples"""
    question: str
    answer: str
    golden_passage: str
    open_ended_question: str
    distractor_passage: str
    distractor_answer: str
    user_time: Optional[str] = None
    location: Optional[str] = None


class DistractorGenerator:
    """
    Generates synthetic distractors for training data.
    
    Based on the PrismRAG paper's approach of systematically modifying
    named entities, locations, and temporal information to create
    confusing but grammatically coherent distractor passages.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70b-instruct",
        device: str = "auto",
        max_iterations: int = 5,
        quality_threshold: int = 4
    ):
        self.model_name = model_name
        self.device = device
        self.max_iterations = max_iterations
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
    
    def generate_distractor(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str] = None,
        location: Optional[str] = None,
        prior_distractor: Optional[str] = None,
        prior_rejection_reason: Optional[str] = None
    ) -> Optional[DistractorSample]:
        """
        Generate a distractor sample for the given question-answer-passage triplet.
        
        Args:
            question: Original question
            answer: Ground truth answer
            passage: Golden passage containing the answer
            user_time: User's current time context
            location: User's location context
            prior_distractor: Previously generated distractor (for iteration)
            prior_rejection_reason: Reason for rejecting prior distractor
            
        Returns:
            DistractorSample if successful, None otherwise
        """
        
        for iteration in range(self.max_iterations):
            try:
                # Generate distractor using LLM
                distractor_data = self._generate_distractor_with_llm(
                    question=question,
                    answer=answer,
                    passage=passage,
                    user_time=user_time,
                    location=location,
                    prior_distractor=prior_distractor,
                    prior_rejection_reason=prior_rejection_reason
                )
                
                if not distractor_data:
                    continue
                
                # Evaluate the generated distractor
                evaluation = self._evaluate_distractor(
                    question=question,
                    answer=answer,
                    passage=passage,
                    user_time=user_time,
                    location=location,
                    open_ended_question=distractor_data["open_ended_question"],
                    distractor_passage=distractor_data["distractor_passage"],
                    distractor_answer=distractor_data.get("distractor_answer", "")
                )
                
                # Check if quality meets threshold
                if evaluation["overall_score"] >= self.quality_threshold:
                    return DistractorSample(
                        question=question,
                        answer=answer,
                        golden_passage=passage,
                        open_ended_question=distractor_data["open_ended_question"],
                        distractor_passage=distractor_data["distractor_passage"],
                        distractor_answer=distractor_data.get("distractor_answer", ""),
                        user_time=user_time,
                        location=location
                    )
                
                # Use evaluation feedback for next iteration
                prior_distractor = distractor_data["distractor_passage"]
                prior_rejection_reason = evaluation["feedback"]
                
            except Exception as e:
                self.logger.warning(f"Error in iteration {iteration}: {e}")
                continue
        
        self.logger.warning(f"Failed to generate quality distractor after {self.max_iterations} iterations")
        return None
    
    def _generate_distractor_with_llm(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str] = None,
        location: Optional[str] = None,
        prior_distractor: Optional[str] = None,
        prior_rejection_reason: Optional[str] = None
    ) -> Optional[Dict]:
        """Generate distractor using LLM with structured prompt"""
        
        prompt = self._build_distractor_generation_prompt(
            question=question,
            answer=answer,
            passage=passage,
            user_time=user_time,
            location=location,
            prior_distractor=prior_distractor,
            prior_rejection_reason=prior_rejection_reason
        )
        
        try:
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
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
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Parse JSON response
            return self._parse_json_response(response)
            
        except Exception as e:
            self.logger.error(f"Error generating distractor: {e}")
            return None
    
    def _build_distractor_generation_prompt(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str] = None,
        location: Optional[str] = None,
        prior_distractor: Optional[str] = None,
        prior_rejection_reason: Optional[str] = None
    ) -> str:
        """Build the prompt for distractor generation"""
        
        prompt = """您是一位在语言学方面有专业知识的智能助手。始终遵循提供的指令，并以有效的json格式生成输出，不包含任何额外信息。

### 用户：
逐步思考：

1. 给定一个问题、答案、段落、地点和用户时间，根据问题和答案识别段落中的相关命名实体、日期时间、地点，这样修改相关命名实体将导致一个新的段落，如果用户没有足够的上下文，可能会引起混淆。

2. 通过修改提供的问题来生成一个开放式问题"open-ended-question"，这样提供的答案可以回答新的开放式问题。

3. 现在使用问题、答案、用户时间、地点，修改段落并通过修改命名实体生成新的段落，这样：
   a. 新的段落与现有段落相关。
   b. 新的段落语法连贯。
   c. 对于"开放式问题"，提供的和您生成的段落都是相关的。
   d. 干扰段落应与原始段落具有相似的字符数和相似的格式。

4. 评分您对生成的段落将满足条件3的信心（从1到5）。

## 要求：
- 您必须根据用户问题、位置、答案、用户时间生成段落。
- 生成的干扰应与原始段落长度相似，并具有相似的特殊字符，例如\\n, \\t。不要减少单词总数。
- 逐步思考并在"thought-steps"字段中提供详细解释。
- 以Json格式输出，包含以下字段：'open-ended-question', 'thought-steps', 'distracting-named-entities', 'distractor-passage', 'score', 'reason'

## 问题：{question}
## 答案：{answer}
## 用户时间：{user_time}
## 地点：{location}
## 段落：{passage}"""

        if prior_distractor and prior_rejection_reason:
            prompt += f"""
## 先前的干扰段落：{prior_distractor}
## 先前拒绝干扰的原因：{prior_rejection_reason}"""

        prompt += "\n\n### 助手："
        
        return prompt.format(
            question=question,
            answer=answer,
            user_time=user_time or "未提供",
            location=location or "未提供", 
            passage=passage
        )
    
    def _evaluate_distractor(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str],
        location: Optional[str],
        open_ended_question: str,
        distractor_passage: str,
        distractor_answer: str
    ) -> Dict:
        """Evaluate the quality of generated distractor"""
        
        prompt = self._build_distractor_evaluation_prompt(
            question=question,
            answer=answer,
            passage=passage,
            user_time=user_time,
            location=location,
            open_ended_question=open_ended_question,
            distractor_passage=distractor_passage,
            distractor_answer=distractor_answer
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
            evaluation = self._parse_json_response(response)
            
            if evaluation:
                # Calculate overall score as average of individual scores
                scores = [
                    evaluation.get("relevance-score", 1),
                    evaluation.get("distraction-score", 1), 
                    evaluation.get("format-score", 1)
                ]
                evaluation["overall_score"] = sum(scores) / len(scores)
                evaluation["feedback"] = evaluation.get("thought-process", "")
            
            return evaluation or {"overall_score": 1, "feedback": "Failed to parse evaluation"}
            
        except Exception as e:
            self.logger.error(f"Error evaluating distractor: {e}")
            return {"overall_score": 1, "feedback": f"Evaluation error: {e}"}
    
    def _build_distractor_evaluation_prompt(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str],
        location: Optional[str],
        open_ended_question: str,
        distractor_passage: str,
        distractor_answer: str
    ) -> str:
        """Build prompt for distractor evaluation"""
        
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
## 用户时间：{user_time}
## 地点：{location}
## 段落：{passage}
## 开放式问题：{open_ended_question}
## 干扰段落：{distractor_passage}
## 干扰段落的答案：{distractor_answer}

### 助手："""
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response from LLM"""
        try:
            # Find JSON content between braces
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            return None
    
    def generate_batch(
        self,
        samples: List[Tuple[str, str, str]],
        user_time: Optional[str] = None,
        location: Optional[str] = None
    ) -> List[DistractorSample]:
        """Generate distractors for a batch of samples"""
        
        results = []
        for question, answer, passage in samples:
            distractor = self.generate_distractor(
                question=question,
                answer=answer,
                passage=passage,
                user_time=user_time,
                location=location
            )
            if distractor:
                results.append(distractor)
        
        return results