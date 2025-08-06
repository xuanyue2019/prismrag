"""
Seed Data Generator for PrismRAG

This module generates seed question-answer-passage triplets from raw documents,
following the approach described in the PrismRAG paper.
"""

import json
import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class SeedSample:
    """Data class for seed samples"""
    question: str
    answer: str
    passage: str
    source: str  # "wikipedia" or "web_search"
    metadata: Optional[Dict] = None


class SeedDataGenerator:
    """
    Generates seed QA data from raw documents.
    
    This class implements the inverse problem approach described in the paper:
    given a document chunk, generate a question-answer pair based on its content.
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
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_from_wikipedia(
        self,
        wikipedia_pages: List[str],
        min_words: int = 500,
        max_words: int = 7000,
        min_lines: int = 10,
        max_lines: int = 1000,
        chunk_size_min: int = 250,
        chunk_size_max: int = 1000,
        num_references_min: int = 2,
        num_references_max: int = 15
    ) -> List[SeedSample]:
        """
        Generate seed data from Wikipedia pages.
        
        Args:
            wikipedia_pages: List of Wikipedia page contents
            min_words: Minimum words per page
            max_words: Maximum words per page
            min_lines: Minimum lines per page
            max_lines: Maximum lines per page
            chunk_size_min: Minimum words per chunk
            chunk_size_max: Maximum words per chunk
            num_references_min: Minimum number of reference chunks
            num_references_max: Maximum number of reference chunks
            
        Returns:
            List of SeedSample objects
        """
        
        seed_samples = []
        
        for page_content in wikipedia_pages:
            # Filter pages by length
            word_count = len(page_content.split())
            line_count = len(page_content.split('\n'))
            
            if word_count < min_words or word_count > max_words:
                continue
            if line_count < min_lines or line_count > max_lines:
                continue
            
            # Split page into chunks
            chunks = self._split_into_chunks(
                page_content, 
                chunk_size_min, 
                chunk_size_max
            )
            
            if len(chunks) < num_references_min:
                continue
            
            # Randomly select chunks as references
            num_refs = random.randint(num_references_min, min(num_references_max, len(chunks)))
            selected_chunks = random.sample(chunks, num_refs)
            
            # Select one chunk as the golden reference for QA generation
            golden_chunk = random.choice(selected_chunks)
            
            # Generate QA pair from golden chunk
            qa_pair = self._generate_qa_from_chunk(golden_chunk)
            
            if qa_pair:
                seed_samples.append(SeedSample(
                    question=qa_pair["question"],
                    answer=qa_pair["answer"],
                    passage=golden_chunk,
                    source="wikipedia",
                    metadata={
                        "num_references": len(selected_chunks),
                        "all_references": selected_chunks,
                        "word_count": word_count,
                        "line_count": line_count
                    }
                ))
        
        return seed_samples
    
    def generate_from_web_search(
        self,
        web_pages: List[Dict],
        max_words_per_page: int = 3000
    ) -> List[SeedSample]:
        """
        Generate seed data from web search results.
        
        Args:
            web_pages: List of web page data with 'content', 'query', 'time', 'location'
            max_words_per_page: Maximum words per page
            
        Returns:
            List of SeedSample objects
        """
        
        seed_samples = []
        
        for page_data in web_pages:
            content = page_data.get("content", "")
            query = page_data.get("query", "")
            time_context = page_data.get("time", "")
            location_context = page_data.get("location", "")
            
            # Truncate content if too long
            words = content.split()
            if len(words) > max_words_per_page:
                content = " ".join(words[:max_words_per_page])
            
            # Generate QA pair from content
            qa_pair = self._generate_qa_from_chunk(
                content,
                context={
                    "query": query,
                    "time": time_context,
                    "location": location_context
                }
            )
            
            if qa_pair:
                seed_samples.append(SeedSample(
                    question=qa_pair["question"],
                    answer=qa_pair["answer"],
                    passage=content,
                    source="web_search",
                    metadata={
                        "original_query": query,
                        "time_context": time_context,
                        "location_context": location_context,
                        "word_count": len(content.split())
                    }
                ))
        
        return seed_samples
    
    def _split_into_chunks(
        self,
        text: str,
        min_size: int,
        max_size: int
    ) -> List[str]:
        """Split text into non-overlapping chunks"""
        
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            if len(current_chunk) >= max_size:
                if len(current_chunk) >= min_size:
                    chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # Add remaining chunk if it meets minimum size
        if len(current_chunk) >= min_size:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _generate_qa_from_chunk(
        self,
        chunk: str,
        context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Generate a question-answer pair from a text chunk"""
        
        prompt = self._build_qa_generation_prompt(chunk, context)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return self._parse_qa_response(response)
            
        except Exception as e:
            self.logger.error(f"Error generating QA from chunk: {e}")
            return None
    
    def _build_qa_generation_prompt(
        self,
        content: str,
        context: Optional[Dict] = None
    ) -> str:
        """Build prompt for QA generation"""
        
        prompt = """你是一个乐于助人的助手。始终遵循提供的指令，并以有效的json格式生成输出，不包含任何额外信息。

根据下面提供的Content生成一个问题和答案对。

## 要求：
- 您必须将您的问题和答案建立在提供的Content的基础上
- 应该选择这个问题，使其类似于一个好奇的大学毕业生会问一个智能对话系统的问题。从难度级别1到10，目标是8。
- 答案应完全并直接基于提供的Content。永远不要使用提供的Content中没有的任何信息来生成问题和答案。
- 永远不要生成一个询问当前时间、日期或地点的问题。
- 这个问题不应该太笼统或模糊。在适用的情况下，在问题中包含特定的实体、名称、时间、地点、事件和关键词。
- 问答必须在语法上正确，并且在对话中自然。
- 一个好的问题应该有意义，并提供足够的上下文。
- 始终以json格式返回，包含两个键："question"和"answer"。如果提供的内容不可读，您可以将与问题和答案键对应的值设置为"N/A"。

## 例子：
以下是一些需要考虑的问题类型的示例：
1. 奥巴马多大了？
2. 美国第一任总统叫什么名字？
3. 中国的人口是多少？
4. 玛格丽塔的主要成分是什么？
5. 为什么巧克力对狗有害？

## 提供的文本：
{content}"""

        if context:
            prompt += f"\n\n## 上下文信息：\n"
            for key, value in context.items():
                if value:
                    prompt += f"- {key}: {value}\n"

        return prompt
    
    def _parse_qa_response(self, response: str) -> Optional[Dict]:
        """Parse QA response from LLM"""
        try:
            # Find JSON content
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = response[start_idx:end_idx]
            qa_data = json.loads(json_str)
            
            # Validate required fields
            if "question" not in qa_data or "answer" not in qa_data:
                return None
            
            # Check for N/A responses
            if qa_data["question"] == "N/A" or qa_data["answer"] == "N/A":
                return None
            
            return qa_data
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse QA response: {e}")
            return None
    
    def generate_batch(
        self,
        documents: List[Dict],
        source_type: str = "wikipedia"
    ) -> List[SeedSample]:
        """Generate seed data for a batch of documents"""
        
        if source_type == "wikipedia":
            return self.generate_from_wikipedia([doc["content"] for doc in documents])
        elif source_type == "web_search":
            return self.generate_from_web_search(documents)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def save_samples(self, samples: List[SeedSample], filepath: str):
        """Save seed samples to file"""
        
        data = []
        for sample in samples:
            data.append({
                "question": sample.question,
                "answer": sample.answer,
                "passage": sample.passage,
                "source": sample.source,
                "metadata": sample.metadata
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(samples)} seed samples to {filepath}")
    
    def load_samples(self, filepath: str) -> List[SeedSample]:
        """Load seed samples from file"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            samples.append(SeedSample(
                question=item["question"],
                answer=item["answer"],
                passage=item["passage"],
                source=item["source"],
                metadata=item.get("metadata")
            ))
        
        self.logger.info(f"Loaded {len(samples)} seed samples from {filepath}")
        return samples