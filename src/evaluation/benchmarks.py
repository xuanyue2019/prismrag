"""
Benchmark loader for PrismRAG evaluation
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional


class BenchmarkLoader:
    """
    Loader for various RAG QA benchmarks.
    
    Supports loading and preprocessing of multiple benchmark datasets
    used in the PrismRAG paper evaluation.
    """
    
    def __init__(self, data_dir: str = "data/benchmarks"):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Benchmark configurations
        self.benchmark_configs = {
            "crag": {
                "file_pattern": "crag*.json",
                "question_key": "question",
                "answer_key": "answer", 
                "references_key": "references"
            },
            "covidqa": {
                "file_pattern": "covidqa*.json",
                "question_key": "question",
                "answer_key": "answer",
                "references_key": "context"
            },
            "delucionqa": {
                "file_pattern": "delucionqa*.json", 
                "question_key": "question",
                "answer_key": "answer",
                "references_key": "context"
            },
            "emanual": {
                "file_pattern": "emanual*.json",
                "question_key": "question", 
                "answer_key": "answer",
                "references_key": "context"
            },
            "expertqa": {
                "file_pattern": "expertqa*.json",
                "question_key": "question",
                "answer_key": "answer", 
                "references_key": "references"
            },
            "finqa": {
                "file_pattern": "finqa*.json",
                "question_key": "question",
                "answer_key": "answer",
                "references_key": "context"
            },
            "hagrid": {
                "file_pattern": "hagrid*.json",
                "question_key": "question",
                "answer_key": "answer",
                "references_key": "references"
            },
            "hotpotqa": {
                "file_pattern": "hotpotqa*.json",
                "question_key": "question", 
                "answer_key": "answer",
                "references_key": "context"
            },
            "ms_marco": {
                "file_pattern": "ms_marco*.json",
                "question_key": "query",
                "answer_key": "answers",
                "references_key": "passages"
            },
            "pubmedqa": {
                "file_pattern": "pubmedqa*.json",
                "question_key": "question",
                "answer_key": "final_decision", 
                "references_key": "context"
            },
            "tatqa": {
                "file_pattern": "tatqa*.json",
                "question_key": "question",
                "answer_key": "answer",
                "references_key": "table_context"
            },
            "techqa": {
                "file_pattern": "techqa*.json",
                "question_key": "question",
                "answer_key": "answer",
                "references_key": "context"
            }
        }
    
    def load_benchmark(
        self,
        benchmark_name: str,
        max_samples: Optional[int] = None,
        max_references: Optional[int] = None,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        Load a specific benchmark dataset.
        
        Args:
            benchmark_name: Name of the benchmark
            max_samples: Maximum number of samples to load
            max_references: Maximum number of references per sample
            shuffle: Whether to shuffle the data
            
        Returns:
            List of benchmark samples
        """
        
        if benchmark_name not in self.benchmark_configs:
            # Try to create mock data for demonstration
            return self._create_mock_benchmark_data(benchmark_name, max_samples or 100)
        
        config = self.benchmark_configs[benchmark_name]
        
        # Find benchmark file
        benchmark_file = self._find_benchmark_file(benchmark_name, config["file_pattern"])
        
        if not benchmark_file:
            self.logger.warning(f"Benchmark file not found for {benchmark_name}, creating mock data")
            return self._create_mock_benchmark_data(benchmark_name, max_samples or 100)
        
        # Load data
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Process data
        processed_data = self._process_benchmark_data(
            raw_data, config, max_references
        )
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(processed_data)
        
        # Limit samples
        if max_samples:
            processed_data = processed_data[:max_samples]
        
        self.logger.info(f"Loaded {len(processed_data)} samples from {benchmark_name}")
        return processed_data
    
    def _find_benchmark_file(self, benchmark_name: str, file_pattern: str) -> Optional[str]:
        """Find the benchmark file matching the pattern"""
        
        if not os.path.exists(self.data_dir):
            return None
        
        import glob
        pattern_path = os.path.join(self.data_dir, file_pattern)
        matching_files = glob.glob(pattern_path)
        
        if matching_files:
            return matching_files[0]  # Return first match
        
        return None
    
    def _process_benchmark_data(
        self,
        raw_data: List[Dict],
        config: Dict,
        max_references: Optional[int] = None
    ) -> List[Dict]:
        """Process raw benchmark data into standard format"""
        
        processed = []
        
        for item in raw_data:
            try:
                # Extract fields based on config
                question = item.get(config["question_key"], "")
                answer = item.get(config["answer_key"], "")
                references = item.get(config["references_key"], [])
                
                # Handle different reference formats
                if isinstance(references, str):
                    references = [references]
                elif isinstance(references, dict):
                    # For some benchmarks, references might be in dict format
                    references = list(references.values())
                
                # Ensure references is a list of strings
                references = [str(ref) for ref in references if ref]
                
                # Limit references if specified
                if max_references and len(references) > max_references:
                    references = references[:max_references]
                
                # Skip if essential fields are missing
                if not question or not answer or not references:
                    continue
                
                processed.append({
                    "question": question,
                    "answer": answer,
                    "references": references
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing benchmark item: {e}")
                continue
        
        return processed
    
    def _create_mock_benchmark_data(
        self,
        benchmark_name: str,
        num_samples: int
    ) -> List[Dict]:
        """Create mock benchmark data for demonstration"""
        
        self.logger.info(f"Creating {num_samples} mock samples for {benchmark_name}")
        
        # Mock questions and answers based on benchmark type
        mock_templates = {
            "crag": {
                "questions": [
                    "什么是人工智能的主要应用领域？",
                    "机器学习和深度学习有什么区别？",
                    "自然语言处理的核心技术有哪些？",
                    "计算机视觉在哪些行业中应用最广泛？",
                    "强化学习的基本原理是什么？"
                ],
                "contexts": [
                    "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                    "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
                    "自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
                    "计算机视觉是一个跨学科领域，研究如何使计算机从数字图像或视频中获得高层次的理解。",
                    "强化学习是机器学习的一个领域，涉及智能代理如何在环境中采取行动以最大化累积奖励。"
                ]
            },
            "covidqa": {
                "questions": [
                    "COVID-19的主要症状有哪些？",
                    "如何预防新冠病毒感染？",
                    "疫苗接种的重要性是什么？",
                    "新冠病毒如何传播？",
                    "什么是群体免疫？"
                ],
                "contexts": [
                    "COVID-19是由SARS-CoV-2病毒引起的传染病。常见症状包括发热、咳嗽和呼吸困难。",
                    "预防措施包括戴口罩、保持社交距离、勤洗手和接种疫苗。",
                    "疫苗接种是预防COVID-19最有效的方法之一，可以显著降低重症和死亡风险。",
                    "新冠病毒主要通过呼吸道飞沫和接触传播，在密闭空间中传播风险更高。",
                    "群体免疫是指当足够多的人对某种疾病具有免疫力时，整个社区都能得到保护。"
                ]
            }
        }
        
        # Use default template if benchmark not found
        template = mock_templates.get(benchmark_name, mock_templates["crag"])
        
        mock_data = []
        for i in range(num_samples):
            question_idx = i % len(template["questions"])
            context_idx = i % len(template["contexts"])
            
            question = template["questions"][question_idx]
            context = template["contexts"][context_idx]
            
            # Generate a simple answer based on the context
            answer = context.split("。")[0] + "。"  # Take first sentence as answer
            
            mock_data.append({
                "question": question,
                "answer": answer,
                "references": [context, context + " 这是额外的参考信息。"]
            })
        
        return mock_data
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available benchmarks"""
        return list(self.benchmark_configs.keys())
    
    def get_benchmark_info(self, benchmark_name: str) -> Optional[Dict]:
        """Get information about a specific benchmark"""
        
        if benchmark_name not in self.benchmark_configs:
            return None
        
        config = self.benchmark_configs[benchmark_name]
        benchmark_file = self._find_benchmark_file(benchmark_name, config["file_pattern"])
        
        info = {
            "name": benchmark_name,
            "config": config,
            "file_exists": benchmark_file is not None,
            "file_path": benchmark_file
        }
        
        if benchmark_file:
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                info["num_samples"] = len(data)
            except Exception as e:
                self.logger.warning(f"Error reading benchmark file: {e}")
                info["num_samples"] = 0
        
        return info