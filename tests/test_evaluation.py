"""
Tests for evaluation modules
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import (
    PrismRAGEvaluator,
    FactualityMetrics,
    BenchmarkLoader
)


class TestFactualityMetrics:
    """Test cases for FactualityMetrics"""
    
    @pytest.fixture
    def metrics(self):
        """Create a mock metrics instance for testing"""
        with patch('evaluation.metrics.AutoModelForCausalLM'):
            with patch('evaluation.metrics.AutoTokenizer'):
                return FactualityMetrics(evaluator_model="mock-model")
    
    def test_is_missing_response_refusal(self, metrics):
        """Test detection of refusal responses"""
        refusal_responses = [
            "我不知道",
            "抱歉，我无法回答这个问题",
            "没有足够的信息",
            "I don't know",
            "Sorry, I cannot answer"
        ]
        
        for response in refusal_responses:
            assert metrics._is_missing_response(response) == True
    
    def test_is_missing_response_short(self, metrics):
        """Test detection of too short responses"""
        short_response = "是的"
        assert metrics._is_missing_response(short_response) == True
    
    def test_is_missing_response_normal(self, metrics):
        """Test normal responses are not classified as missing"""
        normal_response = "人工智能是计算机科学的一个分支，专注于创建智能机器。"
        assert metrics._is_missing_response(normal_response) == False
    
    def test_calculate_factuality_score_empty(self, metrics):
        """Test factuality score calculation with empty input"""
        result = metrics.calculate_factuality_score([])
        
        expected = {
            "accuracy": 0.0,
            "hallucination_rate": 0.0,
            "missing_rate": 0.0,
            "factuality_score": 0.0
        }
        
        assert result == expected
    
    def test_calculate_factuality_score_mixed(self, metrics):
        """Test factuality score calculation with mixed categories"""
        categories = ["accurate", "accurate", "hallucination", "missing", "accurate"]
        result = metrics.calculate_factuality_score(categories)
        
        assert result["accuracy"] == 0.6  # 3/5
        assert result["hallucination_rate"] == 0.2  # 1/5
        assert result["missing_rate"] == 0.2  # 1/5
        assert result["factuality_score"] == 0.4  # 0.6 - 0.2
    
    @patch('evaluation.metrics.FactualityMetrics._evaluate_factuality_with_llm')
    def test_categorize_response_accurate(self, mock_evaluate, metrics):
        """Test response categorization as accurate"""
        mock_evaluate.return_value = 4.0
        
        category = metrics.categorize_response(
            question="What is AI?",
            ground_truth="AI is artificial intelligence",
            generated_response="AI is a field of computer science",
            references=["AI reference document"]
        )
        
        assert category == "accurate"
    
    @patch('evaluation.metrics.FactualityMetrics._evaluate_factuality_with_llm')
    def test_categorize_response_hallucination(self, mock_evaluate, metrics):
        """Test response categorization as hallucination"""
        mock_evaluate.return_value = 1.5
        
        category = metrics.categorize_response(
            question="What is AI?",
            ground_truth="AI is artificial intelligence",
            generated_response="AI was invented by aliens in 1950",
            references=["AI reference document"]
        )
        
        assert category == "hallucination"
    
    def test_categorize_response_missing(self, metrics):
        """Test response categorization as missing"""
        category = metrics.categorize_response(
            question="What is AI?",
            ground_truth="AI is artificial intelligence",
            generated_response="我不知道",
            references=["AI reference document"]
        )
        
        assert category == "missing"


class TestBenchmarkLoader:
    """Test cases for BenchmarkLoader"""
    
    @pytest.fixture
    def loader(self):
        """Create a benchmark loader for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield BenchmarkLoader(data_dir=temp_dir)
    
    def test_get_available_benchmarks(self, loader):
        """Test getting list of available benchmarks"""
        benchmarks = loader.get_available_benchmarks()
        
        assert isinstance(benchmarks, list)
        assert "crag" in benchmarks
        assert "covidqa" in benchmarks
        assert "expertqa" in benchmarks
    
    def test_create_mock_benchmark_data(self, loader):
        """Test creation of mock benchmark data"""
        mock_data = loader._create_mock_benchmark_data("crag", 10)
        
        assert len(mock_data) == 10
        for item in mock_data:
            assert "question" in item
            assert "answer" in item
            assert "references" in item
            assert isinstance(item["references"], list)
    
    def test_load_benchmark_mock_data(self, loader):
        """Test loading benchmark with mock data"""
        data = loader.load_benchmark("crag", max_samples=5)
        
        assert len(data) == 5
        for item in data:
            assert "question" in item
            assert "answer" in item
            assert "references" in item
    
    def test_get_benchmark_info_existing(self, loader):
        """Test getting info for existing benchmark"""
        info = loader.get_benchmark_info("crag")
        
        assert info is not None
        assert info["name"] == "crag"
        assert "config" in info
        assert "file_exists" in info
    
    def test_get_benchmark_info_nonexistent(self, loader):
        """Test getting info for non-existent benchmark"""
        info = loader.get_benchmark_info("nonexistent_benchmark")
        
        assert info is None


class TestPrismRAGEvaluator:
    """Test cases for PrismRAGEvaluator"""
    
    @pytest.fixture
    def evaluator(self):
        """Create a mock evaluator for testing"""
        with patch('evaluation.evaluator.AutoModelForCausalLM'):
            with patch('evaluation.evaluator.AutoTokenizer'):
                with patch('evaluation.evaluator.BenchmarkLoader'):
                    with patch('evaluation.evaluator.FactualityMetrics'):
                        return PrismRAGEvaluator(model_path="mock-model")
    
    def test_extract_final_answer_with_marker(self, evaluator):
        """Test extracting final answer with marker"""
        response = """
        这是一些推理过程。
        
        ## 答案:
        这是最终答案。
        """
        
        answer = evaluator._extract_final_answer(response)
        assert answer == "这是最终答案。"
    
    def test_extract_final_answer_without_marker(self, evaluator):
        """Test extracting final answer without marker"""
        response = """
        第一段内容。
        
        第二段内容。
        
        最后一段作为答案。
        """
        
        answer = evaluator._extract_final_answer(response)
        assert answer == "最后一段作为答案。"
    
    def test_extract_final_answer_single_paragraph(self, evaluator):
        """Test extracting final answer from single paragraph"""
        response = "这是单段回答。"
        
        answer = evaluator._extract_final_answer(response)
        assert answer == "这是单段回答。"
    
    @patch('evaluation.evaluator.PrismRAGEvaluator._generate_responses')
    @patch('evaluation.evaluator.PrismRAGEvaluator._evaluate_single_response')
    def test_evaluate_benchmark(self, mock_evaluate_single, mock_generate, evaluator):
        """Test benchmark evaluation"""
        # Mock benchmark data
        evaluator.benchmark_loader.load_benchmark.return_value = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence",
                "references": ["AI reference"]
            }
        ]
        
        # Mock generated responses
        mock_generate.return_value = ["AI is a computer science field"]
        
        # Mock single response evaluation
        mock_evaluate_single.return_value = {
            "sample_id": 0,
            "category": "accurate",
            "metrics": {}
        }
        
        result = evaluator.evaluate_benchmark("crag", num_samples=1)
        
        assert result.benchmark == "crag"
        assert result.num_samples == 1
        assert result.accuracy == 1.0
        assert result.hallucination_rate == 0.0
        assert result.factuality_score == 1.0
    
    @patch('evaluation.evaluator.PrismRAGEvaluator.evaluate_benchmark')
    def test_evaluate_multiple_benchmarks(self, mock_evaluate_benchmark, evaluator):
        """Test multiple benchmark evaluation"""
        from evaluation.evaluator import EvaluationResult
        
        # Mock individual benchmark results
        mock_result = EvaluationResult(
            benchmark="crag",
            accuracy=0.8,
            hallucination_rate=0.1,
            missing_rate=0.1,
            factuality_score=0.7,
            num_samples=100,
            detailed_results=[]
        )
        mock_evaluate_benchmark.return_value = mock_result
        
        results = evaluator.evaluate_multiple_benchmarks(["crag", "covidqa"])
        
        assert len(results) == 2
        assert "crag" in results
        assert "covidqa" in results
        assert results["crag"].factuality_score == 0.7
    
    def test_build_inference_prompt(self, evaluator):
        """Test building inference prompt"""
        prompt = evaluator._build_inference_prompt(
            question="What is AI?",
            references=["AI is artificial intelligence", "AI is computer science"]
        )
        
        assert "What is AI?" in prompt
        assert "参考资料 1:" in prompt
        assert "参考资料 2:" in prompt
        assert "AI is artificial intelligence" in prompt
        assert "AI is computer science" in prompt
        assert "## 答案:" in prompt


if __name__ == "__main__":
    pytest.main([__file__])