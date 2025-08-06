"""
Tests for data generation modules
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation import (
    SeedDataGenerator,
    DistractorGenerator,
    StrategicCoTGenerator
)


class TestSeedDataGenerator:
    """Test cases for SeedDataGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create a mock generator for testing"""
        with patch('data_generation.seed_data_generator.AutoModelForCausalLM'):
            with patch('data_generation.seed_data_generator.AutoTokenizer'):
                return SeedDataGenerator(model_name="mock-model")
    
    def test_split_into_chunks(self, generator):
        """Test text chunking functionality"""
        text = "This is a test text with many words to split into chunks"
        chunks = generator._split_into_chunks(text, min_size=3, max_size=5)
        
        assert len(chunks) > 0
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count >= 3
    
    def test_generate_from_wikipedia_empty_input(self, generator):
        """Test handling of empty Wikipedia input"""
        result = generator.generate_from_wikipedia([])
        assert result == []
    
    def test_generate_from_web_search_empty_input(self, generator):
        """Test handling of empty web search input"""
        result = generator.generate_from_web_search([])
        assert result == []
    
    def test_save_and_load_samples(self, generator):
        """Test saving and loading samples"""
        from data_generation.seed_data_generator import SeedSample
        
        samples = [
            SeedSample(
                question="Test question?",
                answer="Test answer",
                passage="Test passage",
                source="test"
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            generator.save_samples(samples, filepath)
            loaded_samples = generator.load_samples(filepath)
            
            assert len(loaded_samples) == 1
            assert loaded_samples[0].question == "Test question?"
            assert loaded_samples[0].answer == "Test answer"
            assert loaded_samples[0].source == "test"
        finally:
            os.unlink(filepath)


class TestDistractorGenerator:
    """Test cases for DistractorGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create a mock generator for testing"""
        with patch('data_generation.distractor_generator.AutoModelForCausalLM'):
            with patch('data_generation.distractor_generator.AutoTokenizer'):
                return DistractorGenerator(model_name="mock-model")
    
    def test_parse_json_response_valid(self, generator):
        """Test parsing valid JSON response"""
        response = 'Some text {"key": "value", "score": 4} more text'
        result = generator._parse_json_response(response)
        
        assert result is not None
        assert result["key"] == "value"
        assert result["score"] == 4
    
    def test_parse_json_response_invalid(self, generator):
        """Test parsing invalid JSON response"""
        response = 'Some text without valid JSON'
        result = generator._parse_json_response(response)
        
        assert result is None
    
    @patch('data_generation.distractor_generator.DistractorGenerator._generate_distractor_with_llm')
    @patch('data_generation.distractor_generator.DistractorGenerator._evaluate_distractor')
    def test_generate_distractor_success(self, mock_evaluate, mock_generate, generator):
        """Test successful distractor generation"""
        # Mock successful generation
        mock_generate.return_value = {
            "open_ended_question": "What is AI?",
            "distractor_passage": "AI is a modified passage",
            "distractor_answer": "Modified answer"
        }
        
        # Mock high quality evaluation
        mock_evaluate.return_value = {"overall_score": 4.5, "feedback": "Good quality"}
        
        result = generator.generate_distractor(
            question="What is artificial intelligence?",
            answer="AI is computer science field",
            passage="AI is a branch of computer science"
        )
        
        assert result is not None
        assert result.open_ended_question == "What is AI?"
        assert result.distractor_passage == "AI is a modified passage"
    
    @patch('data_generation.distractor_generator.DistractorGenerator._generate_distractor_with_llm')
    def test_generate_distractor_failure(self, mock_generate, generator):
        """Test distractor generation failure"""
        # Mock failed generation
        mock_generate.return_value = None
        
        result = generator.generate_distractor(
            question="What is AI?",
            answer="AI is computer science",
            passage="AI passage"
        )
        
        assert result is None


class TestStrategicCoTGenerator:
    """Test cases for StrategicCoTGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create a mock generator for testing"""
        with patch('data_generation.strategic_cot_generator.AutoModelForCausalLM'):
            with patch('data_generation.strategic_cot_generator.AutoTokenizer'):
                return StrategicCoTGenerator(model_name="mock-model")
    
    def test_parse_strategic_cot_response_valid(self, generator):
        """Test parsing valid strategic CoT response"""
        response = """
        ## 策略：
        - 步骤1: 分析问题
        - 步骤2: 查找信息
        
        ## 推理：
        - 步骤1: 问题是关于AI的
        - 步骤2: 从参考资料中找到相关信息
        
        ## 答案：
        AI是人工智能的缩写
        """
        
        result = generator._parse_strategic_cot_response(response)
        
        assert result is not None
        assert "步骤1: 分析问题" in result["strategy"]
        assert "步骤1: 问题是关于AI的" in result["reasoning"]
        assert "AI是人工智能的缩写" in result["answer"]
    
    def test_parse_strategic_cot_response_invalid(self, generator):
        """Test parsing invalid strategic CoT response"""
        response = "Invalid response without proper sections"
        result = generator._parse_strategic_cot_response(response)
        
        assert result is None
    
    @patch('data_generation.strategic_cot_generator.StrategicCoTGenerator._generate_cot_random')
    @patch('data_generation.strategic_cot_generator.StrategicCoTGenerator._evaluate_reasoning')
    @patch('data_generation.strategic_cot_generator.StrategicCoTGenerator._evaluate_answer')
    def test_generate_strategic_cot_success(self, mock_eval_answer, mock_eval_reasoning, mock_generate, generator):
        """Test successful strategic CoT generation"""
        # Mock successful generation
        mock_generate.return_value = {
            "strategy": "Test strategy",
            "reasoning": "Test reasoning",
            "answer": "Test answer"
        }
        
        # Mock high quality evaluations
        mock_eval_reasoning.return_value = 4
        mock_eval_answer.return_value = 4
        
        result = generator.generate_strategic_cot(
            question="What is AI?",
            references=["AI reference document"],
            ground_truth_answer="AI is artificial intelligence"
        )
        
        assert result is not None
        assert result.strategy == "Test strategy"
        assert result.reasoning == "Test reasoning"
        assert result.answer == "Test answer"
    
    @patch('data_generation.strategic_cot_generator.StrategicCoTGenerator._generate_cot_random')
    @patch('data_generation.strategic_cot_generator.StrategicCoTGenerator._evaluate_reasoning')
    @patch('data_generation.strategic_cot_generator.StrategicCoTGenerator._evaluate_answer')
    def test_generate_strategic_cot_low_quality(self, mock_eval_answer, mock_eval_reasoning, mock_generate, generator):
        """Test strategic CoT generation with low quality"""
        # Mock generation
        mock_generate.return_value = {
            "strategy": "Poor strategy",
            "reasoning": "Poor reasoning", 
            "answer": "Poor answer"
        }
        
        # Mock low quality evaluations
        mock_eval_reasoning.return_value = 2
        mock_eval_answer.return_value = 2
        
        # Set low max_iterations for quick test
        generator.max_iterations = 2
        
        result = generator.generate_strategic_cot(
            question="What is AI?",
            references=["AI reference"],
            ground_truth_answer="AI is artificial intelligence"
        )
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])