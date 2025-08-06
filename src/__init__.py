"""
PrismRAG: Improving RAG Factuality through Distractor Resilience and Strategic Reasoning
"""

__version__ = "0.1.0"
__author__ = "PrismRAG Team"
__email__ = "prismrag@example.com"

from .data_generation import DistractorGenerator, StrategicCoTGenerator
from .training import PrismRAGTrainer
from .evaluation import PrismRAGEvaluator

__all__ = [
    "DistractorGenerator",
    "StrategicCoTGenerator", 
    "PrismRAGTrainer",
    "PrismRAGEvaluator"
]