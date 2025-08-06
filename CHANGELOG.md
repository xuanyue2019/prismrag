# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-06

### Added
- Initial implementation of PrismRAG based on Meta AI research paper
- Core data generation modules:
  - Seed data generator for Wikipedia and web search content
  - Distractor generator for synthetic noise creation
  - Strategic Chain-of-Thought generator for dynamic reasoning
- Training framework:
  - PrismRAG trainer with LoRA support
  - Mixed training data handling (distractor + CoT)
  - Distributed training capabilities
- Evaluation system:
  - Multi-benchmark evaluation (12 RAG QA benchmarks)
  - Factuality metrics implementation
  - Robustness analysis tools
- Utility functions:
  - Logging utilities
  - Data processing tools
  - Model utilities
- Comprehensive documentation:
  - API documentation
  - Design document
  - Usage examples
- Experiment scripts:
  - Training data generation
  - Model training
  - Model evaluation
- Test suite with unit tests for core modules
- Demo script showcasing full functionality

### Features
- **Distractor Resilience**: Synthetic distractor generation to improve robustness against retrieval noise
- **Strategic CoT**: Dynamic reasoning strategy generation for better inference
- **Factuality Scoring**: Accuracy - Hallucination rate evaluation metric
- **Multi-benchmark Support**: Evaluation on 12 different RAG QA benchmarks
- **Parameter Efficient Training**: LoRA fine-tuning support
- **Scalable Data Generation**: LLM-based synthetic data creation with quality control

### Technical Details
- Python 3.8+ support
- PyTorch and Transformers integration
- Weights & Biases experiment tracking
- vLLM inference acceleration
- Comprehensive error handling and logging

[Unreleased]: https://github.com/xuanyue2019/prismrag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/xuanyue2019/prismrag/releases/tag/v0.1.0