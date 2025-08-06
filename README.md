# PrismRAG: 利用干扰项恢复能力和策略性推理提升 RAG 事实性

[![CI](https://github.com/xuanyue2019/prismrag/workflows/CI/badge.svg)](https://github.com/xuanyue2019/prismrag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

基于 Meta AI 研究论文的 PrismRAG 实现，通过干扰项抵抗力和策略化思维链微调来提升检索增强生成(RAG)的事实性。

## 项目概述

PrismRAG 是一个高效的微调框架，主要包含两个核心机制：

1. **干扰项抵抗力 (Distractor Resilience)**: 使用混合了金标准证据和细微干扰项段落的训练数据来增强模型对检索噪声的鲁棒性
2. **策略化思维链 (Strategic CoT)**: 通过动态生成推理策略来提升模型的推理能力，减少对人工设计指令的依赖

## 主要特性

- 🎯 **提升事实性**: 在12个RAG QA基准测试中平均提升5.4%的事实性得分
- 🛡️ **抗干扰能力**: 有效处理半相关和混淆性检索内容
- 🧠 **智能推理**: 动态生成推理策略，适应不同问题类型
- 📊 **可扩展**: 支持大规模合成数据生成和自动质量评估

## 项目结构

```
.seal/prismRAG/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── config/                      # 配置文件
├── src/                         # 源代码
│   ├── data_generation/         # 数据生成模块
│   ├── training/                # 模型训练模块
│   ├── evaluation/              # 评估模块
│   └── utils/                   # 工具函数
├── data/                        # 数据目录
├── models/                      # 模型存储
├── experiments/                 # 实验脚本
├── tests/                       # 测试代码
└── docs/                        # 文档
```

## 快速开始

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **生成训练数据**
```bash
python experiments/generate_training_data.py
```

3. **训练模型**
```bash
python experiments/train_prismrag.py
```

4. **评估模型**
```bash
python experiments/evaluate_model.py
```

## 核心算法

### 干扰项生成流程
1. 识别黄金段落中的关键实体、位置和时间信息
2. 将原始问题重新表述为开放式问题
3. 系统性修改关键信息生成干扰项段落
4. 评估干扰项质量并迭代优化

### 策略化CoT生成流程
1. 生成解决问题的高层策略大纲
2. 基于策略生成详细的思维链推理
3. 评估推理质量和答案正确性
4. 迭代优化直到达到质量标准

## 实验结果

在12个公开RAG QA基准测试中的表现：

| 基准测试 | 基线 | PrismRAG | 提升 |
|---------|------|----------|------|
| CRAG | 34.2% | 39.2% | +5.0% |
| CovidQA | 80.0% | 95.0% | +15.0% |
| DelucionQA | 89.0% | 97.0% | +8.0% |
| 平均 | 78.4% | 83.8% | +5.4% |

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 许可证

MIT License

## 引用

如果您使用了这个项目，请引用原始论文：

```bibtex
@article{kachuee2025prismrag,
  title={PrismRAG: Improving RAG Factuality through Distractor Resilience and Strategic Reasoning},
  author={Kachuee, Mohammad and Gollapudi, Teja and Kim, Minseok and others},
  journal={arXiv preprint arXiv:2507.18857},
  year={2025}
}
```