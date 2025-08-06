# PrismRAG API 文档

## 概述

本文档描述了 PrismRAG 项目的主要 API 接口和使用方法。

## 数据生成 API

### SeedDataGenerator

种子数据生成器，用于从原始文档生成问答对。

```python
from src.data_generation import SeedDataGenerator

# 初始化
generator = SeedDataGenerator(
    model_name="meta-llama/Llama-3.1-70b-instruct",
    device="auto"
)

# 从维基百科生成数据
wiki_samples = generator.generate_from_wikipedia(
    wikipedia_pages=["页面内容1", "页面内容2"],
    min_words=500,
    max_words=7000,
    chunk_size_min=250,
    chunk_size_max=1000
)

# 从网页搜索生成数据
web_samples = generator.generate_from_web_search(
    web_pages=[
        {
            "content": "网页内容",
            "query": "搜索查询",
            "time": "2025-01-01 10:00:00",
            "location": "北京"
        }
    ]
)
```

### DistractorGenerator

干扰项生成器，用于创建合成干扰项段落。

```python
from src.data_generation import DistractorGenerator

# 初始化
generator = DistractorGenerator(
    model_name="meta-llama/Llama-3.1-70b-instruct",
    max_iterations=5,
    quality_threshold=4
)

# 生成干扰项
distractor = generator.generate_distractor(
    question="什么是人工智能？",
    answer="人工智能是计算机科学的一个分支...",
    passage="人工智能相关的段落内容",
    user_time="2025-01-01 10:00:00",
    location="北京"
)

if distractor:
    print(f"原问题: {distractor.question}")
    print(f"开放式问题: {distractor.open_ended_question}")
    print(f"干扰项段落: {distractor.distractor_passage}")
```

### StrategicCoTGenerator

策略化思维链生成器，用于生成动态推理策略。

```python
from src.data_generation import StrategicCoTGenerator

# 初始化
generator = StrategicCoTGenerator(
    model_name="meta-llama/Llama-3.1-70b-instruct",
    max_iterations=10,
    quality_threshold=4
)

# 生成策略化CoT
cot_sample = generator.generate_strategic_cot(
    question="什么是机器学习？",
    references=["参考文档1", "参考文档2"],
    ground_truth_answer="机器学习是人工智能的一个子集...",
    user_context="技术讨论上下文"
)

if cot_sample:
    print(f"策略: {cot_sample.strategy}")
    print(f"推理: {cot_sample.reasoning}")
    print(f"答案: {cot_sample.answer}")
```

## 训练 API

### PrismRAGTrainer

模型训练器，支持LoRA微调和混合数据训练。

```python
from src.training import PrismRAGTrainer
from transformers import TrainingArguments

# 初始化训练器
trainer = PrismRAGTrainer(
    model_name="meta-llama/Llama-3.1-70b-instruct",
    output_dir="models/prismrag",
    use_lora=True
)

# 准备训练数据
train_dataset = trainer.prepare_training_data(
    distractor_samples=distractor_data,
    strategic_cot_samples=cot_data,
    max_length=4096
)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="models/prismrag",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    logging_steps=100,
    save_steps=500
)

# 开始训练
trainer.train(
    train_dataset=train_dataset,
    training_args=training_args,
    use_wandb=True
)
```

### PrismRAGDataset

训练数据集类，处理数据的标记化和格式化。

```python
from src.training import PrismRAGDataset
from transformers import AutoTokenizer

# 初始化
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70b-instruct")

dataset = PrismRAGDataset(
    samples=[
        {
            "instruction": "回答问题：什么是AI？",
            "response": "AI是人工智能的缩写...",
            "sample_type": "strategic_cot"
        }
    ],
    tokenizer=tokenizer,
    max_length=4096
)

# 获取样本分布
distribution = dataset.get_sample_type_distribution()
print(f"样本分布: {distribution}")
```

## 评估 API

### PrismRAGEvaluator

模型评估器，支持多基准测试和鲁棒性分析。

```python
from src.evaluation import PrismRAGEvaluator

# 初始化评估器
evaluator = PrismRAGEvaluator(
    model_path="models/prismrag/final_model",
    device="auto",
    use_vllm=True
)

# 单个基准评估
result = evaluator.evaluate_benchmark(
    benchmark_name="crag",
    num_samples=100,
    batch_size=8
)

print(f"事实性得分: {result.factuality_score}")
print(f"准确率: {result.accuracy}")
print(f"幻觉率: {result.hallucination_rate}")

# 多基准评估
results = evaluator.evaluate_multiple_benchmarks(
    benchmark_names=["crag", "covidqa", "expertqa"],
    num_samples_per_benchmark=100
)

# 打印结果摘要
evaluator.print_summary(results)

# 鲁棒性评估
robustness_results = evaluator.evaluate_robustness(
    benchmark_name="crag",
    reference_counts=[1, 5, 10, 20, 50],
    num_samples=100
)
```

### FactualityMetrics

事实性评估指标，实现论文中的评估方法。

```python
from src.evaluation import FactualityMetrics

# 初始化
metrics = FactualityMetrics(
    evaluator_model="meta-llama/Llama-3.1-70b-instruct"
)

# 分类单个回答
category = metrics.categorize_response(
    question="什么是机器学习？",
    ground_truth="机器学习是AI的一个子集...",
    generated_response="机器学习是一种算法...",
    references=["参考文档1", "参考文档2"]
)

print(f"回答分类: {category}")  # "accurate", "hallucination", 或 "missing"

# 计算整体事实性指标
categories = ["accurate", "accurate", "hallucination", "missing"]
factuality_scores = metrics.calculate_factuality_score(categories)

print(f"事实性指标: {factuality_scores}")
```

### BenchmarkLoader

基准测试加载器，支持多种RAG QA基准。

```python
from src.evaluation import BenchmarkLoader

# 初始化
loader = BenchmarkLoader(data_dir="data/benchmarks")

# 加载基准数据
benchmark_data = loader.load_benchmark(
    benchmark_name="crag",
    max_samples=100,
    max_references=10,
    shuffle=True
)

# 获取可用基准列表
available_benchmarks = loader.get_available_benchmarks()
print(f"可用基准: {available_benchmarks}")

# 获取基准信息
info = loader.get_benchmark_info("crag")
print(f"基准信息: {info}")
```

## 工具函数 API

### 数据工具

```python
from src.utils import load_json, save_json, split_text

# JSON文件操作
data = load_json("data/samples.json")
save_json(data, "output/processed_data.json")

# 文本分割
chunks = split_text(
    text="很长的文本内容...",
    max_length=1000,
    overlap=100,
    split_on=" "
)
```

### 模型工具

```python
from src.utils import load_model_and_tokenizer, generate_text

# 加载模型和分词器
model, tokenizer = load_model_and_tokenizer(
    model_name="meta-llama/Llama-3.1-70b-instruct",
    device="auto"
)

# 生成文本
generated_texts = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="什么是人工智能？",
    max_new_tokens=500,
    temperature=0.8
)
```

### 日志工具

```python
from src.utils import setup_logging

# 设置日志
logger = setup_logging(
    log_level="INFO",
    log_file="logs/prismrag.log"
)

logger.info("开始处理数据...")
```

## 配置 API

### 配置文件结构

```yaml
# config/default.yaml
model:
  base_model: "meta-llama/Llama-3.1-70b-instruct"
  max_length: 4096
  temperature: 1.0

training:
  learning_rate: 1e-5
  batch_size: 4
  num_epochs: 3

data_generation:
  distractor:
    max_iterations: 5
    quality_threshold: 4
  strategic_cot:
    max_iterations: 10
    quality_threshold: 4

evaluation:
  benchmarks:
    - "crag"
    - "covidqa"
    - "expertqa"
```

### 配置加载

```python
import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config("config/default.yaml")
```

## 错误处理

### 常见异常

```python
from src.data_generation import DistractorGenerator

try:
    generator = DistractorGenerator()
    distractor = generator.generate_distractor(question, answer, passage)
except Exception as e:
    print(f"生成干扰项时出错: {e}")
    # 处理错误逻辑
```

### 日志记录

```python
import logging

logger = logging.getLogger(__name__)

try:
    # 执行操作
    result = some_operation()
except Exception as e:
    logger.error(f"操作失败: {e}", exc_info=True)
    raise
```

## 性能优化建议

### 内存优化

```python
# 使用梯度检查点
training_args = TrainingArguments(
    gradient_checkpointing=True,
    fp16=True,
    dataloader_pin_memory=False
)

# 使用LoRA减少内存占用
trainer = PrismRAGTrainer(use_lora=True)
```

### 推理加速

```python
# 使用vLLM加速推理
evaluator = PrismRAGEvaluator(use_vllm=True)

# 批量处理
results = evaluator.evaluate_multiple_benchmarks(
    benchmark_names=benchmarks,
    batch_size=16  # 增加批次大小
)
```

### 数据处理优化

```python
# 并行数据生成
from concurrent.futures import ThreadPoolExecutor

def generate_batch_distractors(samples):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generator.generate_distractor, **sample)
            for sample in samples
        ]
        return [f.result() for f in futures]
```

## 示例用法

### 完整训练流程

```python
# 1. 生成训练数据
seed_generator = SeedDataGenerator()
distractor_generator = DistractorGenerator()
cot_generator = StrategicCoTGenerator()

# 生成种子数据
seed_samples = seed_generator.generate_from_wikipedia(wiki_pages)

# 生成干扰项数据
distractor_samples = []
for sample in seed_samples[:100]:
    distractor = distractor_generator.generate_distractor(
        sample.question, sample.answer, sample.passage
    )
    if distractor:
        distractor_samples.append(distractor)

# 生成CoT数据
cot_samples = []
for sample in seed_samples[:200]:
    cot = cot_generator.generate_strategic_cot(
        sample.question, [sample.passage], sample.answer
    )
    if cot:
        cot_samples.append(cot)

# 2. 训练模型
trainer = PrismRAGTrainer()
train_dataset = trainer.prepare_training_data(
    distractor_samples, cot_samples
)
trainer.train(train_dataset)

# 3. 评估模型
evaluator = PrismRAGEvaluator("models/prismrag/final_model")
results = evaluator.evaluate_multiple_benchmarks(["crag", "covidqa"])
evaluator.print_summary(results)
```

### 快速评估

```python
# 快速评估现有模型
evaluator = PrismRAGEvaluator("path/to/model")

# 在CRAG基准上评估100个样本
result = evaluator.evaluate_benchmark("crag", num_samples=100)
print(f"CRAG事实性得分: {result.factuality_score:.3f}")

# 鲁棒性测试
robustness = evaluator.evaluate_robustness(
    reference_counts=[1, 5, 10], 
    num_samples=50
)
```