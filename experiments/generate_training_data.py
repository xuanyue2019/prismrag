#!/usr/bin/env python3
"""
Generate training data for PrismRAG

This script generates both distractor resilience and strategic CoT training data
following the methodology described in the PrismRAG paper.
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List

import yaml
from tqdm import tqdm

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation import (
    SeedDataGenerator,
    DistractorGenerator, 
    StrategicCoTGenerator
)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_data_generation.log')
        ]
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_wikipedia_data(data_dir: str, num_pages: int = 10000) -> List[str]:
    """Load Wikipedia pages (mock implementation)"""
    # In a real implementation, this would load from Wikipedia dumps
    # For now, we'll create mock data
    
    logging.info(f"Loading {num_pages} Wikipedia pages from {data_dir}")
    
    # Mock Wikipedia pages
    mock_pages = []
    for i in range(min(num_pages, 100)):  # Limit for demo
        mock_page = f"""
        这是维基百科页面 {i+1} 的内容。这个页面包含了关于某个主题的详细信息。
        页面内容包括历史背景、主要特征、相关事件和重要人物。
        
        历史背景：
        这个主题有着悠久的历史，可以追溯到很久以前。在过去的几个世纪中，
        它经历了许多重要的发展和变化。
        
        主要特征：
        1. 特征一：这是一个重要的特征，对理解这个主题很关键。
        2. 特征二：另一个重要方面，与特征一相互关联。
        3. 特征三：第三个关键特征，提供了额外的见解。
        
        相关事件：
        - 事件A：发生在某年某月，对主题产生了重大影响。
        - 事件B：另一个重要事件，改变了发展轨迹。
        - 事件C：最近的发展，展示了当前状态。
        
        重要人物：
        张三：在这个领域做出了重要贡献的专家。
        李四：另一位关键人物，推动了相关研究。
        王五：当代的重要代表，继续推进发展。
        
        结论：
        这个主题在当今世界仍然具有重要意义，继续影响着相关领域的发展。
        """ * (i % 3 + 2)  # Vary length
        
        mock_pages.append(mock_page.strip())
    
    return mock_pages


def load_web_search_data(data_dir: str) -> List[Dict]:
    """Load web search data (mock implementation)"""
    # In a real implementation, this would load from web search results
    
    logging.info(f"Loading web search data from {data_dir}")
    
    # Mock web search data
    mock_data = []
    queries = [
        "今天的天气如何",
        "最新科技新闻",
        "股市行情分析", 
        "体育比赛结果",
        "电影推荐"
    ]
    
    for i, query in enumerate(queries * 20):  # Create multiple entries
        mock_data.append({
            "content": f"""
            关于"{query}"的搜索结果页面内容。
            
            这是一个包含相关信息的网页，提供了关于查询主题的详细信息。
            页面包含了最新的数据、分析和见解。
            
            主要内容：
            - 相关信息点1：提供了基础背景信息
            - 相关信息点2：包含了具体的数据和统计
            - 相关信息点3：给出了专家的分析和观点
            
            详细分析：
            根据最新的研究和数据，我们可以看到这个主题的几个重要趋势。
            首先，有一个明显的发展方向。其次，相关的影响因素也在变化。
            最后，未来的预期也值得关注。
            
            结论：
            基于当前的信息，我们可以得出一些重要的结论和建议。
            """,
            "query": query,
            "time": f"2025-01-{(i % 30) + 1:02d} 10:00:00",
            "location": "北京"
        })
    
    return mock_data


def generate_seed_data(
    config: Dict,
    wikipedia_pages: List[str],
    web_search_data: List[Dict],
    output_dir: str
) -> Dict[str, List]:
    """Generate seed QA data"""
    
    logging.info("Generating seed QA data...")
    
    # Initialize seed data generator
    seed_generator = SeedDataGenerator(
        model_name=config["model"]["base_model"],
        device="auto"
    )
    
    # Generate from Wikipedia
    wiki_samples = seed_generator.generate_from_wikipedia(
        wikipedia_pages=wikipedia_pages,
        **config["data_sources"]["wikipedia"]
    )
    
    # Generate from web search
    web_samples = seed_generator.generate_from_web_search(
        web_pages=web_search_data,
        **config["data_sources"]["web_search"]
    )
    
    # Save seed data
    os.makedirs(output_dir, exist_ok=True)
    
    seed_generator.save_samples(
        wiki_samples, 
        os.path.join(output_dir, "seed_wikipedia.json")
    )
    seed_generator.save_samples(
        web_samples,
        os.path.join(output_dir, "seed_web_search.json")
    )
    
    logging.info(f"Generated {len(wiki_samples)} Wikipedia samples and {len(web_samples)} web search samples")
    
    return {
        "wikipedia": wiki_samples,
        "web_search": web_samples
    }


def generate_distractor_data(
    config: Dict,
    seed_samples: List,
    output_dir: str
) -> List[Dict]:
    """Generate distractor resilience training data"""
    
    logging.info("Generating distractor resilience data...")
    
    # Initialize distractor generator
    distractor_generator = DistractorGenerator(
        model_name=config["model"]["base_model"],
        device="auto",
        **config["data_generation"]["distractor"]
    )
    
    # Generate distractors
    distractor_samples = []
    
    for sample in tqdm(seed_samples, desc="Generating distractors"):
        distractor = distractor_generator.generate_distractor(
            question=sample.question,
            answer=sample.answer,
            passage=sample.passage,
            user_time=sample.metadata.get("time_context") if sample.metadata else None,
            location=sample.metadata.get("location_context") if sample.metadata else None
        )
        
        if distractor:
            distractor_samples.append({
                "question": distractor.question,
                "answer": distractor.answer,
                "golden_passage": distractor.golden_passage,
                "distractor_passage": distractor.distractor_passage,
                "open_ended_question": distractor.open_ended_question,
                "user_time": distractor.user_time,
                "location": distractor.location
            })
    
    # Save distractor data
    output_path = os.path.join(output_dir, "distractor_training_data.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(distractor_samples, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Generated {len(distractor_samples)} distractor samples")
    return distractor_samples


def generate_strategic_cot_data(
    config: Dict,
    seed_samples: List,
    output_dir: str
) -> List[Dict]:
    """Generate strategic CoT training data"""
    
    logging.info("Generating strategic CoT data...")
    
    # Initialize strategic CoT generator
    cot_generator = StrategicCoTGenerator(
        model_name=config["model"]["base_model"],
        device="auto",
        **config["data_generation"]["strategic_cot"]
    )
    
    # Generate strategic CoT samples
    cot_samples = []
    
    for sample in tqdm(seed_samples, desc="Generating strategic CoT"):
        # Create references list (for now, just use the passage)
        references = [sample.passage]
        if sample.metadata and "all_references" in sample.metadata:
            references = sample.metadata["all_references"][:5]  # Limit to 5 references
        
        cot_sample = cot_generator.generate_strategic_cot(
            question=sample.question,
            references=references,
            ground_truth_answer=sample.answer,
            user_context=sample.metadata.get("time_context") if sample.metadata else None
        )
        
        if cot_sample:
            cot_samples.append({
                "question": cot_sample.question,
                "references": cot_sample.references,
                "strategy": cot_sample.strategy,
                "reasoning": cot_sample.reasoning,
                "answer": cot_sample.answer,
                "user_context": cot_sample.user_context
            })
    
    # Save strategic CoT data
    output_path = os.path.join(output_dir, "strategic_cot_training_data.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cot_samples, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Generated {len(cot_samples)} strategic CoT samples")
    return cot_samples


def main():
    parser = argparse.ArgumentParser(description="Generate PrismRAG training data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load raw data
    wikipedia_pages = load_wikipedia_data(
        args.data_dir,
        config["data_sources"]["wikipedia"]["num_pages"]
    )
    web_search_data = load_web_search_data(args.data_dir)
    
    # Generate seed data
    seed_data = generate_seed_data(
        config=config,
        wikipedia_pages=wikipedia_pages,
        web_search_data=web_search_data,
        output_dir=args.output_dir
    )
    
    # Combine seed samples for further processing
    all_seed_samples = seed_data["wikipedia"] + seed_data["web_search"]
    
    # Sample for distractor and CoT generation (to manage computational cost)
    distractor_seed_samples = random.sample(
        [s for s in all_seed_samples if s.source == "web_search"],
        min(50, len([s for s in all_seed_samples if s.source == "web_search"]))
    )
    
    cot_seed_samples = random.sample(
        all_seed_samples,
        min(100, len(all_seed_samples))
    )
    
    # Generate distractor data
    distractor_data = generate_distractor_data(
        config=config,
        seed_samples=distractor_seed_samples,
        output_dir=args.output_dir
    )
    
    # Generate strategic CoT data
    strategic_cot_data = generate_strategic_cot_data(
        config=config,
        seed_samples=cot_seed_samples,
        output_dir=args.output_dir
    )
    
    # Save final training data summary
    summary = {
        "total_seed_samples": len(all_seed_samples),
        "wikipedia_samples": len(seed_data["wikipedia"]),
        "web_search_samples": len(seed_data["web_search"]),
        "distractor_samples": len(distractor_data),
        "strategic_cot_samples": len(strategic_cot_data),
        "total_training_samples": len(distractor_data) + len(strategic_cot_data)
    }
    
    with open(os.path.join(args.output_dir, "data_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("Training data generation completed!")
    logging.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()