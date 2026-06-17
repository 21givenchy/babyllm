"""Convert Argilla feedback into training data."""

import json
from typing import List, Tuple
from feedback_collector import ArguillaFeedbackCollector


class TrainingDataBuilder:
    """Converts human feedback into training data format."""
    
    TOOL_TO_IDX = {
        "local_llm": 0,
        "browser": 1,
        "build": 2,
        "camera": 3,
        "sensor": 4,
    }
    
    def __init__(self, api_url: str = "http://localhost:6900"):
        self.collector = ArguillaFeedbackCollector(api_url=api_url)
    
    def build_training_data(self, min_quality: int = 3) -> Tuple[List[str], List[int]]:
        """Convert feedback to (text, tool_idx) pairs.
        
        Args:
            min_quality: Only include feedback with quality >= min_quality
            
        Returns:
            (texts, actions) for training InstinctModel
        """
        labeled_records = self.collector.get_labeled_feedback()
        
        texts = []
        actions = []
        
        for record in labeled_records:
            # Check quality threshold
            quality = record["feedback"].get("quality", 1)
            if quality < min_quality:
                continue
            
            # Check if output was correct
            correct = record["feedback"].get("correct") == "yes"
            if not correct:
                continue
            
            # Get best tool(s)
            best_tools = record["feedback"].get("best_tool", [])
            if not best_tools:
                continue
            
            # Use first best tool
            best_tool = best_tools[0] if isinstance(best_tools, list) else best_tools
            tool_idx = self.TOOL_TO_IDX.get(best_tool, 0)
            
            texts.append(record["input"])
            actions.append(tool_idx)
        
        return texts, actions
    
    def save_training_data(self, output_path: str):
        """Save training data to JSON file."""
        texts, actions = self.build_training_data()
        
        data = {
            "texts": texts,
            "actions": actions,
            "num_samples": len(texts),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(texts)} training samples to {output_path}")
        return texts, actions


if __name__ == "__main__":
    builder = TrainingDataBuilder()
    texts, actions = builder.save_training_data("data/feedback_training.json")
    print(f"\nTraining data: {len(texts)} samples")
    print(f"Sample: {texts[0][:50]}... → tool {actions[0]}")
