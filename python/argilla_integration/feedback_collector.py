"""Collect human feedback via Argilla for self-improvement."""

import argilla as rg
from typing import Optional, Dict, Any


class ArguillaFeedbackCollector:
    """Collects human feedback using Argilla."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:6900",
        api_key: str = "owner.apikey",
    ):
        self.client = rg.Argilla(api_url=api_url, api_key=api_key)
    
    def create_dataset(self, name: str = "babyllm_feedback"):
        """Create Argilla dataset for labeling."""
        settings = rg.Settings(
            guidelines="Label the model output quality and tool correctness.",
            fields=[
                rg.TextField(
                    name="input_text",
                    title="User Input",
                    use_markdown=False,
                ),
                rg.TextField(
                    name="output_text",
                    title="Model Output",
                    use_markdown=True,
                ),
                rg.TextField(
                    name="tool_used",
                    title="Tool Used",
                    use_markdown=False,
                ),
            ],
            questions=[
                rg.LabelQuestion(
                    name="correct",
                    title="Is the output correct?",
                    labels=["yes", "no"],
                ),
                rg.RatingQuestion(
                    name="quality",
                    title="Explanation quality (1-5)",
                    values=[1, 2, 3, 4, 5],
                ),
                rg.MultiLabelQuestion(
                    name="best_tool",
                    title="Which tool(s) would be best?",
                    options=[
                        "browser",
                        "build",
                        "camera",
                        "sensor",
                        "local_llm",
                    ],
                ),
            ],
        )
        
        dataset = rg.Dataset(
            name=name,
            settings=settings,
            client=self.client,
        )
        dataset.create()
        return dataset
    
    def log_output_for_feedback(
        self,
        input_text: str,
        output_text: str,
        tool_used: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log output to Argilla for human feedback.
        
        Args:
            input_text: Original user input
            output_text: Model output
            tool_used: Which tool was selected
            metadata: Optional metadata (understanding_score, etc.)
            
        Returns:
            Record ID in Argilla
        """
        dataset = self.client.datasets.get("babyllm_feedback")
        
        record = rg.Record(
            fields={
                "input_text": input_text,
                "output_text": output_text,
                "tool_used": tool_used,
            },
            metadata=metadata or {},
        )
        
        dataset.records.log([record])
        return record.id
    
    def get_labeled_feedback(self) -> list:
        """Get all labeled feedback from Argilla.
        
        Returns:
            List of labeled records with human feedback
        """
        dataset = self.client.datasets.get("babyllm_feedback")
        
        labeled_records = []
        for record in dataset.records.filter(status="submitted"):
            # Extract responses
            responses = {}
            for response in record.responses:
                responses[response.question.name] = response.value
            
            labeled_records.append({
                "id": record.id,
                "input": record.fields["input_text"],
                "output": record.fields["output_text"],
                "tool_used": record.fields["tool_used"],
                "feedback": responses,
                "metadata": record.metadata,
            })
        
        return labeled_records


def collect_feedback(
    input_text: str,
    output_text: str,
    tool_used: str,
    api_url: str = "http://localhost:6900",
    api_key: str = "owner.apikey",
) -> str:
    """Convenience function to log output for feedback."""
    collector = ArguillaFeedbackCollector(api_url=api_url, api_key=api_key)
    return collector.log_output_for_feedback(
        input_text=input_text,
        output_text=output_text,
        tool_used=tool_used,
    )
