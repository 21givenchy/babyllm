# Integration Guide: BabyLLM Complete System

## 🔗 How Everything Connects

### 1. **InstinctModel** (Your Fast Understander)
- Trained on (text → tool) pairs
- Outputs: understanding_score + tool_selection
- Runs in <100ms on CPU
- Located: `python/models/instinct_model.py`

### 2. **ExplanationEngine with MoE** (Scalable Generator)
- Based on nanoBeard training architecture
- Uses Tutel for expert routing
- Routed through InstinctModel's tool selection
- Located: `python/training/moe_explanation_engine.py`

### 3. **MCP Tools** (Real-World Integration)
- 5 tools: Browser, Build, Camera, Sensor, LocalLLM
- Routed by InstinctModel's tool_idx
- Located: `python/tools/mcp_tools.py`

### 4. **Argilla** (Human Feedback Collection)
- Labels outputs: correct? quality? tool_right?
- Creates training data for self-improvement
- Located: `python/argilla_integration/`

### 5. **Self-Improvement Loop**
- Collects human feedback via Argilla
- Converts to training data format
- Retrains InstinctModel + ExplanationEngine
- A/B tests new version

## 📊 Data Flow Diagram

```
┌─────────────────────┐
│   User Input        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  InstinctModel.forward(text)            │
│  → understanding_score: float (0-1)     │
│  → tool_idx: int (0-4)                  │
│  → understanding: tensor                │
└─────────────────────┬───────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
    Score > 0.7              Score ≤ 0.7
          │                       │
          ▼                       ▼
   Use Tool             ExplanationEngine
   Directly            .forward(text, score)
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
          ┌──────────────────────┐
          │  MCP Tool Router     │
          │  (by tool_idx)       │
          └──────────┬───────────┘
                     │
     ┌───┬───┬───┬───┴───┬───┐
     │   │   │   │       │   │
     ▼   ▼   ▼   ▼       ▼   ▼
   [0] [1] [2] [3]     [4]
   BRO BUI CAM SEN  LOCALLLM
   WS  LD  ERA  SOR  PARTNER
     │   │   │   │       │
     └───┬───┬───┬───┬───┘
         │   │   │   │
         ▼   ▼   ▼   ▼
      TOOL RESULTS
         │   │   │
         └───┼───┘
             │
             ▼
      ┌─────────────────┐
      │  Output & Score │
      └────────┬────────┘
               │
               ▼
      ┌──────────────────────┐
      │  Argilla: Collect    │
      │  Human Feedback      │
      │  - correct? (bool)   │
      │  - quality (1-5)     │
      │  - best_tool (idx)   │
      └────────┬─────────────┘
               │
               ▼
      ┌──────────────────────┐
      │ Feedback Dataset     │
      │ (training/feedback/) │
      └────────┬─────────────┘
               │
               ▼
      ┌──────────────────────┐
      │ Retrain Models       │
      │ Stage 4: Autonomous  │
      │ Learning             │
      └──────────────────────┘
```

## 🏗️ Code Integration Points

### **Point 1: InstinctModel → Tool Selection**

```python
# In api/agent.py
from models.instinct_model import InstinctModel
from tools.mcp_tools import ToolUser

class Agent:
    def process(self, text_input: str):
        # Get understanding from Tier 1
        result = self.instinct_model(tokenize(text_input))
        score = result["score"].item()
        tool_idx = result["tool"].argmax().item()
        tool_name = self.instinct_model.get_tool_name(tool_idx)
        
        # Route by score
        if score > 0.7:
            # Direct tool use
            result = self.tool_user.use_tool(tool_name, text_input)
        else:
            # Call ExplanationEngine first
            explanation = self.explanation_engine.generate(
                text_input, score, tool_name
            )
            # Then use tool
            result = self.tool_user.use_tool(tool_name, text_input)
            return {"explanation": explanation, "tool_result": result}
        
        return result
```

### **Point 2: ExplanationEngine with MoE**

```python
# In api/agent.py
from training.moe_explanation_engine import MoEExplanationEngine

class Agent:
    def __init__(self):
        # Load MoE explanation engine
        self.explanation_engine = MoEExplanationEngine(
            vocab_size=8192,
            embed_dim=384,
            num_experts=8,
            experts_per_token=2,
        )
        checkpoint = torch.load("runs/moe_explanation/moe_explanation_ckpt.pth")
        self.explanation_engine.load_state_dict(checkpoint["model"])
        self.explanation_engine.eval()
    
    def generate_explanation(self, text: str, score: float, tool_name: str):
        """Generate explanation using MoE engine."""
        tokens = self.tokenizer.tokenize(text)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.explanation_engine(input_ids)
        
        # Sample next token
        next_token_logits = logits[0, -1, :]
        next_token = torch.multinomial(
            torch.softmax(next_token_logits / 0.8, -1), 1
        )
        
        return self.tokenizer.decode([next_token.item()])
```

### **Point 3: MCP Tool Routing**

```python
# In tools/mcp_tools.py
class ToolUser:
    def use_tool(self, tool_name: str, query: str):
        """Route by tool_idx from InstinctModel."""
        tools = {
            "browser": self.use_browser,
            "build": self.use_build,
            "camera": self.use_camera,
            "sensor": self.use_sensor,
            "local_llm": self.use_local_llm,
        }
        return tools[tool_name](query)
```

### **Point 4: Argilla Feedback Collection**

```python
# In argilla_integration/collect_feedback.py
import argilla as rg

def collect_user_feedback(output: str, metadata: dict):
    """Collect human feedback via Argilla."""
    client = rg.Argilla(
        api_url="http://localhost:6900",
        api_key="owner.apikey"
    )
    
    dataset = rg.Dataset(name="babyllm_feedback", client=client)
    
    # Create record for labeling
    record = rg.Record(
        text=output,
        metadata=metadata,
        status="pending",
    )
    
    dataset.records.log([record])
    
    # Human labels: correct?, quality (1-5), best_tool?
    # → Returns labeled feedback
    return dataset.records.suggestions
```

### **Point 5: Self-Improvement Loop**

```python
# In training/train_from_feedback.py
def train_from_feedback():
    """Retrain InstinctModel using human feedback."""
    
    # 1. Collect feedback from Argilla
    feedback_data = collect_argilla_feedback()
    
    # 2. Convert to training format
    train_texts = [f["output"] for f in feedback_data]
    train_labels = [f["best_tool_idx"] for f in feedback_data]
    
    # 3. Retrain Tier 1 (InstinctModel)
    trainer = InstinctTrainer()
    trainer.stage4_autonomous(texts=train_texts, actions=train_labels)
    trainer.save_checkpoint("instinct_agent_v2.pth")
    
    # 4. A/B test: old vs new version
    score_old = evaluate_on_test_set(old_model)
    score_new = evaluate_on_test_set(new_model)
    
    if score_new > score_old:
        # Deploy new version
        deploy(new_model)
    else:
        # Keep old version
        rollback()
```

## 🚀 Setup Instructions

### 1. Install Everything

```bash
# BabyLLM
pip install -r python/requirements.txt

# Tutel (MoE)
pip install -U git+https://github.com/microsoft/tutel@main

# Argilla (feedback)
pip install argilla

# WaveML (tracking)
pip install wandb
```

### 2. Start Services

```bash
# Terminal 1: BabyLLM API
cd python
python api/agent.py  # Runs on http://localhost:8000

# Terminal 2: Argilla (feedback collection)
argilla server --port 6900

# Terminal 3: LocalLLM partner (optional)
python -m ollama run llama2
```

### 3. Train Models

```bash
# Train Tier 1 (InstinctModel)
cd python
python training/train.py

# Train Tier 2 (ExplanationEngine with MoE)
python training/train_moe_explanation.py
```

### 4. Collect Feedback

```bash
# Run inference, send to Argilla
python scripts/collect_feedback.py

# Human labels the outputs
# → Generates training data
```

### 5. Self-Improve

```bash
# Retrain on feedback
python training/train_from_feedback.py

# A/B test new version
python scripts/ab_test.py

# Deploy if better
python scripts/deploy.py
```

## 🔄 Self-Improvement Cycle

```
Day 1:
  └─ Train initial models (Tier 1 + 2)
  └─ Deploy to users

Day 2-7:
  └─ Users interact with system
  └─ Argilla collects feedback
  └─ ~100 labeled examples per day

Day 8:
  └─ Retrain on feedback (~700 examples)
  └─ A/B test: old vs new
  └─ Deploy if better
  └─ Metrics improve: +5-10% accuracy

Day 9-14:
  └─ Repeat cycle
  └─ Metrics compound: +10-20% total improvement

Day 15+:
  └─ Model converges on domain
  └─ Human feedback becomes low-frequency
  └─ Fully autonomous learning (Stage 4)
```

## 📈 Metrics to Track

### **InstinctModel (Tier 1)**
- Accuracy: % correct tool selection
- Latency: inference time (<100ms)
- Understanding score: calibration (actual vs predicted)

### **ExplanationEngine (Tier 2)**
- BLEU/ROUGE: explanation quality
- Perplexity: language modeling performance
- MoE expert utilization: balance across experts

### **Overall System**
- User satisfaction: Argilla ratings (1-5)
- Tool accuracy: % correct tool used
- End-to-end latency: full pipeline time
- Self-improvement rate: metric improvement per cycle

## 🎯 Example Workflow

```python
# User query
query = "I want to build a robot"

# 1. InstinctModel scores & routes
scores = instinct_model(query)
# {"score": 0.75, "tool_idx": 2}  # tool_idx=2 is "build"

# 2. Since score > 0.7, use tool directly
result = mcp_tools.use_build(query)
# "Execute build command for robot project"

# 3. Send to user + Argilla for feedback
user_feedback = argilla.collect(result, metadata={...})
# {"correct": true, "quality": 5, "best_tool": 2}

# 4. Add to training data
training_data.append((query, tool_idx=2, feedback=5))

# 5. Weekly retraining
train_from_feedback(training_data)
```

## ⚙️ Configuration Files

- `python/training/train.py`: InstinctModel config
- `python/training/train_moe_explanation.py`: MoE engine config
- `python/api/agent.py`: API server config
- `argilla_integration/config.yaml`: Argilla config
- `.env`: Environment variables (API keys, paths)

## 🔐 Security

- API key required for all endpoints
- Human feedback validated before training
- Model checkpoints signed + versioned
- A/B tests require approval before deployment

## 📚 Further Reading

- [nanoBeard Training](../nanobeard/README.md)
- [Tutel MoE Paper](https://arxiv.org/pdf/2206.03382.pdf)
- [Argilla Docs](https://docs.argilla.io/)
- [MCP Protocol](https://modelcontextprotocol.io/)
