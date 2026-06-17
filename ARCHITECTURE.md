# BabyLLM: Complete Self-Improving LLM Architecture

> Combining **Instinct Understanding** + **MoE Scaling** + **Self-Supervised Learning** + **Human Feedback**

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│         TIER 1: INSTINCT MODEL (Understanding)                 │
│  • Input meaning scoring (0-1 confidence)                      │
│  • Tool selection (5 MCP tools)                                │
│  • ~1M params, runs on CPU                                     │
└──────────────────┬─────────────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
    Score > 0.7        Score < 0.7
          │                 │
          ▼                 ▼
    Use Tool         ┌──────────────────┐
    Directly    │ TIER 2: EXPLANATION  │
                │    ENGINE (MoE)      │
                │                      │
                │ • Local LLM          │
                │ • 13.8M params       │
                │ • Tutel MoE routing  │
                │ • Self-supervised    │
                └──────────┬───────────┘
                           │
                           ▼
         ┌─────────────────────────────┐
         │   TIER 3: MCP TOOLS         │
         │                             │
         │ • Browser (web search)      │
         │ • Build (code execution)    │
         │ • Camera (image capture)    │
         │ • Sensor (data reading)     │
         │ • Local LLM (explanation)   │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  OUTPUT & FEEDBACK   │
         │                      │
         │ • Human feedback     │
         │ • Argilla labeling   │
         │ • Self-improvement   │
         └──────────────────────┘
```

## 📊 Three-Tier System

### **Tier 1: InstinctModel** (Fast Understanding)
- **Purpose**: Understand input meaning + select tool
- **Size**: ~1M params
- **Inference**: CPU, <100ms
- **Training**: Supervised on (text, tool) pairs
- **Output**: understanding_score (0-1), tool_idx (0-4)

### **Tier 2: Explanation Engine** (Accurate Generation)
- **Purpose**: Generate explanations + reasoning
- **Size**: 13.8M params (nanoBeard-sized)
- **Architecture**: MoE-enhanced GPT with Tutel
- **Inference**: GPU or CPU, <1s for 100 tokens
- **Training**: SFT on instruction pairs + self-supervised
- **Output**: Natural language explanation

### **Tier 3: Tool Layer** (Real-World Integration)
- **Purpose**: Execute actions and fetch data
- **Protocol**: MCP (Model Context Protocol)
- **Tools**: Browser, Build, Camera, Sensor, LocalLLM
- **Feedback Loop**: Human annotations → Argilla → Training data

## 🧠 Self-Improvement Loop

```
1. INFERENCE
   Input → InstinctModel → Understanding Score & Tool
           ↓
2. TOOL USE
   Score < 0.7 → ExplanationEngine (Tutel MoE)
                 ↓
3. MCP EXECUTION
   Tool → Real-world data (browser, build, camera, sensor)
          ↓
4. HUMAN FEEDBACK
   Output → Human → Argilla (labeling platform)
                    ↓
5. SELF-IMPROVEMENT
   Feedback → Training data → Retrain InstinctModel + ExplanationEngine
   (Stage 4 Autonomous Learning)
```

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|----------|
| **Training** | nanoBeard (PyTorch) | Modular training pipeline |
| **Scaling** | Tutel MoE | Mixture-of-Experts routing |
| **GPU Support** | ROCm-LLMExt | AMD GPU compatibility |
| **Distributed** | Torch DDP + Tutel | Multi-GPU training |
| **Data Feedback** | Argilla | Annotation & labeling |
| **Tool Integration** | MCP | Standardized tool calls |
| **API** | FastAPI | Inference server |
| **Orchestration** | Weft | Visual workflows |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd python
pip install -r requirements.txt
```

### 2. Train InstinctModel
```bash
python training/train.py
# Runs 4 stages: Exploration → Association → Interaction → Autonomous
```

### 3. Train ExplanationEngine (with MoE)
```bash
python training/train_moe_explanation.py
# Uses Tutel MoE routing + nanoBeard architecture
```

### 4. Start API Server
```bash
python api/agent.py
# Serves on http://localhost:8000
```

### 5. Collect Human Feedback
```bash
# Use Argilla to label outputs
python argilla_integration/collect_feedback.py
```

### 6. Self-Improve
```bash
# Retrain on human feedback
python training/train_from_feedback.py
```

## 🎯 Key Features

### **Understanding-Based Routing**
```python
score = instinct_model(text)  # 0-1 confidence
if score > 0.7:
    return use_tool_directly()
else:
    return explanation_engine(text)  # Call Tier 2
```

### **Mixture-of-Experts Scaling**
```python
# ExplanationEngine uses Tutel MoE
moe_layer = tutel_moe.moe_layer(
    gate_type={'type': 'top', 'k': 2},  # Top-2 expert routing
    model_dim=384,
    experts={
        'num_experts_per_device': 4,
        'type': 'ffn',
        'hidden_size_per_expert': 1024
    }
)
```

### **Self-Supervised Learning**
- **Stage 4 (Autonomous)**: Model learns from its own observations
- **Maximize understanding_score**: Self-improvement without labels
- **MoE routing optimization**: Experts specialize through gradient flow

### **Human-in-the-Loop**
```python
# Argilla integration
from argilla_integration import collect_feedback

feedback = collect_feedback(output, metadata)
# Human labels: correct? (yes/no), explanation quality (1-5), tool correct? (yes/no)
# → Training data for next iteration
```

## 📈 Scaling Path

| Stage | InstinctModel | ExplanationEngine | MoE Experts |
|-------|---------------|-------------------|-------------|
| **v1 (Current)** | 1M params | 13.8M params | 8 experts |
| **v2 (MoE)** | 1M params | 13.8M params | 32 experts |
| **v3 (Multi-Node)** | 1M params | 100M params | 256 experts |
| **v4 (Production)** | 5M params | 1B params | 1000+ experts |

## 🔌 MCP Tool Integration

```python
from mcp_tools import ToolUser

tools = ToolUser()

# Browser tool: search web
result = tools.use_browser("machine learning")

# Build tool: execute code
result = tools.use_build("python code")

# Camera tool: capture image
image = tools.use_camera()

# Sensor tool: read IMU/GPS
data = tools.use_sensor("temperature")

# Local LLM: ask explanation partner
exp = tools.use_local_llm("What does this mean?")
```

## 💾 Data Flow for Self-Improvement

```
Human Feedback (Argilla)
         ↓
    Validate quality
         ↓
    Extract features
         ↓
    Format as training data
         ↓
    Retrain InstinctModel + ExplanationEngine
         ↓
    Update checkpoint
         ↓
    Deploy new version
         ↓
    A/B test with users
```

## 🎓 Why This Design?

1. **Fast + Accurate**: Two-tier system for speed vs. quality
2. **Scalable**: MoE allows scaling without dense parameters
3. **Self-improving**: Feedback loop automates improvement
4. **Efficient**: InstinctModel filters unnecessary generation
5. **Interpretable**: Understand why tools are selected
6. **Flexible**: Swap tools, models, or experts without retraining entire system

## 📚 References

- **nanoBeard**: Training architecture from `younissk/nanoBeard`
- **Tutel MoE**: Expert routing from `microsoft/Tutel`
- **ROCm-LLMExt**: GPU support from `ROCm/ROCm-LLMExt`
- **Argilla**: Data labeling from `argilla-io/argilla`
- **MCP**: Tool protocol from Anthropic
- **Weft**: Orchestration from `WeaveMindAI/weft`

## 🚢 Deployment

### Local
```bash
./dev.sh server  # BabyLLM API
./dev.sh argilla # Feedback collection
```

### Docker
```bash
docker-compose up
# Starts: BabyLLM API, Argilla, LocalLLM partner
```

### Distributed (Multi-GPU)
```bash
python -m torch.distributed.run --nproc_per_node=8 \
  training/train_moe_explanation.py
```

## 🎯 Next Steps

1. ✅ Implement Tier 1 (InstinctModel) — DONE
2. ✅ Implement Tier 2 (ExplanationEngine with MoE) — IN PROGRESS
3. ⬜ Integrate Argilla for feedback collection
4. ⬜ Build self-improvement loop
5. ⬜ Deploy to multi-GPU cluster
6. ⬜ A/B test with real users
7. ⬜ Scale to 1B+ parameters

---

**Philosophy**: Build a system that learns like a human baby — observing, understanding, getting feedback, and improving itself over time.
