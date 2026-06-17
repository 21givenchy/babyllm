# 🏴‍☠️ BabyLLM: Self-Improving Pirate LLM Agent

> A tiny LLM that learns like a human baby — observing, understanding, getting feedback, and improving itself over time.

## 🎯 What is BabyLLM?

BabyLLM is a **three-tier system** that combines:

1. **Tier 1: InstinctModel** (~1M params)
   - Fast understanding + tool selection
   - Scores confidence (0-1)
   - Runs on CPU in <100ms

2. **Tier 2: ExplanationEngine with MoE** (13.8M params)
   - Accurate text generation
   - Mixture-of-Experts for scaling
   - Routed by Tier 1

3. **Tier 3: MCP Tools** 
   - Browser (web search)
   - Build (code execution)
   - Camera (image capture)
   - Sensor (data reading)
   - LocalLLM (explanation partner)

## 🚀 Quick Start

```bash
# 1. Install
cd python
pip install -r requirements.txt

# 2. Train Tier 1 (InstinctModel)
python training/train.py

# 3. Train Tier 2 (ExplanationEngine with MoE)
python training/train_moe_explanation.py

# 4. Start API
python api/agent.py
# Opens http://localhost:8000/docs

# 5. Collect feedback
python argilla_integration/feedback_collector.py

# 6. Self-improve
python training/train_from_feedback.py
```

## 📊 Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

```
User Input
    ↓
InstinctModel (understand + route)
    ├─ score > 0.7 → Use tool directly
    └─ score ≤ 0.7 → ExplanationEngine + Tool
        ↓
ExplanationEngine (MoE routing)
    ↓
MCP Tools (browser, build, camera, sensor, local_llm)
    ↓
Argilla (collect human feedback)
    ↓
Retrain (Stage 4: Autonomous Learning)
```

## 🔗 Integration Guide

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for:
- How Tier 1 → Tier 2 → Tier 3 connect
- Code integration points
- Self-improvement loop
- Deployment instructions

## 📦 Repository Structure

```
python/
├── models/
│   ├── instinct_model.py          # Tier 1: Fast understanding
│   └── meaning_explainer.py       # TinyLlama partner
├── training/
│   ├── train.py                   # Train Tier 1 (4 stages)
│   ├── moe_explanation_engine.py  # Tier 2 with Tutel MoE
│   ├── train_moe_explanation.py   # Train Tier 2
│   └── train_from_feedback.py     # Self-improvement
├── api/
│   └── agent.py                   # FastAPI server
├── tools/
│   └── mcp_tools.py               # MCP tool wrappers
├── argilla_integration/
│   ├── feedback_collector.py      # Argilla integration
│   └── training_data_builder.py   # Convert feedback → training data
└── requirements.txt
```

## 🏗️ Technology Stack

- **Training**: nanoBeard (PyTorch)
- **Scaling**: Tutel MoE (Mixture-of-Experts)
- **GPU**: ROCm-LLMExt (AMD GPU support)
- **Feedback**: Argilla (data labeling)
- **Tools**: MCP (Model Context Protocol)
- **Orchestration**: Weft (visual workflows)
- **API**: FastAPI (inference server)

## 📈 Self-Improvement Cycle

```
Day 1:     Train initial models
Day 2-7:   Users interact, Argilla collects feedback (~100/day)
Day 8:     Retrain on feedback, A/B test, deploy if better
Day 9-14:  Repeat, +10-20% improvement total
Day 15+:   Model converges, fully autonomous (Stage 4)
```

## 🎓 Three-Stage Learning

### Stage 1: Exploration (Week 1)
- Baby observes training data
- No understanding yet
- Collects diverse observations

### Stage 2: Association (Week 1-2)
- Baby connects inputs to meanings
- Understanding score improves
- Begins tool selection

### Stage 3: Interaction (Week 2-3)
- Baby learns from human corrections
- Accuracy increases
- Better tool selection

### Stage 4: Autonomous (Week 3+)
- Baby improves on its own
- Self-supervised learning
- Fully self-improving

## 🔄 Feedback Loop

```python
# User query
query = "I want to build a robot"

# 1. InstinctModel scores
score = 0.85  # High confidence
tool = "build"

# 2. Use tool directly
result = tools.use_build(query)

# 3. Argilla collects feedback
feedback = {
    "correct": True,
    "quality": 5,
    "best_tool": "build"
}

# 4. Add to training data
training_data.append((query, tool="build", feedback=5))

# 5. Retrain weekly
train_from_feedback(training_data)
```

## 🚢 Deployment

### Local
```bash
# Terminal 1: API
python api/agent.py

# Terminal 2: Argilla (feedback)
argilla server --port 6900
```

### Docker
```bash
docker-compose up
```

### Distributed (Multi-GPU)
```bash
python -m torch.distributed.run --nproc_per_node=8 \
  training/train_moe_explanation.py
```

## 📊 Metrics

- **InstinctModel**: Tool selection accuracy
- **ExplanationEngine**: BLEU/ROUGE (explanation quality)
- **System**: User satisfaction (Argilla ratings)
- **Learning**: Metric improvement per week

## 🎯 Next Steps

- [ ] Deploy to production
- [ ] Collect real user feedback
- [ ] Retrain weekly
- [ ] Scale to 100M+ parameters
- [ ] Multi-modal (vision + text)
- [ ] Multi-language support
- [ ] Real-time feedback loop

## 📚 References

- [nanoBeard](https://github.com/younissk/nanoBeard) — Training architecture
- [Tutel MoE](https://github.com/microsoft/Tutel) — Expert routing
- [ROCm-LLMExt](https://github.com/ROCm/ROCm-LLMExt) — GPU support
- [Argilla](https://github.com/argilla-io/argilla) — Data labeling
- [MCP](https://modelcontextprotocol.io/) — Tool protocol
- [Weft](https://github.com/WeaveMindAI/weft) — Orchestration

## 🏴‍☠️ Philosophy

> *"A baby doesn't come into the world knowing everything. It learns through observation, interaction, and feedback. BabyLLM works the same way — it starts small, learns from experience, and improves itself over time."*

## 📄 License

MIT — Free to use, modify, and distribute.

---

**Built with 🏴‍☠️ by the BabyLLM team**
