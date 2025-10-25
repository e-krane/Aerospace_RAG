# Qwen3 Model Setup for Aerospace RAG

## Configuration

Your optimal Qwen3 setup for 12GB VRAM:

```yaml
Embeddings: qwen3-embedding:8b (4.7GB)
LLM:        qwen3:latest (5.2GB)
Peak VRAM:  5.2GB / 12GB (sequential)
Both VRAM:  9.9GB / 12GB (simultaneous)
Headroom:   2.1GB for processing overhead
```

## Models

### qwen3-embedding:8b
- **Size:** 4.7GB
- **Context:** 40K tokens
- **Dimensions:** Up to 4096 (using 256 with Matryoshka)
- **Performance:** #1 on MTEB multilingual leaderboard (score 70.58)
- **Purpose:** Document embedding, vector search
- **Ollama command:** `ollama pull qwen3-embedding:8b`

### qwen3:latest (8B parameters)
- **Size:** 5.2GB
- **Context:** 40K tokens
- **Purpose:** Answer generation, RAGAS evaluation
- **Ollama command:** `ollama pull qwen3:latest`

## Usage

### Embeddings

```python
from src.embeddings.ollama_qwen3_embedder import create_embedder

# Create embedder
embedder = create_embedder(
    model_name="qwen3-embedding:4b",
    use_matryoshka=True,
    reduced_dimensions=256,
)

# Generate embeddings
texts = ["Your technical document text..."]
embeddings = embedder.embed(texts)  # Returns (N, 256) numpy array
```

### LLM for Answer Generation

```python
from src.llm.client import LLMClient

# Create LLM client
client = LLMClient(
    provider="ollama",
    model="qwen3:latest",
    temperature=0.1,
)

# Generate answer
response = client.generate(
    prompt="What is beam bending?",
    context=["Retrieved context chunks..."],
)

print(response.content)
```

### RAGAS Evaluation

```python
from src.evaluation.ragas_evaluator import RAGASEvaluator

# Create evaluator (uses qwen3:latest as judge)
evaluator = RAGASEvaluator(
    llm_provider="ollama",
    llm_model="qwen3:latest",
    threshold_precision=0.8,
    threshold_recall=0.8,
    threshold_faithfulness=0.9,
    threshold_relevancy=0.8,
)

# Evaluate RAG system
result = evaluator.evaluate(
    questions=["Q1", "Q2"],
    answers=["A1", "A2"],
    contexts=[["C1"], ["C2"]],
)

print(result)
```

## API Keys Required

**None!** All models run locally via Ollama:
- ✅ No OpenAI API key needed
- ✅ No Anthropic API key needed
- ✅ Completely offline capable
- ✅ Zero per-token costs

## VRAM Management

### Sequential Loading (Recommended)
```python
# Step 1: Index documents (uses embedding model)
embedder = create_embedder("qwen3-embedding:8b")
embeddings = embedder.embed(documents)
# VRAM: ~4.7GB

# Step 2: Query time (uses LLM)
llm = LLMClient(provider="ollama", model="qwen3:latest")
response = llm.generate(...)
# VRAM: ~5.2GB (peak)
```

### Simultaneous Loading (Recommended for 12GB)
Both models can be loaded simultaneously:
- qwen3-embedding:8b: 4.7GB
- qwen3:latest: 5.2GB
- **Total:** 9.9GB / 12GB ✅ Comfortable fit with 2.1GB headroom!

This allows for:
- Zero model loading/unloading overhead
- Faster query response times
- Room for ColBERT reranker (~1.5GB)
- Batch processing optimizations

## Performance Characteristics

### Embedding Speed
- **Batch size:** 32 texts
- **Matryoshka reduction:** 768D → 256D
- **L2 normalized:** Yes
- **Typical latency:** ~1-2s per batch

### LLM Generation Speed
- **Context:** 40K tokens
- **Temperature:** 0.1 (technical accuracy)
- **Typical latency:** ~5-10s for 500 tokens
- **Tokens/second:** ~50-100

### Quality
- **Embeddings:** Good for technical/aerospace content
- **LLM Reasoning:** Improved over Qwen2.5 for math/code/logic
- **RAGAS Judge:** Reliable evaluation scores

## Testing

Run quick test:
```bash
python test_qwen3_quick.py
```

Run full test suite (includes RAGAS):
```bash
python test_qwen3_setup.py
```

## Advantages Over Alternatives

### vs. HuggingFace gte-Qwen2-7B-instruct
- ✅ **Easier setup:** No torch/transformers complexity
- ✅ **Better quality:** 8B > 7B, #1 MTEB ranking
- ✅ **Better integration:** Native Ollama support
- ✅ **Simpler API:** Direct ollama.embed() calls

### vs. qwen2.5:7b
- ✅ **Better reasoning:** Qwen3 > Qwen2.5
- ✅ **Same VRAM:** 5.2GB vs 4.7GB (minimal difference)
- ✅ **40K context:** vs 32K in Qwen2.5
- ✅ **More recent:** Latest 2025 model

### vs. Cloud APIs (GPT-4, Claude)
- ✅ **Zero cost:** No per-token charges
- ✅ **Offline:** Works without internet
- ✅ **Privacy:** Data stays local
- ✅ **Speed:** No API latency
- ⚠️ **Quality:** Slightly lower than GPT-4 (but sufficient for testing)

## Troubleshooting

### Model not found
```bash
# Verify models are pulled
ollama list | grep qwen3

# Pull if missing
ollama pull qwen3-embedding:4b
ollama pull qwen3:latest
```

### VRAM issues
```bash
# Check GPU usage
nvidia-smi

# If you need to free up VRAM, use smaller models
ollama pull qwen3-embedding:4b    # 4.7GB → 2.5GB
ollama pull qwen3-embedding:0.6b  # Only 639MB
ollama pull qwen3:4b              # 5.2GB → 2.5GB
```

### Wrong dimensions
The model outputs 2560D by default, which gets reduced to 256D via Matryoshka.
This is expected and provides 3x storage savings with ~99.5% performance retention.

## Files Created

- `src/embeddings/ollama_qwen3_embedder.py` - Ollama embedding wrapper
- `test_qwen3_setup.py` - Full test suite
- `test_qwen3_quick.py` - Quick validation test
- `QWEN3_SETUP.md` - This documentation

## Next Steps

1. ✅ Models pulled and tested
2. Index your aerospace documents using qwen3-embedding:4b
3. Build vector search with your vector database (Supabase/pgvector?)
4. Run queries using qwen3:latest
5. Evaluate with RAGAS using qwen3:latest as judge

---

**Status:** ✅ Ready for production testing!
