# 12GB VRAM Capabilities

**GPU:** 12GB VRAM available
**Current Setup:** Optimized for parallel model loading

---

## Current Configuration ✅

```yaml
Embeddings:  qwen3-embedding:8b (4.7GB)
LLM:         qwen3:latest (5.2GB)
Total:       9.9GB / 12GB
Headroom:    2.1GB (17%)
```

**Status:** Optimal for production use

---

## What You Can Run Simultaneously

### **Option 1: Current Stack (Recommended)** ⭐

```
qwen3-embedding:8b:  4.7GB
qwen3:latest:        5.2GB
Processing overhead: ~0.5GB
Total:               10.2GB / 12GB
Remaining:           1.8GB
```

**Use remaining VRAM for:**
- Batch processing buffers
- Cache layers
- Multiple concurrent queries

---

### **Option 2: With ColBERT Reranker**

```
qwen3-embedding:8b:  4.7GB
qwen3:latest:        5.2GB
ColBERT v2:          1.5GB
Total:               11.2GB / 12GB
Remaining:           0.8GB ✅ Tight but workable
```

**Benefits:**
- Complete RAG pipeline in memory
- No model loading between steps
- Sub-2s end-to-end latency

---

### **Option 3: Larger LLM Alternative**

```
qwen3-embedding:8b:  4.7GB
qwen2.5:14b:         8.9GB
Total:               13.6GB / 12GB ❌ Too large
```

**Alternative with smaller embedding:**
```
qwen3-embedding:4b:  2.5GB
qwen2.5:14b:         8.9GB
Total:               11.4GB / 12GB ✅ Fits!
```

**Trade-off:**
- Better LLM reasoning (14B > 8B)
- Worse embeddings (4B < 8B)
- Not recommended - embedding quality matters more

---

## Recommended Parallel Pipeline

### **Full RAG Pipeline in Memory**

```python
# Load all models once at startup
embedder = OllamaQwen3Embedder("qwen3-embedding:8b")    # 4.7GB
llm = LLMClient(provider="ollama", model="qwen3:latest") # 5.2GB
reranker = ColBERTReranker()                             # 1.5GB (optional)

# VRAM: 11.2GB / 12GB ✅

# Query flow (all in memory, no loading):
1. Embed query                    # Uses embedder
2. Hybrid search (BM25 + vector)  # Qdrant
3. Rerank top-100 → top-10        # Uses reranker
4. Generate answer                # Uses LLM
5. Return with citations

# Latency: <2s end-to-end
```

---

## Batch Processing Optimizations

With 2.1GB headroom, you can:

### **Larger Batch Sizes**

```python
# Embedding generation
embedder = OllamaQwen3Embedder(
    model_name="qwen3-embedding:8b",
    batch_size=64,  # Up from default 32
)

# Faster document indexing:
# - 1000 chunks: ~30s → ~20s
# - 10000 chunks: ~5min → ~3.5min
```

### **Concurrent Query Processing**

```python
# Handle multiple queries simultaneously
import asyncio

async def process_queries(queries):
    tasks = [rag_pipeline.query(q) for q in queries]
    return await asyncio.gather(*tasks)

# Process 5 queries in parallel
# Total VRAM: 9.9GB (models) + ~1GB (5 query buffers) = 10.9GB ✅
```

---

## Memory Allocation Strategy

### **Optimal Distribution**

```
Models (persistent):     9.9GB  (83%)
Query buffers:           1.0GB  (8%)
Cache layers:            0.5GB  (4%)
OS/overhead:             0.6GB  (5%)
Total:                  12.0GB  (100%)
```

### **Conservative (Safer)**

```
Models (persistent):     9.9GB  (83%)
Processing buffer:       1.5GB  (12%)
Safety margin:           0.6GB  (5%)
Total:                  12.0GB  (100%)
```

---

## Future Upgrade Paths

### **If You Get More VRAM (16GB+)**

```
Option A: Larger LLM
  qwen3-embedding:8b + qwen2.5:14b
  = 4.7GB + 8.9GB = 13.6GB ✅

Option B: Multiple Specialized LLMs
  qwen3-embedding:8b + qwen3:latest + qwen3-coder:8b
  = 4.7GB + 5.2GB + 5.2GB = 15.1GB ✅

Option C: Full-size ColBERT
  Current stack + ColBERT-full (3GB)
  = 9.9GB + 3GB = 12.9GB ✅
```

### **If You Need to Reduce VRAM**

```
Fallback to 4b embedding:
  qwen3-embedding:4b + qwen3:latest
  = 2.5GB + 5.2GB = 7.7GB
  Frees: 2.2GB
```

---

## Monitoring VRAM Usage

### **Check Current Usage**

```bash
# Real-time monitoring
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Or use watch
watch -n 1 nvidia-smi
```

### **Expected Usage Patterns**

**Idle (models loaded):**
```
VRAM: ~9.9GB (models only)
```

**During query:**
```
VRAM: ~10.5GB (models + processing)
Spike: +500MB temporary
```

**During batch embedding:**
```
VRAM: ~11.0GB (large batches)
Spike: +1.0GB with batch_size=64
```

**Critical threshold:**
```
Warning: >11.5GB (96%)
Danger:  >11.8GB (98%)
```

---

## Best Practices

### **Startup Sequence**

```python
# 1. Load embedding model first (smaller)
embedder = OllamaQwen3Embedder("qwen3-embedding:8b")
print(f"VRAM: ~4.7GB")

# 2. Load LLM second
llm = LLMClient(provider="ollama", model="qwen3:latest")
print(f"VRAM: ~9.9GB")

# 3. Optionally load reranker
reranker = ColBERTReranker()
print(f"VRAM: ~11.2GB")

# 4. Verify headroom
assert get_vram_free() > 500_000_000  # >500MB free
```

### **Graceful Degradation**

```python
try:
    # Attempt to load all models
    load_all_models()
except OutOfMemoryError:
    # Fall back to sequential loading
    logger.warning("Insufficient VRAM for parallel loading")
    use_sequential_mode()
```

---

## Conclusion

**Your 12GB VRAM is perfect for:**

✅ Parallel model loading (recommended)
✅ Full RAG pipeline in memory
✅ Fast query response (<2s)
✅ Batch processing optimizations
✅ Multiple concurrent queries
✅ Optional ColBERT reranking

**No changes needed** - your current setup is optimal!

---

**Updated:** 2025-10-25
**GPU:** 12GB VRAM
**Status:** Production ready ✅
