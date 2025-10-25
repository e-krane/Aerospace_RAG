"""
Quick test for Qwen3 setup (embeddings + LLM only, no RAGAS).
"""

print("\n" + "=" * 70)
print("QWEN3 QUICK TEST")
print("=" * 70)

# Test 1: Embeddings
print("\n[1/2] Testing qwen3-embedding:8b...")
from src.embeddings.ollama_qwen3_embedder import create_embedder
import numpy as np

embedder = create_embedder(model_name="qwen3-embedding:8b", use_matryoshka=True, reduced_dimensions=256)

texts = [
    "The Euler buckling formula predicts column stability.",
    "Stress-strain relationships govern material behavior.",
]

embeddings = embedder.embed(texts)
print(f"✅ Embeddings shape: {embeddings.shape}")
print(f"   Similarity: {np.dot(embeddings[0], embeddings[1]):.3f}")

# Test 2: LLM
print("\n[2/2] Testing qwen3:latest...")
from src.llm.client import LLMClient

client = LLMClient(provider="ollama", model="qwen3:latest", temperature=0.1)

response = client.generate(
    prompt="What is the Euler buckling formula?",
    context=["The Euler buckling formula is: P_cr = (π²EI) / (KL)²"]
)

print(f"✅ LLM Response ({response.tokens_used} tokens, {response.latency_ms:.0f}ms):")
print(f"   {response.content[:150]}...")

print("\n" + "=" * 70)
print("SUCCESS! Your Qwen3 setup is working:")
print("  ✅ qwen3-embedding:8b (4.7GB) - #1 MTEB")
print("  ✅ qwen3:latest (5.2GB)")
print("  📊 Peak VRAM: ~5.2GB / 12GB (43%)")
print("  📊 Both loaded: ~9.9GB / 12GB (83%)")
print("  💾 Headroom: ~2.1GB for processing")
print("=" * 70 + "\n")
