# LLM Inference Fundamentals: The Nitty-Gritty Details

## Table of Contents
1. [Understanding LLM Architecture](#understanding-llm-architecture)
2. [The Inference Pipeline](#the-inference-pipeline)
3. [Tokenization Deep Dive](#tokenization-deep-dive)
4. [Attention Mechanism in Inference](#attention-mechanism-in-inference)
5. [Memory and Computation](#memory-and-computation)
6. [Optimization Techniques](#optimization-techniques)

## Understanding LLM Architecture

### Transformer Architecture Basics

Large Language Models (LLMs) are built on the Transformer architecture, which consists of:

1. **Embedding Layer**: Converts tokens to dense vectors
2. **Positional Encoding**: Adds position information
3. **Transformer Blocks** (repeated N times):
   - Multi-Head Self-Attention
   - Feed-Forward Network
   - Layer Normalization
   - Residual Connections
4. **Output Layer**: Projects to vocabulary size

### Key Components for Inference

#### 1. Embedding Layer
- **Input**: Token IDs (integers)
- **Output**: Dense vectors (embedding_dim)
- **Size**: Vocabulary size × embedding_dim
- **Example**: For a 50K vocab and 4096-dim embeddings = ~800MB

#### 2. Attention Mechanism
- **Purpose**: Determine which tokens to focus on
- **Computation**: Q (Query), K (Key), V (Value) matrices
- **Complexity**: O(n²) where n is sequence length
- **Memory**: Stores KV cache for efficiency

#### 3. Feed-Forward Network
- **Structure**: Two linear layers with activation
- **Size**: Typically 4× the embedding dimension
- **Example**: 4096-dim → 16384-dim → 4096-dim

## The Inference Pipeline

### Step-by-Step Process

```
Input Text
    ↓
Tokenization (text → token_ids)
    ↓
Embedding Lookup (token_ids → embeddings)
    ↓
Positional Encoding
    ↓
[For each layer:]
    ↓
Multi-Head Attention
    ↓
Feed-Forward Network
    ↓
Layer Normalization
    ↓
[End layer loop]
    ↓
Output Projection (embeddings → logits)
    ↓
Softmax (logits → probabilities)
    ↓
Sampling (probabilities → next_token)
    ↓
Decode Token (token_id → text)
```

### Autoregressive Generation

LLMs generate text **one token at a time**:

1. **Initial Forward Pass**: Process entire prompt
2. **Generate Loop**:
   - Take last generated token
   - Run forward pass (only on new token + cached KV)
   - Sample next token
   - Append to sequence
   - Repeat until stop condition

**Why this matters**: Each token generation requires a full forward pass, but we can cache intermediate computations.

## Tokenization Deep Dive

### What is Tokenization?

Tokenization converts human-readable text into tokens that the model understands.

### Types of Tokenizers

#### 1. Word-Level
- **Example**: "Hello world" → ["Hello", "world"]
- **Pros**: Intuitive, preserves meaning
- **Cons**: Large vocabulary, out-of-vocab issues

#### 2. Subword Tokenization (BPE, SentencePiece)
- **Example**: "Hello" → ["Hel", "lo"]
- **Pros**: Handles unknown words, smaller vocab
- **Cons**: Less intuitive, can split words oddly

### Tokenization Process

```python
# Example with HuggingFace tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, how are you?"
tokens = tokenizer.encode(text)
# Output: [15496, 11, 527, 499, 345, 30]

# Decode back
decoded = tokenizer.decode(tokens)
# Output: "Hello, how are you?"
```

### Important Tokenization Concepts

1. **Special Tokens**:
   - `<BOS>`: Beginning of sequence
   - `<EOS>`: End of sequence
   - `<PAD>`: Padding token
   - `<UNK>`: Unknown token

2. **Token IDs**: Each token maps to an integer ID in the vocabulary

3. **Attention Masks**: Indicate which tokens to attend to (ignore padding)

## Attention Mechanism in Inference

### Self-Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q (Query)**: What information am I looking for?
- **K (Key)**: What information do I have?
- **V (Value)**: What is the actual information?
- **d_k**: Dimension of keys (for scaling)

### Why Attention is Expensive

1. **Quadratic Complexity**: For sequence length n, compute n×n attention matrix
2. **Memory**: Store Q, K, V for all tokens
3. **Computation**: Matrix multiplications scale with n²

### KV Caching: The Key Optimization

During autoregressive generation:
- **First token**: Compute full Q, K, V
- **Subsequent tokens**: Only compute Q for new token, reuse cached K, V

**Memory Savings**: Instead of storing KV for all tokens each step, cache grows incrementally.

**Example**:
- Prompt: 100 tokens
- Generate: 50 tokens
- Without cache: Compute attention for 150×150 each step
- With cache: Compute 1×150 (new token × all previous)

## Memory and Computation

### Memory Requirements

#### Model Weights
- **FP32**: 4 bytes per parameter
- **FP16**: 2 bytes per parameter
- **INT8**: 1 byte per parameter
- **INT4**: 0.5 bytes per parameter

**Example (7B parameter model)**:
- FP32: 7B × 4 bytes = 28 GB
- FP16: 7B × 2 bytes = 14 GB
- INT8: 7B × 1 byte = 7 GB

#### Activation Memory
- **During forward pass**: Store intermediate activations
- **Size**: Depends on batch size, sequence length, hidden size
- **Example**: Batch=8, Seq=2048, Hidden=4096 ≈ 256 MB per layer

#### KV Cache Memory
- **Per token**: 2 × num_layers × hidden_size × num_heads
- **Example**: 32 layers, 4096 hidden, 32 heads = 8.4 MB per token
- **For 2048 tokens**: ~17 GB

### Computation Requirements

#### FLOPs (Floating Point Operations)

For a single forward pass:
```
FLOPs ≈ 2 × num_params × sequence_length
```

**Example (7B model, 2048 tokens)**:
- FLOPs ≈ 2 × 7B × 2048 = 28.7 TFLOPs
- On A100 (312 TFLOPs): ~92ms per forward pass

#### Throughput Considerations

- **Latency**: Time for one request (ms)
- **Throughput**: Requests per second
- **Trade-off**: Batching increases throughput but may increase latency

## Optimization Techniques

### 1. Quantization

Reduce model precision to save memory and speed up computation.

**Types**:
- **FP16**: Half precision (2x memory reduction)
- **INT8**: 8-bit integers (4x memory reduction)
- **INT4**: 4-bit integers (8x memory reduction)

**Trade-offs**: Lower precision → faster inference, but potential accuracy loss

### 2. Continuous Batching (vLLM)

Traditional batching:
- Wait for batch to fill
- Process entire batch
- Wait for all to complete

Continuous batching:
- Add requests as they arrive
- Remove completed requests
- Process partial batches efficiently

**Benefit**: Higher GPU utilization, lower latency

### 3. PagedAttention (vLLM)

Problem: KV cache fragmentation wastes memory

Solution: 
- Divide KV cache into fixed-size pages
- Allocate pages on-demand
- Reuse freed pages

**Benefit**: Efficient memory usage, support for longer sequences

### 4. Tensor Parallelism

Split model across multiple GPUs:
- Each GPU holds a portion of weights
- Communication between GPUs during forward pass

**Benefit**: Run larger models, distribute memory

### 5. Speculative Decoding

Use a smaller "draft" model to generate multiple tokens, verify with main model.

**Benefit**: Faster generation for certain workloads

## Practical Implications

### Choosing a Serving Framework

1. **HuggingFace Transformers**: Simple, flexible, slower
2. **vLLM**: Fast, optimized, continuous batching
3. **TensorRT-LLM**: NVIDIA-optimized, best for NVIDIA GPUs
4. **Triton Inference Server**: Multi-framework, production-ready

### Performance Metrics

- **Latency (P50, P95, P99)**: Response time percentiles
- **Throughput**: Tokens/second or requests/second
- **GPU Utilization**: How much GPU is being used
- **Memory Usage**: Peak and average memory consumption

### Common Bottlenecks

1. **Memory**: Not enough GPU memory for model + KV cache
2. **Computation**: Model too large for single GPU
3. **I/O**: Slow tokenization, network latency
4. **Batching**: Inefficient batching strategy

## Next Steps

Now that you understand the fundamentals, let's build practical serving solutions:
- **Topic 1**: Basic serving with HuggingFace Transformers
- **Topic 5**: High-performance serving with vLLM
- **Topics 6-12**: Production deployment and optimization

