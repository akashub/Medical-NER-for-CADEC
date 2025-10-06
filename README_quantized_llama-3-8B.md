# README 1: Quantized Llama-3-8B On-Device Approach

## Medical NER with Llama-3-8B-Instruct (4-bit Quantization)

### Overview

This implementation uses Meta's Llama-3-8B-Instruct model with 4-bit quantization (via bitsandbytes) to perform medical Named Entity Recognition on patient forum posts from the CADEC dataset. The model runs entirely on-device using a single GPU.

### Performance Summary

**Overall Results (50 files)**:
- Macro-Avg Precision: 0.2837
- Macro-Avg Recall: 0.2632
- Macro-Avg F1-Score: **0.2459**

**Performance Distribution**:
- Complete failures (F1=0): 9/50 (18%)
- Poor (F1 < 0.3): 25/50 (50%)
- Moderate (0.3-0.6): 12/50 (24%)
- Good (F1 ≥ 0.6): 4/50 (8%)

### Critical Assessment

This approach **significantly underperforms** compared to expectations:

- Published zero-shot baselines: F1 = 0.35-0.45
- Our result: F1 = 0.25 (40% below baseline)
- 68% of files score below F1 = 0.3
- 18% complete failure rate

**This is not acceptable performance for any practical application.**

---

## Technical Approach

### Model Configuration

**Model**: `meta-llama/Meta-Llama-3-8B-Instruct`

**Quantization**:
```
- 4-bit quantization (NF4)
- Double quantization enabled
- Compute dtype: bfloat16
- Memory footprint: ~5GB VRAM
```

**Generation Parameters**:
```
- Temperature: 0.05 (very low for consistency)
- Top-p: 0.9
- Max new tokens: 3072
- Repetition penalty: 1.2
```

### Prompt Design Evolution

I went through three major prompt iterations:

**Version 1**: Basic instruction-following format
- Result: Generated invalid tags (B-QUANTITY, B-BODYPART)
- Issue: Model invented its own tag schema

**Version 2**: Strict tag enumeration with negative examples
- Result: Still inconsistent, mixed formats
- Issue: Model added explanations, commentary

**Version 3**: Llama-3 chat template with minimal few-shot
- Result: Current version, but still poor performance
- Issue: Fundamental model limitation on this task

### Why This Failed

**1. Model Capacity Limitations**

Llama-3-8B, even without quantization, lacks the medical domain knowledge for this task. The model:
- Confuses body parts with ADRs ("hand", "feet", "head" tagged as ADR)
- Fragments entities incorrectly ("Severe" separate from "numbness")
- Misses obvious multi-word entities ("sharp shock like pain in head" broken into 4 separate ADRs)

**2. Quantization Quality Loss**

4-bit quantization compounds the problem:
- Pattern recognition degrades
- Instruction following becomes inconsistent
- One file (LIPITOR.64.txt) produced completely unparseable output

**3. Output Format Instability**

Despite explicit format instructions, the model:
- Adds commentary ("Here is the tagged text:")
- Uses malformed tags (words split mid-tag: "hand/I-AD")
- Skips words entirely
- Inserts random characters

---

## Design Decisions & Rationale

### Why Quantization?

**Chosen**: 4-bit NF4 quantization

**Reasoning**:
- Enables running 8B model on consumer GPU (16GB VRAM)
- Faster inference (~2s per file vs 5-10s for full precision)
- No external API costs

**Trade-off**: Accepted quality loss for accessibility and cost

**Outcome**: Quality loss was far greater than anticipated - not worth it

### Why Llama-3-8B?

**Chosen**: Llama-3-8B-Instruct over alternatives

**Reasoning**:
- Open source and commercially viable
- Strong general instruction-following
- Better than Mistral-7B on benchmarks
- Smaller than 70B variants (fits on single GPU)

**Outcome**: General capabilities didn't translate to medical NER performance

### Why BIO Tagging?

**Chosen**: BIO (Begin-Inside-Outside) format

**Reasoning**:
- Standard in NER literature
- Handles multi-word entities naturally
- Unambiguous entity boundaries
- Easy to parse programmatically

**Outcome**: Format was sound, but model couldn't follow it consistently

### Why Relaxed Matching (66% Jaccard)?

**Chosen**: Jaccard similarity > 0.66 for entity matching

**Reasoning**:
- Patient language is informal and variable
- "severe numbness in hand" vs "numbness in hand" should both count
- Clinical utility matters more than exact boundaries

**Outcome**: Even with relaxed matching, performance was abysmal

---

## What Went Wrong: Detailed Analysis

### Failure Mode 1: Entity Fragmentation

**Example**: LIPITOR.521.txt

Ground truth:
```
"Severe numbness in hand"
"sharp shock like pain in head"
```

Predicted:
```
"Severe" (separate)
"numbness in" (incomplete)
"sharp" (separate)
"shock like" (separate)
"pain" (separate)
```

**Issue**: Model treats each word as independent entity rather than recognizing phrases.

### Failure Mode 2: Unparseable Output

**Example**: LIPITOR.64.txt

LLM Output:
```
low/O B-ADR mood/O, loss/O of/O I-ADR self-confidence/O
```

**Issue**: Tags appear without words attached. Parser extracted zero entities.

### Failure Mode 3: Hallucinated Tags

Despite explicit valid tag list, model still generated:
- `I-AD` (incomplete)
- `B-DAY`
- `B-BODYPART` (previous runs)

### Failure Mode 4: Context Misunderstanding

Model consistently failed to distinguish:
- ADR (side effect from drug) vs Symptom (from disease)
- Drug names vs dosages
- Clinical terms vs casual mentions

---

## Lessons Learned

### What I Would Do Differently

**1. Don't Use Quantized Models for Complex NER**

The quality loss from 4-bit quantization is too severe for tasks requiring:
- Precise pattern recognition
- Consistent output formatting
- Domain-specific knowledge

**Better approach**: Use full-precision model with API (Gemini, GPT-4) or fine-tune smaller model.

**2. Don't Rely on Instruction-Following for Structured Output**

Zero-shot prompting, even with examples, is insufficient for:
- Medical domain specificity
- Consistent structured output
- Complex multi-token entities

**Better approach**: Fine-tune on labeled data or use constrained decoding.

**3. Validate Output Format Aggressively**

Should have implemented:
- Format validation before parsing
- Retry mechanism for malformed output
- Fallback to rule-based extraction

---

## Why This Approach Was Attempted

Despite poor results, this was a valuable learning exercise:

**Hypothesis**: Open-source LLMs have reached sufficient capability for zero-shot medical NER.

**Test**: Can a quantized 8B model match commercial API performance?

**Result**: No. Strong capabilities gap remains.

**Value**: Established baseline showing limitations, informed better approach (Gemini pipeline).

---

## Pros and Cons Summary

### Pros

✓ Completely offline, no API dependency
✓ No per-query costs
✓ Reproducible results
✓ Full data privacy (medical data never leaves device)
✓ Fast inference (2-3s per document)
✓ Runs on consumer hardware (16GB VRAM)

### Cons

✗ Unacceptably low F1 score (0.25)
✗ 18% complete failure rate
✗ Inconsistent output formatting
✗ Entity fragmentation issues
✗ Poor medical domain understanding
✗ Quantization quality loss
✗ Not suitable for any production use
✗ Requires extensive post-processing that still fails

---

## Conclusion

This quantized on-device approach demonstrates that **accessibility and privacy do not compensate for poor performance**. While the technical implementation is sound (proper quantization, reasonable prompt engineering, robust parsing), the fundamental limitation is model capacity.

For medical NER requiring F1 > 0.5, you need either:
1. Larger models (70B+) with full precision
2. Fine-tuned domain-specific models
3. Commercial API models (GPT-4, Gemini)

The 4-bit quantized Llama-3-8B is unsuitable for this task, regardless of prompt optimization.

This failure informed the development of the Gemini + LangGraph pipeline, which achieved F1 = 0.57 (2.3x improvement).

---

**Recommendation**: Do not use this approach. Use Gemini pipeline or fine-tuned BioBERT instead.