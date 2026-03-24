# Exp 86: Saliency-Based Mixed Precision (PB-LLM style)

## Purpose
Per-weight mixed precision quantization inspired by PB-LLM: top 1% salient weights (by magnitude) are preserved as fp16, while the remaining 99% are quantized to int4. This achieves better reconstruction accuracy than uniform int6 quantization while reducing artifact size.

## Base
Exp 73 (val_bpb 1.1221, 11L, 512dim, 8 heads, 4 KV heads, int6+zstd compression)

## Changes

| Parameter | Base (Exp 73) | This Experiment |
|-----------|--------------|-----------------|
| `saliency_top_pct` | N/A | 0.01 (1%) |
| `saliency_low_bits` | N/A | 4 (int4) |
| Quantization scheme | Uniform int6 | Mixed: 1% fp16 + 99% int4 |

## Size Analysis
- Current: 6 bits per weight (int6)
- New: 1% x 16 bits (fp16) + 99% x 4 bits (int4) + index overhead
  - Average: 0.01 * 16 + 0.99 * 4 = 4.12 bits/weight (before index overhead)
  - Index overhead: 32-bit int per salient weight = 0.01 * 32 = 0.32 bits/weight
  - Effective: ~4.44 bits/weight vs 6 bits/weight -> ~26% size reduction
- Salient weights preserve critical model capacity while aggressive quantization reduces size

## Approach
- No training changes - purely a quantization improvement at export time
- Saliency is computed as absolute weight magnitude (|w|)
- Top-k salient weights stored as (fp16 value, int32 index) pairs
- Remaining weights quantized with per-row scaling at `low_bits` precision
- Clip percentile search (0.999, 0.9995, 0.9999, 1.0) for optimal row scaling
- Reconstruction error includes salient value restoration for accurate calibration

## Risk
Low - quantization-only change, no impact on training dynamics.

## Expected Results
- val_bpb: 1.1210 ~ 1.1250 (better quantization fidelity -> less degradation from compression)
- Artifact size: significant reduction from ~6 bits/weight to ~4.44 bits/weight

## Environment Variables
- `SALIENCY_TOP_PCT`: fraction of weights to keep as fp16 (default: 0.01)
- `SALIENCY_LOW_BITS`: bit width for non-salient weights (default: 4)

## 실제 결과
(학습 후 기록 예정)
