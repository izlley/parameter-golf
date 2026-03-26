# Exp 128: 11L SGT V2 (Speed Optimized)

## Purpose
Exp 117 SGT의 속도 최적화. Entropy-guided looping 유지하되 overhead 최소화.

## Base
Exp 117 (11L SGT EntropyLoop, val_bpb 1.1348 sliding_window, step_avg 125.96ms)

## Changes from Exp 117

### 1. Active layers 3→1
- `max_active_layers`: 3 → 1
- Layer 10만 looping → overhead ~20ms 절감
- 3개 layer looping이 step_avg 150ms→126ms로 전체 +33% 부담이었음

### 2. Entropy 빈도 50→500
- `sgt_entropy_interval`: 50 → 500
- Entropy 계산 횟수 10x 감소 → 평균 ~2ms 절감
- Entropy EMA가 충분히 안정적이므로 500 간격으로도 유효

### 3. Entropy batch 4→1
- `ent_x[:4]` → `ent_x[:1]`
- [B,H,T,T] attention matrix 메모리 75% 절감
- 단일 시퀀스로도 head별 entropy 경향 파악 가능

## Architecture
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15
- **SGT V2**: max_active=1, entropy_interval=500, entropy_batch=1

## Expected Results
- step_avg: ~98ms (125.96ms 대비 ~22% 빠름)
- Target: < 1.1198 bpb (base Exp 93)
- 예상 steps: ~6,100 (4,764 대비 +1,300)
- Size: ~15.85MB (동일)
