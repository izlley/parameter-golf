# Exp 119: 11L SupermaskTTT V4 (Aggressive LR Decay)

## Purpose
Exp 115(SupermaskTTT V3)에서 TTT chunk 진행 시 U자형 반등 패턴 수정.
chunk 51에서 1.1094 달성 후 chunk 101~181에서 1.135까지 상승 → constant LR 구간이 너무 길어 overfitting.

## Base
Exp 115 (11L SupermaskTTT V3, val_bpb 1.11962742 sliding+TTT)

## V3 TTT Chunk 분석 (문제 진단)
| Chunk | BPB | Phase |
|-------|-----|-------|
| 1 | 1.1615 | 초기 적응 |
| 51 (2.7%) | **1.1094** | 최적점 |
| 101 (5.3%) | 1.1188 | 반등 시작 |
| 181 (9.6%) | 1.1280 | 최악점 (U자 바닥) |
| 501 (26.5%) | 1.1309 | 여전히 높음 |
| 1301 (68.8%) | 1.1252 | 서서히 회복 |
| 1893 (100%) | 1.1218 | V3 최종 |

## Changes from V3
- **TTT LR Schedule 변경**:
  - V3: warmup 5% + constant 65% + cosine decay 30%
  - V4: warmup 2% + constant 20% + **cosine decay 78%**
  - LR min ratio: 0.05 (LR이 0이 아닌 base의 5%까지만 감소)
- constant 구간을 대폭 축소하여 최적점(chunk ~50) 이후 빠르게 LR 감소
- 78% cosine decay로 overfitting 방지하면서도 장기 적응 유지

## Architecture (Exp 93/115 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997) + U-Net skip
- Late QAT threshold 0.15

## TTT Hyperparameters
- ttt_lr: 0.005 (V3와 동일)
- ttt_epochs: 3
- ttt_chunk_tokens: 32768
- **ttt_warmup_frac: 0.02** (V3: 0.05)
- **ttt_constant_frac: 0.20** (V3: 0.65)
- **ttt_lr_min_ratio: 0.05** (V3: 0.0)
- mask_init: 3.0, momentum: 0.9

## Expected Results
- Target: < 1.1196 bpb (V3의 1.11962742 개선)
- U자형 반등 패턴 완화 → chunk 전체에서 안정적 성능
- Artifact: ~16.70MB (V3와 동일)
