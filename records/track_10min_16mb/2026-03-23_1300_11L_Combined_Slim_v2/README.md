# 11L Combined Slim v2 (GQA2 + MLP2.75x + Pruning15% + Bigram9216)

## 목적
Exp 34 (11L Combined Slim)의 val_bpb 1.13984는 SOTA 대비 -0.00255 개선이었으나, 16.48MB로 478KB 초과.
Pruning 강화(10%→15%)와 BigramHash 축소(10240→9216)로 크기를 16MB 이내에 수용.

## 변경 사항 (vs Exp 34)
- Pruning: 10% → **15%**  (~300KB 추가 절약)
- BigramHash: 10240 → **9216** (~100KB 절약)
- 합계: ~400KB 절약 → 16.48MB - 0.4MB ≈ **16.08MB** (예산 내 목표)

## 변경 사항 (vs SOTA 10L)
- NUM_LAYERS: 10 → 11
- Encoder(5L) KV heads: 4 → 2
- Encoder(3L) MLP: 3.0x → 2.75x
- Pruning: 5% → 15%
- BigramHash: 10240 → 9216
- FP16_KEEP: blocks.9.attn.c_k

## 예상 결과
- val_bpb: 1.140~1.143 (Exp 34의 1.13984에서 pruning/bigram 축소로 +0.001~0.003)
- 크기: ~16.0MB (예산 내)
- 목표: SOTA(1.14239) 대비 개선 + 16MB 이내

## 실제 결과
(학습 후 기록 예정)
