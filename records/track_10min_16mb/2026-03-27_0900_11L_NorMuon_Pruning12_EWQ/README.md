# Exp 132: NorMuon + Pruning 12% + EWQ Combined

## Purpose
Pruning과 EWQ를 동시에 적용하여 최대 크기 절감. 두 기법은 직교적이므로 효과가 합산될 것으로 기대.

## Base
Exp 121 (11L NorMuon, val_bpb 1.1183, artifact 17,158,779 bytes)

## Changes from Exp 121

### 1. Magnitude Pruning 12%
- EMA 적용 후 하위 12% 절대값을 0으로 마스킹
- 0값 → zstd 압축 효율 향상

### 2. EWQ: Mixed int5/int6 Quantization
- Layers 0-5: int5, Layers 6-10: int6
- QAT에서도 int5 fake quantization 적용

## Architecture (Exp 121과 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15
- NorMuon

## Expected Results
- val_bpb: ~1.1210-1.1250 (+0.003~0.007)
- artifact: ~13.5-14.5MB (pruning + int5 combined savings)
- 크기 제한: **통과 예상 (여유 있음)**
