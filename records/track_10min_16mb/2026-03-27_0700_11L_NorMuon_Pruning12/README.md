# Exp 130: NorMuon + Magnitude Pruning 12%

## Purpose
NorMuon의 best bpb (1.1183)를 유지하면서, 12% magnitude pruning으로 아티팩트 크기를 16MB 이하로 줄임.
Pruning으로 생긴 0값들이 zstd 압축 효율을 크게 높일 것으로 기대.

## Base
Exp 121 (11L NorMuon, val_bpb 1.1183, artifact 17,158,779 bytes)

## Changes from Exp 121

### Magnitude Pruning 12%
- EMA 적용 후, 직렬화 직전에 magnitude pruning 수행
- ndim==2 이고 numel > 65536인 파라미터에 대해 하위 12% 절대값을 0으로 마스킹
- 0값은 int6 양자화 시 0으로 유지 → zstd에서 매우 효율적으로 압축
- 이전 Exp 22 (10% pruning)에서 ~0.001 bpb/1% 비용 확인

## Architecture (Exp 121과 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15
- NorMuon: neuron-wise L2 normalization after NS5

## Expected Results
- val_bpb: ~1.1195 (+0.001~0.002 from pruning)
- artifact: ~14.5-15.5MB (0값 증가 → zstd 압축률 개선)
- 크기 제한: **통과 예상**
