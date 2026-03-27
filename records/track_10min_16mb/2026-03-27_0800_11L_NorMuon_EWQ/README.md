# Exp 131: NorMuon + EWQ (Entropy-Weighted Quantization)

## Purpose
NorMuon 베이스에 EWQ(int5 early layers) 적용. 초기 레이어는 덜 민감하므로 int5로 양자화하여 크기 절감.

## Base
Exp 121 (11L NorMuon, val_bpb 1.1183, artifact 17,158,779 bytes)

## Changes from Exp 121

### EWQ: Mixed int5/int6 Quantization
- Layers 0-5: int5 양자화 ([-15, 15], 31 levels) — 6개 레이어
- Layers 6-10: int6 양자화 ([-31, 31], 63 levels) — 5개 레이어
- QAT에서도 early layers는 int5 fake quantization 적용
- Exp 124 대비 int5 적용 범위를 확대 (0-3 → 0-5)
- `quantize_int5_per_row` 함수 추가

## Architecture (Exp 121과 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15
- NorMuon

## Expected Results
- val_bpb: ~1.1200-1.1230 (+0.002~0.005 from int5)
- artifact: ~15.0-15.8MB (int5가 int6 대비 ~17% 작은 양자화 값)
- 크기 제한: **통과 예상**
