# Exp 101: Mamba2 Mid-Sandwich

## Purpose
Exp 98 (Mamba2 Dual)에서 Mamba 레이어 배치를 [0,10]에서 [5,10]으로 변경.
중간+끝 배치가 입력+끝 배치보다 나은지 검증.

## Base
Exp 98: Mamba2 Dual (`2026-03-25_2300_11L_Mamba2_Dual`)

## Approach
- Mamba2 레이어를 layer 5, 10에 배치 (mid-sandwich 패턴)
- 나머지 9개 레이어는 standard attention
- VE_LAYERS="8,9" — Mamba 레이어와 비충돌
- 가설: layer 0의 Mamba는 토큰 임베딩 직후라 효과 제한적, 중간 레이어가 더 유의미한 sequential pattern 포착

## Key Changes
- MAMBA_LAYERS default: "0,10" → "5,10" (1줄 변경)

## Expected Results
- SW baseline: ~1.1217 (Exp 98) 대비 ±0.002
- Size: ~17.0MB (Exp 98과 동일, over budget — size 최적화 필요)
