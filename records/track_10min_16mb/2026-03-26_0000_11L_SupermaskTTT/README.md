# Exp 99: Supermask TTT

## Purpose
TTT(Test-Time Training)가 signalrush 베이스(EMA + LN Scale)에서 실패하는 원인이 EMA 가중치 파괴라는 가설 검증.
모델 가중치를 완전히 동결하고, MLP 채널별 마스크만 학습하여 EMA 파괴 없이 TTT 효과를 얻는다.

## Base
Exp 95: GatedAttn + TTT (`2026-03-25_2000_11L_GatedAttn_TTT`)

## Approach
- **Supermask TTT**: 기존 TTT의 전체 모델 업데이트 대신, per-MLP channel mask (sigmoid(score)) 만 학습
- 마스크 파라미터: 11 layers × 1536 hidden = 16,896개 (vs 모델 27.8M)
- Init: score=5.0 → sigmoid≈0.993 (near-identity start)
- 마스크는 square 후에 적용: `leaky_relu → square → sigmoid_mask → proj`
- 모든 모델 가중치는 frozen (requires_grad=False)
- SGD로 마스크만 최적화, chunk 단위 score-first legal TTT

## Key Changes
- `eval_val_sliding_ttt` 함수 전체 교체 (line ~917-1072)
- MLP forward를 monkey-patch하여 masked version으로 대체
- TTT 완료 후 원래 forward 복원

## Expected Results
- SW baseline: ~1.1208 (Exp 95와 동일)
- TTT 적용 시: 1.1180~1.1200 (EMA 파괴 없이 -0.001~0.003 개선 기대)
- Size: ~16.9MB (Exp 95와 동일)
