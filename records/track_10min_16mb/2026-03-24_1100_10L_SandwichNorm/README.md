# 10L Sandwich Norm (Exp 63)

## 목적
각 sub-layer(attention, MLP) 출력에 post-RMSNorm 추가 (Sandwich Norm).
Pre-norm만 사용하는 SOTA 대비, 출력 정규화로 학습 안정성 및 양자화 내성 향상 기대.

## 변경 사항 (vs SOTA 10L)
- Block에 post_attn_norm, post_mlp_norm 추가
- 파라미터 추가: 거의 0 (RMSNorm에 학습 파라미터 없음)
- step_avg 영향: <0.5ms

## 예상 결과
- val_bpb: ~1.140~1.143
- 크기: ~15.8MB (SOTA와 동일)

## 실제 결과
(학습 후 기록 예정)
