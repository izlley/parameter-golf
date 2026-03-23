# 10L Gated Attention (Exp 64)

## 목적
정적 attn_scale(per-dim 상수)을 입력 의존적 게이팅으로 교체.
각 토큰/위치에서 어텐션 출력의 기여도를 동적으로 조절.

## 변경 사항 (vs SOTA 10L)
- attn_scale(512 params) → attn_gate Linear(512→512, 262K params)
- 파라미터 추가: +262K/layer × 10 = +2.62M total (~1.6MB)
- step_avg 영향: +1~2ms
- Zero-init → sigmoid(0)=0.5 → 초기에는 attn 출력의 50% 통과
- prune_quantile: 0.05 → 0.12 (12% pruning으로 크기 보상)

## 예상 결과
- val_bpb: ~1.139~1.143
- 크기: ~17.3MB → pruning 12%로 ~16MB 이내 목표

## 실제 결과
(학습 후 기록 예정)
