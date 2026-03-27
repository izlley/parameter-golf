# Exp 136: Soft Weight Tying (Flat Training, Shared at Save)

## 목적 (Purpose)

Training은 flat 11L 유지 (95ms/step), 저장 시 U-Net 대칭 layer pair를 병합하여 사이즈 절감.
Soft weight tying regularization으로 layer 5와 layer 10의 weights를 유사하게 유도.

**핵심 가설**:
1. Soft tying loss (lambda * ||W_5 - W_10||^2)로 두 layer의 가중치 수렴 유도
2. 저장 시 layer 10 제거, layer 5로 대체 + low-rank delta (rank-16)로 차이 복원
3. Training 속도 영향 없음 (95ms 유지), 사이즈 ~1.5MB 절감

## 구현 방향 (Implementation Approach)

**베이스**: Exp 121 (NorMuon, 1.1183 bpb, 95ms/step)

**Training 변경**:
- Soft weight tying loss: `tying_loss = lambda * sum(||p5 - p10||^2 for corresponding params)`
- lambda: 0.01 (작게 시작, 성능 영향 최소화)
- Tying targets: blocks.5 ↔ blocks.10 (U-Net 대칭 pair)
- Total loss = main_loss + tying_loss

**Save 변경**:
- Layer 10의 각 weight에 대해: delta = W_10 - W_5
- SVD 분해: delta ≈ U @ diag(S) @ V^T (rank-16)
- 저장: layer 5 weights (원본) + rank-16 deltas (fp16)
- Layer 10 weights는 저장하지 않음

**Eval 변경**:
- Layer 10 weights 복원: W_10 = W_5 + U @ diag(S) @ V^T
- 나머지 동일

## 예상 결과 (Expected Results)

- **step time**: 95ms (변경 없음)
- **val_bpb**: ~1.1190-1.1220 (soft tying이 약간의 성능 손실 유발 가능)
- **artifact size**: ~15.4MB (attention+MLP 1세트 절약 - rank-16 delta)

## 구현 완료 (Implementation Complete)

- Training: soft weight tying loss `lambda * ||W_5 - W_10||^2` on all matching 2D params
- tying_loss logged to wandb every 100 steps for monitoring
- Save: SVD delta compression (rank-16) — target layer replaced with source, only rank-16 U/S/Vh stored
- Eval: target layer reconstructed from source + `U @ diag(S) @ Vh` delta
- wandb final metrics (roundtrip, sliding_window, sliding_window_s64) logged

## 실제 결과 (Actual Results)

(실행 후 기록)
