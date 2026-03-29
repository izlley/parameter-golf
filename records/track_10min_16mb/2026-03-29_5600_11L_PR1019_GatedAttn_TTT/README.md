# Exp 196: PR#1019 Base + GatedAttention + SupermaskTTT

## 목적
PR#1019 (abaybektursun) base에 GatedAttention과 SupermaskTTT를 결합하여 val_bpb를 개선한다.
- GatedAttention: per-head sigmoid gate로 attention output을 modulate하여 학습 안정성과 표현력 향상
- SupermaskTTT: test-time training으로 per-channel masks를 학습하여 validation 시 모델 적응

## 베이스
PR#1019 (abaybektursun/pr1019_ValCalib_GPTQ_XSA_BigramHash3072)

## 변경 사항

### 1. GatedAttention 활성화
- `GATED_ATTENTION` 기본값: `"0"` -> `"1"`
- CausalSelfAttention에서 per-head sigmoid gate가 attention output에 적용됨
- 추가 파라미터: attn_gate (nn.Linear(dim, num_heads)), bias init=4.0

### 2. SupermaskTTT 추가
- `MuonLiteTTT` optimizer: sign-based updates with Nesterov momentum
- `eval_val_sliding_ttt` function: Parameter Banking 아키텍처에 맞는 monkey-patch 방식
  - MLP mask: per hidden channel (1536) init=3.0 + bias
  - Attn mask: per model_dim channel (512) init=3.0 + bias
  - Score-first (legal): 각 chunk를 먼저 scoring 후 mask 학습
  - LR schedule: warmup 5% -> constant 65% -> cosine decay 30%
- TTT hyperparameters:
  - ttt_lr=0.002, ttt_epochs=3, ttt_chunk_tokens=32768
  - ttt_momentum=0.9, ttt_batch_seqs=32, ttt_grad_clip=1.0

## 예상 결과
- val_bpb: ~1.045 이하 (GatedAttn + TTT 결합 효과)
- step_time: ~63ms (GatedAttn으로 약간 증가)
- artifact_size: 기존 대비 약간 증가 (attn_gate 파라미터 추가)

## 실제 결과
_(실험 후 기록)_
