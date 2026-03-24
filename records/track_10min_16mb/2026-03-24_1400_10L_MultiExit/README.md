# 10L Multi-Exit Ensemble (Exp 66)

## 목적
디코더 중간 레이어에서 보조 예측 헤드를 추가하여 multi-exit 앙상블 효과를 달성.
Exp 40에서 SWA + last의 logit 앙상블이 best val_bpb(1.1365)를 기록한 것에 착안.
모델 내부에 앙상블을 내재화하여 단일 모델로 제출 가능한 형태로 구현.

## 변경 사항 (vs SOTA 10L)
- GPT에 `exit_decoder_indices=[1, 3]` 추가 (디코더 레이어 1, 3 = global 레이어 6, 8)
- 각 exit에 RMSNorm 추가 (`aux_norms`)
- **학습 시**: 보조 exit에서 auxiliary cross-entropy loss 계산, `aux_loss_weight=0.1`로 main loss에 합산
- **평가 시**: 모든 exit logits + final logits 평균 (3개 logits 앙상블)
- Projection은 tied embedding 공유 (추가 lm_head 없음)

## 파라미터 비용
- RMSNorm 2개: 학습 파라미터 없음 (F.rms_norm 기반)
- 총 추가 비용: **~0** (코드 로직 변경만)

## 근거
- 중간 레이어의 hidden state도 유의미한 예측력을 가짐
- 학습 시 auxiliary loss가 중간 레이어의 표현 학습을 강화 (deep supervision)
- 평가 시 logit 앙상블로 예측 분산 감소

## 예상 결과
- val_bpb: ~1.138~1.142
- 크기: ~15.9MB (SOTA와 동일)
- 위험: 중간 (auxiliary loss가 main task를 방해할 수 있음)

## 실제 결과
(학습 후 기록 예정)
