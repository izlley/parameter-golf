# Exp 182: Norm Tweaking + AWQ Channel Scaling

## 목적 (Purpose)
양자화 후 품질 손실을 최소화하는 두 가지 post-quantization 보정 기법 결합.
1. **AWQ (Activation-Aware Weight Quantization)**: 활성화 통계 기반 per-channel scaling으로 salient weight 보호
2. **Norm Tweaking**: 양자화 후 scale 파라미터 미세조정으로 activation distribution shift 보상

## 베이스
Exp 172 (11L MuonTTT SensitivityEWQ, TTT val_bpb 1.1227, 15.9MB)

## 변경 사항
- **AWQ Channel Scaling**: 양자화 전 calibration forward pass로 per-channel activation magnitude 수집
  - 채널별 중요도(activation magnitude) 기반 scaling factor 적용
  - salient channel의 양자화 에러 감소 (가중치를 scaling하여 양자화 grid에 더 적합하게)
- **Norm Tweaking**: 양자화 후 eval_model에서 attn_scale, mlp_scale, skip_weights, resid_mix만 fine-tune
  - 원본(pre-quant) 모델과 양자화 모델 간 KL-divergence 최소화
  - ~50 steps gradient descent (lr=1e-3)
  - 조정된 scale params를 artifact에 재저장
- **근거**: AWQ (MLSys 2024 Best Paper), Norm Tweaking (AAAI 2024)

## 예상 결과
- TTT val_bpb: ~1.1190-1.1210 (EWQ 비용 +0.0067 → +0.003~0.004로 감소)
- artifact size: ~15.9MB ✅ (scale params 크기 무시 가능)
- step_avg: ~95ms (변경 없음)
- 추가 시간: 양자화 후 보정 ~3-5초

## 실제 결과
(실행 후 기록)
