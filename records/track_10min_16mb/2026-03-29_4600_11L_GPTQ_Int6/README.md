# Experiment 186: 11L GPTQ Int6

## 목적
기존 uniform int6 per-row quantization을 GPTQ (Gradient-based Post-Training Quantization)로 교체하여 quantization error를 줄이고 val_bpb를 개선한다. GPTQ는 Hessian 정보를 활용하여 각 column의 quantization error를 나머지 column에 보상 분배함으로써 전체 reconstruction error를 최소화한다.

## 베이스
- 최신 11L baseline (이전 실험의 train_gpt.py)

## 변경 사항
1. **`gptq_quantize_int6` 함수 추가**: Diagonal Hessian 기반 GPTQ 알고리즘 구현
   - Per-row scale은 기존과 동일하게 percentile search로 결정
   - Column을 group_size=8 단위로 처리하며, quantization error를 group 내 남은 column에 Hessian weight 비례로 분배
   - H_diag이 없으면 기존 uniform quantization으로 fallback

2. **`collect_gptq_hessian` 함수 추가**: Calibration 데이터로 Hessian diagonal 수집
   - Validation data에서 8 batch (각 4 sequences) forward pass 실행
   - 각 Linear layer의 input activation X에 대해 X^T X diagonal 누적
   - n_samples로 나누어 정규화

3. **`mixed_quantize_int6` 수정**: Optional `hessian_dict` 파라미터 추가
   - Hessian이 있고 2D weight인 경우 `gptq_quantize_int6` 사용
   - 그 외에는 기존 `quantize_int6_per_row` 유지

4. **`main()` 수정**: Quantization 전에 calibration 실행
   - `collect_gptq_hessian`으로 Hessian 수집 후 `mixed_quantize_int6`에 전달

## 예상 결과
- val_bpb: 기존 대비 0.001~0.005 개선 (GPTQ의 error compensation 효과)
- step time: 변화 없음 (post-training quantization만 변경)
- artifact size: 변화 없음 (동일한 int6 format)
- calibration overhead: ~2-3초 추가 (8 batch forward pass)

## 실제 결과
_(실험 후 기록)_
