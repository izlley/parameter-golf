# Experiment 190: 11L Self-Generated GPTQ Calibration

## 목적
GPTQ 양자화 시 validation data 대신 모델 자체가 생성한 데이터를 calibration에 사용하여 data leakage 우려를 제거하고 양자화 품질을 개선한다.
핵심 아이디어는 abaybektursun (PR#1019)에서 제안된 AR self-generation 기반 GPTQ calibration이다.

## 베이스
현재 SOTA 11L 모델 (NorMuon + EMA + int6 양자화)

## 변경 사항
1. **`generate_calibration_data()`**: 학습 완료 후 모델이 AR 방식으로 64개 시퀀스 (각 1024 토큰)를 자체 생성
   - Temperature 0.8 샘플링, batch_size=8로 효율적 생성
   - Context window 512 토큰으로 제한하여 생성 속도 최적화
   - Rank 0에서만 생성 후 broadcast

2. **`collect_gptq_hessian()`**: 생성된 calibration 데이터를 모델에 통과시켜 각 Linear 레이어의 Hessian diagonal (X^T X) 수집
   - Forward hook 기반으로 입력 activation의 제곱합 누적
   - 샘플 수로 정규화

3. **`gptq_quantize_int6()`**: Hessian 기반 GPTQ 양자화
   - Hessian이 큰 컬럼(중요한 입력 차원)부터 순서대로 양자화
   - 양자화 오류를 Hessian 가중치에 비례하여 나머지 컬럼에 보상
   - int6 범위 [-31, 31], per-row scaling

4. **`mixed_quantize_int6()` 수정**: `hessian_dict` 파라미터 추가
   - Hessian이 있는 2D 텐서에 대해 GPTQ 양자화 적용
   - 없는 경우 기존 `quantize_int6_per_row` fallback

## 예상 결과
- **val_bpb**: 기존 대비 0.002~0.005 개선 (양자화 손실 감소)
- **step_time**: 학습 중 변화 없음 (양자화는 학습 후 수행)
- **artifact_size**: 동일 (int6 양자화 포맷 동일)
- **추가 시간**: Self-generation ~30-60초, Hessian 수집 ~5초

## 실제 결과
_(실험 후 기록)_
