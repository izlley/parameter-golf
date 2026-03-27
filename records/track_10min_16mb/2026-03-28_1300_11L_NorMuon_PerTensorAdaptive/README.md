# Exp 153: NorMuon + Per-tensor Adaptive Precision

## 목적
NorMuon에 per-tensor sensitivity 기반 int5/int6 혼합 양자화를 적용하여 모델 크기를 16MB 이하로 줄이면서 성능 저하를 최소화한다.

## 베이스
- Exp 121: NorMuon (1.1183 bpb, 17.2MB)

## 참고
- Exp 142: Per-tensor Adaptive (NorMuon+EWQ 베이스에 per-tensor sensitivity 적용)

## 변경 사항
1. **Per-tensor sensitivity analysis**: 각 텐서를 int6과 int5로 양자화한 후 MSE 차이(sensitivity)를 계산
2. **Adaptive int5 적용**: sensitivity가 낮은 하위 ~40% 텐서에 int5 양자화 적용 (고정 레이어 기반이 아닌 데이터 기반 결정)
3. **QAT int5 마킹**: layers 0-4의 CastedLinear 모듈에 `_ewq_int5=True` 설정하여 훈련 중 clip_range=15 사용
4. **quantize_int5_per_row 함수 추가**: [-15, 15] 범위의 int5 양자화 함수

## 예상 결과
- val_bpb: ~1.1250-1.1310
- artifact size: ~15.0-15.5MB (16MB 제한 이내)
- step time: NorMuon 베이스와 유사

## 실제 결과
_(실험 후 기록)_
