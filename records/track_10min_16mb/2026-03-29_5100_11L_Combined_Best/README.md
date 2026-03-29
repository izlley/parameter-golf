# Exp 191: 11L Combined Best (XSA All + LZMA + GPTQ)

## 목적
abaybektursun 접근법에서 검증된 세 가지 개선사항을 결합하여 val_bpb를 낮춘다:
1. XSA를 마지막 4개 레이어가 아닌 전체 11개 레이어에 적용하여 cross-sequence attention 효과를 극대화
2. zstd 대신 LZMA 압축으로 artifact size를 줄여 모델 용량 여유 확보
3. GPTQ int6 양자화 (diagonal Hessian)로 양자화 에러를 줄여 post-quantization BPB 개선

## 베이스
Exp 162: 11L NorMuon + TTT_MuonOptimizer

## 변경 사항

### 1. XSA All Layers
- `xsa_last_n`: 4 -> 11 (모든 레이어에 XSA 적용)
- 학습 시 모든 attention 레이어에서 cross-sequence attention 사용

### 2. LZMA Compression
- `_COMPRESSOR = "lzma"` (기존 zstd에서 변경)
- 압축: `lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)`
- 해제: `lzma.decompress(data)`
- LZMA는 zstd보다 느리지만 압축률이 높아 artifact size 절감

### 3. GPTQ Int6 Quantization
- `gptq_quantize_int6()`: Column-wise quantization with Hessian error compensation (group_size=8)
- `collect_gptq_hessian()`: Validation data 8 batches로 Linear layer별 X^T X diagonal 수집
- `mixed_quantize_int6()`에 `hessian_dict` 파라미터 추가
- Hessian이 있는 2D weight는 GPTQ 사용, 없으면 기존 per-row quantization fallback

## 예상 결과
- val_bpb: ~1.045 이하 (XSA all layers + GPTQ로 양자화 에러 감소)
- step_time: ~63ms (XSA all layers로 약간 증가 가능)
- artifact_size: 기존 대비 감소 (LZMA 압축)

## 실제 결과
_(실험 후 기록)_
