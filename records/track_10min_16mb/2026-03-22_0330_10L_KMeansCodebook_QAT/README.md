# 10L K-means Codebook Quantization + QAT

## 목적

현재 SOTA (10L, val_bpb 1.14239)는 uniform per-row 양자화 (int5/int6)를 사용한다. Uniform 양자화는 가중치 분포가 비균일(일반적으로 가우시안)임에도 등간격 레벨을 사용하므로, **분포 밀집 영역에서 정밀도가 부족**하고 **꼬리 영역에서는 레벨이 낭비**된다.

K-means codebook 양자화는 가중치 분포에 최적화된 비균일 centroid를 학습하여, **동일한 레벨 수(32/64)에서 양자화 에러를 근본적으로 줄이는** 접근이다.

## 구현 방향

- **베이스**: 현재 최고 결과 (10L QAT, val_bpb 1.14239)
- **핵심 변경**: **학습은 동일** (QAT STE, uniform fake-quant 유지), **post-training 양자화만 변경**
  - `kmeans_quantize_per_row()` 함수 추가:
    - Per-row scaling으로 정규화 (기존과 동일)
    - 정규화된 값에 대해 k-means 클러스터링 (20 iterations)
    - 대규모 텐서는 4096 samples 서브샘플링으로 속도 확보
    - MLP: 32 centroids (int5 대응), Attention: 64 centroids (int6 대응)
  - `mixed_quantize_int6()` 수정: uniform → k-means 양자화
  - `dequantize_mixed_int6()` 수정: codebook lookup 역양자화
- **저장 형식**: indices (int8) + per_row_scale (fp16) + codebook (fp16, 32 or 64 values per tensor)
- **크기**: codebook 오버헤드 ~64-128 bytes/tensor (무시 가능), 전체 크기 기존과 동일
- **학습 시간**: 학습은 100% 동일, post-training k-means ~10초 추가 (10분 이내 충분)

## K-means vs Uniform 양자화

```
가중치 분포 (가우시안):
                    ████
                  ████████
                ████████████
              ████████████████
          ████████████████████████

Uniform:    |  |  |  |  |  |  |  |     → 등간격, 꼬리 영역 낭비
K-means:   |    || ||| || |    |       → 밀집 영역에 집중, 에러 최소화
```

## 예상 결과

- **val_bpb**: ~1.139~1.142
  - 양자화 에러 감소 → dequantized 모델의 bpb 개선
  - 학습은 동일하므로 학습 중 val_bpb는 변화 없음 (post-training에서만 차이)
  - 개선폭은 양자화 에러가 전체 bpb에 기여하는 비율에 의존
- **아티팩트 크기**: ~15.8MB (기존과 동일, codebook 오버헤드 무시 가능)
- **학습 시간**: step_avg ~93ms (기존과 완전 동일)
- **위험**: 낮음
  - 학습에 전혀 영향 없음 (가장 안전한 실험)
  - k-means가 per-row + subsample 조합에서 충분한 품질을 내는지가 관건

## 실제 결과

(학습 후 기록 예정)
