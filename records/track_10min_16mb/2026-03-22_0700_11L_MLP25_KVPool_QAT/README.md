# 11L + MLP 2.5x + KV Pooling (encoder) + QAT

## 목적

11L은 val_bpb 1.1369 (전체 최고)를 달성했으나 17.1MB로 크기 초과. Pruning 20%로는 품질 손해가 과도(+0.011). **모델 구조를 변경하여 근본적으로 크기를 줄이면서, KV pooling으로 attention 품질을 보완**한다.

## 구현 방향

- **베이스**: Exp 26 (11L, pruning 10%, val_bpb 1.13690)
- **핵심 변경**:
  1. **MLP expansion 3.0x → 2.5x**: 레이어당 MLP 크기 축소 (1536→1280 hidden)
     - 레이어당 절약: 512×256×2 = 262K params → int5+zstd 후 ~150KB
     - 11L 합계: ~1.5-1.7MB 절약 → 17.1MB - 1.6MB ≈ **15.5MB** (예산 내)
  2. **KV Pooling (encoder 레이어)**: 초반 5개 레이어에서 K,V를 stride=2 avg pool
     - Attention matrix 크기 절반 → step_avg 감소
     - 파라미터 추가 없음 (avg pool은 학습 파라미터 없음)
     - Causal mask 정확히 구현: pooled position이 커버하는 모든 원래 위치가 query 이전일 때만 attend
  3. **Pruning 10% 유지**, **BigramHash 10240 유지**

## KV Pooling 설계

```
Encoder (5 layers): blocks 0~4 → KV pooled (stride=2, seqlen 2048→1024)
Decoder (6 layers): blocks 5~10 → Full attention (seqlen 2048)
```

- K, V를 reshape(bsz, heads, kv_len, stride, dim).mean(dim=3)으로 pool
- Causal mask: query position i는 pooled position j에 attend if (j+1)*stride - 1 ≤ i
  - 즉, pooled 위치가 커버하는 마지막 원래 위치가 query 위치 이전이어야 함

## 크기 추정

- MLP 3.0x→2.5x: 레이어당 ~150KB 절약 × 11 = ~1.65MB
- 11L Pruning 10% 기준: 17.1MB - 1.65MB ≈ **15.45MB** (1.55MB 여유)
- KV pooling: 파라미터 없음, 크기 영향 없음

## 예상 결과

- **val_bpb**: ~1.140~1.145
  - MLP 축소 손해: +0.002~0.004 bpb (hidden 1536→1280, 표현력 감소)
  - KV pooling 보완: -0.001~0.002 bpb (step_avg 감소 → 더 많은 학습 스텝)
  - 순 효과: SOTA(1.1424)와 비슷하거나 약간 개선
- **아티팩트 크기**: ~15.5MB (예산 내)
- **학습 시간**: step_avg ~85-90ms (MLP 축소 + KV pooling으로 감소 기대)
- **위험**: 중-높
  - MLP 2.5x가 3.0x 대비 얼마나 품질 손해인지 미지수
  - KV pooling이 torch.compile + causal mask에서 정상 작동하는지 확인 필요
  - Custom attn_mask 사용 시 FlashAttention이 비활성화될 수 있음 → 속도 저하 위험

## 실제 결과

(학습 후 기록 예정)
