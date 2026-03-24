# Exp 84: 14L Shared FFN Across Layers

## 목적
MLP 가중치를 레이어 그룹 간 공유하여 파라미터 수를 줄이고, 그 여유로 레이어를 11 -> 14로 증가.
Embedding Sharing (tie_embeddings)은 이미 적용 중.

## 베이스
`2026-03-24_2100_11L_LeakyReLU_Bigram6144/train_gpt.py` (Exp 73, val_bpb 1.1221)

## 변경 사항

| 파라미터 | Exp 73 | Exp 84 | 근거 |
|----------|--------|--------|------|
| NUM_LAYERS | 11 | **14** | Shared FFN으로 예산 확보 |
| shared_ffn_groups | 0 (없음) | **2** | encoder/decoder 각 1개 MLP 공유 |
| VE_LAYERS | 9,10 | **12,13** | 14L에 맞게 마지막 2개 레이어로 조정 |
| XSA_LAST_N | 4 | 4 (유지) | layers 10,11,12,13에 XSA 적용 |

## Shared FFN 설계
- `shared_ffn_groups=2`: 2개의 공유 MLP 생성
  - Group 0: layers 0-6 (encoder) — 1개 MLP 공유
  - Group 1: layers 7-13 (decoder) — 1개 MLP 공유
- Block은 자체 MLP를 생성하지 않고, GPT가 forward 시 해당 그룹의 shared MLP를 전달
- 각 Block은 고유한 attention, norm, scale, resid_mix 파라미터를 유지

## 파라미터 예산 분석
- Per-layer attention: ~789K params (~590KB compressed)
- Shared MLP (2개): ~1.57M × 2 = ~3.14M params (~2.36MB compressed)
- 기존 11L (각자 MLP): 11 × 1.57M = ~17.3M MLP params
- 14L Shared (2 MLP): 2 × 1.57M = ~3.14M MLP params → **~14.1M 절약**
- 추가 attention (3 layers): 3 × 789K = ~2.37M params
- **예상 총**: ~14.5MB (16MiB 이내, 충분한 여유)

## 환경 변수
```bash
NUM_LAYERS=14              # 레이어 수
SHARED_FFN_GROUPS=2        # 공유 MLP 그룹 수 (0=공유 없음)
VE_LAYERS="12,13"          # Value Embedding 적용 레이어
```

## 예상 결과
- **장점**: 더 깊은 네트워크로 representation capacity 증가
- **위험**: 중간 (MLP 공유가 표현력 제한 가능, step_avg 증가 가능)
- val_bpb: 1.1180~1.1230 (깊이 증가 효과 vs MLP 공유 손실)

## 실제 결과
(학습 후 기록 예정)
