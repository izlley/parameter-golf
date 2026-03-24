# Exp 85: Hybrid Ternary QAT — MLP Ternary + Attention int6

## 목적
BitNet b1.58 방식으로 MLP 가중치를 ternary ({-1, 0, 1})로 양자화.
Attention은 기존 int6 유지 (precision-sensitive).
MLP 아티팩트 크기를 ~75% 절감하여 예산 여유 확보.

## 베이스
`2026-03-24_2100_11L_LeakyReLU_Bigram6144/train_gpt.py` (Exp 73, val_bpb 1.1221)

## 변경 사항

| 파라미터 | Exp 73 | Exp 85 | 근거 |
|----------|--------|--------|------|
| MLP quantization | int6 | **Ternary {-1,0,1}** | 2 bits/weight (vs 6 bits) |
| Attention quantization | int6 | int6 (유지) | Attention은 precision 중요 |
| Late ternary threshold | - | **0.25** | int6 QAT(0.15)보다 일찍 시작 |
| Late QAT threshold | 0.15 | 0.15 (유지) | Attention int6 QAT 유지 |

## 구현 상세

### Training-time (QAT with STE)
```python
# MLP CastedLinear forward:
scale = w.abs().mean(dim=1)           # per-row scale (BitNet b1.58 방식)
w_q = clamp(round(w / scale), -1, 1)  # ternary quantization
w = w + (w_q * scale - w).detach()    # STE: forward는 quantized, backward는 full precision
```

### Artifact serialization
- Ternary packing: {-1,0,1} → {0,1,2} → 4 values per byte (2 bits each)
- Per-row float16 scale 저장
- **MLP 크기**: ~2 bits/param + scale ≈ int6 대비 ~3x 압축

### 크기 예산
- MLP params (11L): ~17.3M × 2 bits = **~4.3MB** (int6: ~13.0MB)
- Attention params: ~8.7M × 6 bits = **~6.5MB** (변경 없음)
- 기타 (embed, bigram, etc.): ~2.0MB
- **예상 총**: ~12.8MB (vs 기존 ~15.5MB, **~2.7MB 절약**)

## 환경 변수
```bash
TERNARY_MLP=1                  # MLP ternary QAT 활성화 (기본: 1)
LATE_TERNARY_THRESHOLD=0.25    # ternary QAT 활성화 시점 (기본: 0.25)
LATE_QAT_THRESHOLD=0.15        # int6 QAT for attention (기본: 0.15)
```

## 예상 결과
- **크기**: ~12.8MB (16MiB 이내, ~3.9MB 여유)
- **val_bpb**: 1.1250~1.1350 (ternary 품질 손실 예상)
- **위험**: 높음
  - Ternary(3 levels) vs int6(64 levels): 표현력 21배 감소
  - 작은 모델에서 각 weight의 precision이 더 중요
  - Muon optimizer와 ternary gradient 호환성 미검증
- **성공 시**: 여유 예산으로 레이어 추가 (14-16L) 가능

## 실제 결과
(학습 후 기록 예정)
