# 10L Delta Compression Ensemble (Exp 69)

## 목적
SWA 모델과 last 모델을 단일 16MB artifact에 함께 저장하여 eval 시 두 모델의 logit 앙상블 수행.
delta (last - SWA)를 int2 (3레벨: -1, 0, 1)로 압축하여 용량 오버헤드를 최소화.

## 베이스라인
- Weighted SWA (Exp 65, NEW SOTA val_bpb 1.14136)

## 모델 구조

학습은 Exp 65와 완전히 동일. 후처리(post-training)에서 delta 압축 수행.

```
┌─────────────────────────────────────────────────────┐
│                Post-Training Pipeline               │
│                                                     │
│  last_state ─────────┐                              │
│                      ├─► delta = last - SWA         │
│  SWA_state ──────────┘         │                    │
│      │                         │                    │
│      ▼                         ▼                    │
│  int5(MLP)/int6(Attn)    int2 per-row quantize      │
│  + zstd compress         normalize by row_max       │
│      │                   round to {-1, 0, 1}        │
│      │                   store as int8 + fp16 scale │
│      │                   + zstd-22 compress         │
│      │                         │                    │
│      ▼                         ▼                    │
│  ┌─────────┐            ┌───────────┐               │
│  │ SWA blob│            │delta blob │               │
│  │ ~14.5MB │            │  ~1-2MB   │               │
│  └────┬────┘            └─────┬─────┘               │
│       └──────────┬────────────┘                     │
│                  ▼                                   │
│          artifact (≤ 16MB)                          │
└─────────────────────────────────────────────────────┘
```

## 학습 시 (forward)
Exp 65와 완전히 동일. 변경 없음.

## 후처리 (post-training)
1. SWA 적용 전 `last_state` 저장
2. SWA 모델을 표준 int5(MLP)/int6(Attn) + zstd로 양자화/압축
3. `delta = last_state - SWA_state` 계산
4. delta를 int2 per-row 양자화:
   - 각 row의 max abs 값으로 normalize
   - {-1, 0, 1} 3레벨로 반올림
   - int8 텐서 + fp16 scale factor로 저장
5. delta를 zstd-22로 압축
6. SWA blob (~14.5MB) + delta blob (~1-2MB) → 단일 artifact

## 평가 시 (forward_logits)
1. SWA 모델 복원 (표준 역양자화)
2. last 모델 복원: `last = SWA + delta` (delta 역양자화 후 덧셈)
3. 두 모델로 각각 forward → logits 평균

```
입력 x
  │
  ├──► SWA model  ──► logits_swa
  │
  ├──► last model ──► logits_last
  │    (SWA + delta)
  │
  ▼
logits = (logits_swa + logits_last) / 2
```

## 설계 근거
- Exp 40 (LogitEnsemble): SWA + last 앙상블이 best val_bpb → 두 체크포인트의 다양성이 도움
- Exp 65: Weighted SWA가 SOTA → SWA 모델은 이미 최적
- delta를 int2로 압축하면 용량 오버헤드 ~1-2MB로 16MB 예산 내 유지 가능
- 학습 코드 변경 없음 → 위험 최소

## 예상 결과
- val_bpb: ~1.137-1.141 (두 체크포인트 앙상블이 도움되면 개선)
- 크기: ~15.5-16.0MB (SWA ~14.5MB + delta ~1-2MB)
- 위험: int2 delta가 너무 lossy할 수 있음; SWA가 이미 trajectory를 포착하면 delta가 다양성을 추가하지 못할 수 있음

## 실제 결과
(학습 후 기록 예정)
