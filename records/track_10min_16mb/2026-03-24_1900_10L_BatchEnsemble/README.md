# 10L BatchEnsemble (Eval-Only) (Exp 71)

## 목적
각 decoder layer에 rank-1 perturbation vector를 적용하여 eval 시 다양한 앙상블 멤버를 생성.
학습 시에는 perturbation 없이 표준 학습 수행. perturbation vector는 1.0 근처에 작은 noise로 초기화.

## 베이스라인
- Weighted SWA (Exp 65, NEW SOTA val_bpb 1.14136)

## 모델 구조

```
ensemble_r: shape [num_members=2, num_decoder_layers=5, model_dim=512]
초기화: 1.0 + small noise

┌──────────────────────────────────────────────────┐
│              Decoder Layer i                     │
│                                                  │
│  입력 x                                          │
│    │                                             │
│    ├─── (학습) ──────────────► 그대로 통과         │
│    │                                             │
│    ├─── (eval, main) ────────► 그대로 통과         │
│    │                                             │
│    ├─── (eval, member m) ──► x * ensemble_r[m,i] │
│    │                         (element-wise)      │
│    ▼                                             │
│  block(x)  →  다음 layer                         │
└──────────────────────────────────────────────────┘
```

### Eval 시 전체 흐름

```
입력 x
  │
  ├──► main path (no perturbation)
  │    decoder_layer_0 → ... → decoder_layer_4 → logits_main
  │
  ├──► member 0
  │    x*r[0,0] → block → x*r[0,1] → block → ... → logits_0
  │
  ├──► member 1
  │    x*r[1,0] → block → x*r[1,1] → block → ... → logits_1
  │
  ▼
logits = (logits_main + logits_0 + logits_1) / 3
```

### 추가 파라미터
- `ensemble_r`: 2 x 5 x 512 = 5120 파라미터 (무시 가능)

## 학습 시 (forward)
- 표준 학습 (Exp 65와 동일)
- perturbation 적용 없음 (`member_idx = -1`)
- ensemble_r은 gradient 업데이트 없음 (초기화된 값 유지)

## 평가 시 (forward_logits)
1. main path: perturbation 없이 표준 forward → `logits_main`
2. member m (m=0,1): 각 decoder layer i에서 `x = x * ensemble_r[m, i]` 적용 후 block → `logits_m`
3. 3개 logits 평균: `logits = (logits_main + logits_0 + logits_1) / 3`
4. decoder를 3번 실행 (main + 2 members)

## 설계 근거
- Exp 66: 학습 시 aux_loss가 catastrophic → eval-only perturbation으로 학습 영향 제거
- BatchEnsemble (Wen et al.): rank-1 perturbation으로 저비용 앙상블 구현
- 1.0 근처 초기화 → perturbation이 작아 원본 모델 성능 보존
- random member selection per step → torch.compile fullgraph 호환 불가 → eval-only 선택
- eval 시 3x decoder 비용이지만, decoder 자체는 전체 eval 대비 빠름

## 예상 결과
- val_bpb: ~1.139-1.143
- 크기: ~15.9MB (SOTA와 동일)
- eval 시간: ~3x decoder 비용 (허용 범위)
- 위험: eval-only perturbation은 학습 시 다양성 훈련이 없어 효과 제한적일 수 있음; perturbation vector가 명시적으로 다양성을 위해 학습되지 않음

## 실제 결과
(학습 후 기록 예정)
