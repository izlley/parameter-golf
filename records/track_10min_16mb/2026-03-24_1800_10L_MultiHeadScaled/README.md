# 10L Multi-Head Scaled Views (Exp 70)

## 목적
단일 모델 내에서 앙상블 다양성을 확보. dim 크기의 learnable scaling vector로 최종 hidden state를 변조하여 각 "head"가 서로 다른 가중치 view를 보도록 함.
학습 시 aux loss로 scaling vector를 훈련하고, eval 시 모든 head의 logits를 가중 합산.

## 베이스라인
- Weighted SWA (Exp 65, NEW SOTA val_bpb 1.14136)

## 모델 구조

```
                    final_norm(x)
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
     head_scales[0]  head_scales[1]  (identity)
     * final_norm(x) * final_norm(x)  main head
            │            │            │
            ▼            ▼            ▼
       project()    project()    project()
      (tok_emb.W)  (tok_emb.W)  (tok_emb.W)
            │            │            │
            ▼            ▼            ▼
       logits_0     logits_1     logits_main
            │            │            │
            └────────────┼────────────┘
                         ▼
              w = softmax(head_logit_weights)
              logits = Σ w_k * logits_k
```

### 공유 구성 요소
- `tok_emb.weight`: 모든 head가 동일한 projection matrix 공유
- `final_norm`: RMSNorm (learnable param 없음) → 모든 head 공유

### 추가 파라미터
- `head_scales`: 2 x dim (= 2 x 512 = 1024) — aux head별 scaling vector
- `head_logit_weights`: 3 scalars — head별 앙상블 가중치, 초기값 [0, 0, 1] (main 편향)
- 총 추가 파라미터: ~1027개 (무시 가능)

## 학습 시 (forward)

```
loss = main_CE + 0.1 * mean(aux_head_CEs)
```

1. `h = final_norm(x)`
2. main logits: `project(h, tok_emb.weight)` → main CE loss
3. aux head k: `h_k = h * head_scales[k]` → `project(h_k, tok_emb.weight)` → aux CE loss
4. 총 loss = main_CE + 0.1 * mean(aux_head_CEs)
   - aux_loss 가중치 0.1 → Exp 66 대비 보수적

## 평가 시 (forward_logits)

1. `h = final_norm(x)`
2. main logits: `project(h, tok_emb.weight)`
3. aux head k: `h_k = h * head_scales[k]` → `project(h_k, tok_emb.weight)`
4. `w = softmax(head_logit_weights)` → [w_0, w_1, w_main]
5. `logits = w_0 * logits_0 + w_1 * logits_1 + w_main * logits_main`

## 설계 근거
- Exp 66: aux_loss가 catastrophic → 가중치 0.1로 대폭 축소
- Exp 40: SWA + last 앙상블 효과 확인 → 단일 모델 내 다양성 시도
- scaling vector는 representation을 element-wise로 변조 → projection 후 다른 logit 분포 생성
- learnable ensemble weights로 최적 조합을 모델이 스스로 학습
- 추가 파라미터 ~1027개 → 크기/학습 비용 영향 없음

## 예상 결과
- val_bpb: ~1.138-1.142
- 크기: ~15.9MB (SOTA와 동일)
- 위험: scaled view가 충분한 다양성을 생성하지 못할 수 있음; aux_loss가 0.1이라도 간섭 가능성 있으나 Exp 66 대비 위험 낮음

## 실제 결과
(학습 후 기록 예정)
