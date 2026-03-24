# 10L Multi-Exit V2: Eval-Only Ensemble (Exp 68)

## 목적
Exp 66(Multi-Exit)의 교훈을 반영: aux_loss 없이 **평가 시에만** 중간 레이어 logits를 앙상블.
학습은 SOTA와 완전히 동일하게 진행하되, 평가 시 decoder 중간 hidden state에 final_norm + projection을 적용해 다양한 logits를 생성.

## 핵심 변경 (vs 원래 Exp 68 설계)
- **aux_loss 완전 제거**: Exp 66에서 aux_loss가 학습을 방해하는 것이 확인됨 (+0.028 bpb)
- **별도 AuxNorm 제거**: final_norm 하나를 모든 exit에 공유 (완전 weight tying)
- **학습 가능 가중치 → 고정 가중치**: 학습 시 aux_loss 없으므로 가중치 학습 불가, 고정 버퍼로 전환

## 모델 구조

```
Input Tokens
    |
    v
[tok_emb + BigramHash + SmearGate]
    |
    v  x0
+===================================+
|       ENCODER (Layer 0~4)         |
|  Layer 0~4 → skip[0]~skip[4]     |
+================|==================+
                 |
                 v
+=======================================================+
|       DECODER (Layer 5~9)                             |
|                                                       |
|  Layer 5  (dec_idx=0) + skip[4]                       |
|  Layer 6  (dec_idx=1) + skip[3]                       |
|  Layer 7  (dec_idx=2) + skip[2]                       |
|      |                                                |
|      +----> [FinalNorm] --> tok_emb.weight --> EXIT_0 |
|      |                                                |
|  Layer 8  (dec_idx=3) + skip[1]                       |
|      |                                                |
|      +----> [FinalNorm] --> tok_emb.weight --> EXIT_1 |
|      |                                                |
|  Layer 9  (dec_idx=4) + skip[0]                       |
|      |                                                |
|      +----> [FinalNorm] --> tok_emb.weight --> MAIN   |
+=======|===============|===============|===============+
        |               |               |
        v               v               v
      w[0]*EXIT_0 + w[1]*EXIT_1 + w[2]*MAIN
                        |
                        v
                  Final Logits
```

## 학습 시 (forward)

```
Loss = Main_CE  (표준 학습, aux_loss 없음)
```

- SOTA (Exp 65)와 완전히 동일한 학습
- exit logits는 학습에 참여하지 않음

## 평가 시 (forward_logits)

```python
# 고정 가중치: [1/3, 1/2, 1] → 정규화 후 사용
w = exit_logit_weights / exit_logit_weights.sum()
logits = w[0]*EXIT_0 + w[1]*EXIT_1 + w[2]*MAIN
```

- EXIT_0: Layer 7 출력에 final_norm + tok_emb.weight projection
- EXIT_1: Layer 8 출력에 final_norm + tok_emb.weight projection
- MAIN: Layer 9 출력에 final_norm + tok_emb.weight projection
- 가중치 비율: MAIN에 가장 높은 비중 (1/(1/3+1/2+1) ≈ 0.545)

## Weight Tying 분석
- **Projection**: 모든 exit가 tok_emb.weight 공유 (기존 tie_embeddings)
- **Normalization**: RMSNorm은 learnable parameter 없음 → final_norm 공유 가능
- **추가 파라미터**: 0 (고정 buffer 3개만, 학습되지 않음)

## vs Exp 66 (Multi-Exit V1) 차이점

| | Exp 66 (V1) | Exp 68 (V2) |
|--|------------|-------------|
| 학습 시 aux_loss | 있음 (0.1) | **없음** |
| Exit Norm | 별도 AuxNorm | **FinalNorm 공유** |
| 앙상블 가중치 | 단순 평균 | **역 거리 가중 (고정)** |
| 추가 파라미터 | ~0 | **0** |
| 학습 방해 | 심각 (+0.028) | **없음 (동일 학습)** |

## 설계 근거
- Exp 66 실패 원인: aux_loss가 학습을 방해 → 제거
- 중간 레이어 표현도 final_norm 후에는 합리적인 logits 생성 가능
- MAIN에 높은 가중치를 줘서 안전하게 앙상블

## 예상 결과
- val_bpb: ~1.139~1.143 (학습은 SOTA 동일, 앙상블 이점 불확실)
- 크기: ~15.89MB (SOTA와 동일)
- 위험: 낮음 (학습 방해 없음, 최악의 경우 SOTA와 동일)

## 실제 결과
(학습 후 기록 예정)
