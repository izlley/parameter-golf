# 10L Multi-Exit V2: Weighted Ensemble (Exp 68)

## 목적
Exp 66(Multi-Exit)을 개선: (1) exit 위치를 마지막 레이어 근처로 이동, (2) 단순 평균 대신 학습 가능한 가중 평균.
Exp 40(SWA+Last logit 앙상블, 1.1365)의 효과를 단일 모델로 내재화.

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
|                                   |
|  Layer 0 ---------> skip[0]       |
|  Layer 1 ---------> skip[1]       |
|  Layer 2 ---------> skip[2]       |
|  Layer 3 ---------> skip[3]       |
|  Layer 4 ---------> skip[4]       |
+================|==================+
                 |
                 v
+===================================================+
|       DECODER (Layer 5~9)                         |
|                                                   |
|  Layer 5  (dec_idx=0) + skip[4]                   |
|      |                                            |
|  Layer 6  (dec_idx=1) + skip[3]                   |
|      |                                            |
|  Layer 7  (dec_idx=2) + skip[2]                   |
|      |                                            |
|      +------> [AuxNorm_2] --> projection          |
|      |              |                             |
|      |         EXIT_0 logits ---+                 |
|      |                         |                  |
|  Layer 8  (dec_idx=3) + skip[1]|                  |
|      |                         |                  |
|      +------> [AuxNorm_3] --> projection          |
|      |              |         |                   |
|      |         EXIT_1 logits -+-+                 |
|      |                        | |                 |
|  Layer 9  (dec_idx=4) + skip[0] |                 |
|      |                        | |                 |
|      v                        | |                 |
|  [FinalNorm] --> projection   | |                 |
|      |                        | |                 |
|  MAIN logits --------+-------+ |                  |
+==========================|===|==|=================+
                           |   |  |
                           v   v  v
                  +---------------------+
                  | Learnable Weighted  |
                  |      Average        |
                  |                     |
                  | w = softmax([w0,    |
                  |        w1, w_main]) |
                  |                     |
                  | out = w0 * EXIT_0   |
                  |     + w1 * EXIT_1   |
                  |     + w_main * MAIN |
                  +----------+----------+
                             |
                             v
                       Final Logits
```

## 학습 시 (forward)

```
Total Loss = Main_CE + 0.1 * mean(EXIT_0_CE, EXIT_1_CE)
```

- 보조 exit의 CE loss가 Layer 7, 8에 직접적인 gradient 전달 (deep supervision)
- aux_loss_weight=0.1로 main task 방해 최소화
- exit_logit_weights도 학습되어 최적 가중치 자동 결정

## 평가 시 (forward_logits)

```
w = softmax(exit_logit_weights)   # 학습된 3개 scalar
logits = w[0]*EXIT_0 + w[1]*EXIT_1 + w[2]*MAIN
```

## vs Exp 66 (Multi-Exit V1) 차이점

| | Exp 66 (V1) | Exp 68 (V2) |
|--|------------|-------------|
| Exit 위치 | decoder_idx 1, 3 (Layer 6, 8) | decoder_idx **2, 3** (Layer 7, 8) |
| 앙상블 방식 | 단순 평균 (1/3) | **학습 가능 weighted avg** (softmax) |
| 가중치 초기값 | N/A | [0, 0, 1] (main 편향) |
| 추가 파라미터 | ~0 | **scalar 3개** (무시 가능) |

## 설계 근거
- Layer 7, 8은 표현이 거의 완성된 상태 → 고품질 logits 3개로 앙상블
- Layer 6(V1)은 아직 표현 미숙 → 평균을 끌어내릴 가능성
- 학습 가능 가중치로 모델이 각 exit의 최적 비중을 자동 결정
- 초기값 [0,0,1] → softmax 후 main~0.58 → 안전한 출발점

## 예상 결과
- val_bpb: ~1.138~1.142
- 크기: ~15.9MB (SOTA와 동일, scalar 3개 추가만)
- 위험: 중간 (auxiliary loss + weighted avg의 조합 효과 미지수)

## 실제 결과
(학습 후 기록 예정)
