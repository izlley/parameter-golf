# Exp 81: DDP-Free Manual AllReduce (11L LeakyReLU Base)

## 목적
DDP 오버헤드 제거로 step_avg 절감. DDP 래핑 대신 manual all-reduce로 gradient 동기화.

## 베이스
`2026-03-24_2100_11L_LeakyReLU_Bigram6144/train_gpt.py` (Exp 73, val_bpb 1.1221)

## 변경 사항

| 파라미터 | Exp 73 | Exp 81 | 근거 |
|----------|--------|--------|------|
| Gradient sync | DDP | **Manual all-reduce** | DDP 오버헤드 ~10ms 제거 |
| Model wrapping | DDP(compiled_model) | compiled_model 직접 사용 | 불필요한 래핑 제거 |
| Warmup grad sync | DDP require_backward_grad_sync | manual all-reduce | DDP 제거에 따른 변경 |

## 구현 핵심
- DDP 래핑 제거, torch.compile만 사용
- backward() 후 모든 파라미터에 대해 dist.all_reduce(p.grad, AVG) 호출
- Muon optimizer는 이미 자체 all-reduce 포함 → matrix_params 제외
- 나머지 params (embedding, scalar, bigram, VE 등)만 manual all-reduce

## 크기 예산
- 변경 없음. **~16.52MB**

## 예상 결과
- step_avg: ~82-85ms (현재 92ms에서 -7~10ms)
- 추가 steps: +500~700 steps
- val_bpb: ~1.1200~1.1220 (-0.001~0.002 bpb)
- 위험: 중간 (gradient 동기화 구현 정확성)

## 실제 결과
(학습 후 기록 예정)
