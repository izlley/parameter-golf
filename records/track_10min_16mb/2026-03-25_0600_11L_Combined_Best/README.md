# Exp 82: Combined Best (Bigram8192 + Legal TTT + DDP-Free)

## 목적
Exp 79-81의 모든 개선을 결합하여 SOTA#2 (1.1194) 도달 목표.

## 베이스
`2026-03-24_2100_11L_LeakyReLU_Bigram6144/train_gpt.py` (Exp 73, val_bpb 1.1221)

## 변경 사항

| 파라미터 | Exp 73 | Exp 82 | 근거 |
|----------|--------|--------|------|
| BigramHash | 6144 | **8192** | 해시 충돌 감소 (Exp 79) |
| TTT | 없음 | **Legal Score-First TTT** | eval-time adaptation (Exp 80) |
| Gradient sync | DDP | **Manual all-reduce** | step_avg 절감 (Exp 81) |

## 예상 효과 합산
- Bigram 8192: -0.001 bpb
- Legal TTT: -0.002~0.003 bpb
- DDP-free: step_avg -10ms → -0.001 bpb (추가 스텝으로 인한 개선)
- **총 예상**: ~1.1170~1.1200 (SOTA#2 1.1194 근접/돌파 가능)

## 크기 예산
- BigramHash 8192: +~0.20MB
- 총: **~16.7MB** (16MiB 이내)

## 위험
- 높음 (3개 변경 동시 적용). 개별 실험 결과 확인 후 최적 조합 결정 권장

## 실제 결과
(학습 후 기록 예정)
