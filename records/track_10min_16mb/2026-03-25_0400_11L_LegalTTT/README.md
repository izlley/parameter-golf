# Exp 80: Legal Score-First TTT (11L LeakyReLU Base)

## 목적
SOTA#2 (1.1194)의 핵심 기법인 Legal Score-First TTT를 Exp 73 베이스에 추가.
eval 시 chunk 기반 score→train adaptation으로 추가 bpb 개선.

## 베이스
`2026-03-24_2100_11L_LeakyReLU_Bigram6144/train_gpt.py` (Exp 73, val_bpb 1.1221)

## 변경 사항

| 파라미터 | Exp 73 | Exp 80 | 근거 |
|----------|--------|--------|------|
| TTT | 없음 | **Legal Score-First TTT** | SOTA#2의 핵심 기여 기법 |
| TTT LR | - | 0.002 | SOTA#2 기본값 |
| TTT Epochs | - | 3 | SOTA#2 기본값 |
| TTT Chunk | - | 32768 tokens | SOTA#2 기본값 |
| TTT Freeze | - | 첫 2 블록 | 얕은 레이어는 일반적 → 고정 |

## 이전 TTT 실험과의 차이
- Exp 2 (LoRA TTT, 1.1928): rank-8 LoRA, 비효과적
- Exp 30 (LoRA TTT Stride16, ~1.1437): LoRA, 20x 느린 eval
- **이번**: LoRA 아닌 full SGD, chunk-level score→train, "Legal" (미래 토큰 미사용)

## Legal TTT 알고리즘
1. Validation 데이터를 chunk(32768 tokens)로 분할
2. 각 chunk에 대해:
   a. Phase 1 (Score): 슬라이딩 윈도우로 chunk의 모든 토큰 scoring (inference_mode)
   b. Phase 2 (Train): scored chunk에 대해 SGD(lr=0.002, momentum=0.9) 3 epochs 학습
3. 마지막 chunk은 score만 (train 없음 — legality 보장)
4. 적응이 누적됨 (chunk 간 모델 상태 유지)

## 크기 예산
- 변경 없음 (TTT는 eval-time only). **~16.52MB**

## 예상 결과
- val_bpb: ~1.1190~1.1210 (-0.001~0.003 bpb)
- eval 시간: 학습 시간 + TTT eval 시간 = 10분 이내 필요
- 위험: 중간 (eval 시간 초과 가능)

## 실제 결과
(학습 후 기록 예정)
