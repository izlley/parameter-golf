# 11L Combined: LeakyReLU + Bigram4096 + TrigramHash (Exp 75 v2)

## 목적
Exp 73의 검증된 LeakyReLU(0.5)²에 TrigramHash를 추가하여 3-token 패턴 포착으로 추가 개선.

## 베이스
`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

## 변경 사항

| 파라미터 | SOTA #1 | Exp 75 v2 | 근거 |
|----------|---------|-----------|------|
| MLP activation | relu² | **LeakyReLU(0.5)²** | dead neuron 제거 (Exp 73에서 검증) |
| BigramHash buckets | 2048 | **4096** | 충돌 감소 (trigram 공간 확보) |
| TrigramHash | 없음 | **4096 buckets, dim=64** | 3-token 패턴 포착 |
| SWA/블렌딩 | EMA만 (SWA dead code) | **EMA만** | Weighted SWA 블렌딩 효과 없음 확인 (Exp 74) |

## v1 → v2 변경점
- Weighted SWA + EMA 블렌딩 제거 → EMA만 사용 (Exp 74 결과: 블렌딩 효과 없음)
- SWA 수집 코드 제거 (CPU 메모리 절약, 오버헤드 제거)

## TrigramHash 설계
- Hash: `(48271 * t[i] XOR 36313 * t[i-1] XOR 27191 * t[i-2]) % (vocab-1)`
- 4096 buckets, dim=64 → model_dim(512) 프로젝션
- scale 초기값 0.03 (bigram의 0.05보다 낮게)
- 위치 0,1은 trigram 불가 → 기본 bucket 사용

## 크기 예산
- BigramHash 4096: (4096-2048) × 128 × 6bit/8 ≈ 196KB
- TrigramHash 4096×64: 4096 × 64 × 6bit/8 ≈ 192KB
- Trigram proj 64→512: 64 × 512 × 6bit/8 ≈ 24KB
- 총 추가: ~412KB → 15.55 + 0.41 ≈ **~15.96MB**

## 예상 결과
- val_bpb: ~1.1190~1.1220 (Exp 73 1.1221 기준 trigram 추가 이득 기대)
- 크기: ~15.96MB (16MiB 이내)
- 위험: 낮음

## 실제 결과
(학습 후 기록 예정)
