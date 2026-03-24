# 11L Combined: LeakyReLU + Bigram4096 + TrigramHash + Weighted SWA (Exp 75)

## 목적
Exp 73, 74의 검증된 개선들을 결합하고 TrigramHash를 추가하여 최대 시너지 달성.

## 베이스
`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

## 변경 사항

| 파라미터 | SOTA #1 | Exp 75 | 근거 |
|----------|---------|--------|------|
| MLP activation | relu² | **LeakyReLU(0.5)²** | dead neuron 제거 |
| BigramHash buckets | 2048 | **4096** | 충돌 감소 (trigram 공간 확보) |
| TrigramHash | 없음 | **4096 buckets, dim=64** | 3-token 패턴 포착 |
| SWA 수집 | uniform (dead code) | **Weighted (linearly increasing)** | 최신 checkpoint 강조 |
| SWA 적용 | 미적용 | **EMA + Weighted SWA 블렌딩 (0.5:0.5)** | 상보적 활용 |

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
- val_bpb: ~1.1180~1.1215
- 크기: ~15.96MB (16MB 이내)
- 위험: 낮음~중간

## 실제 결과
(학습 후 기록 예정)
