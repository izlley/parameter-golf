# 11L LeakyReLU(0.5)² + BigramHash 6144 (Exp 73)

## 목적
리더보드 SOTA #1 (signalrush, val_bpb 1.1228)을 베이스로 두 가지 low-risk 개선 적용.

## 베이스
`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- 11L, 512dim, MLP 3x, EMA(0.997), GPTQ-lite, XSA(last 4), Partial RoPE(16/64), LN Scale

## 변경 사항

| 파라미터 | SOTA #1 | Exp 73 | 근거 |
|----------|---------|--------|------|
| MLP activation | relu² | **LeakyReLU(0.5)²** | dead neuron 제거, 실질 용량 ~25% 증가 |
| BigramHash buckets | 2048 | **6144** | 해시 충돌 감소, 1.16MB 여유 활용 |

## 설계 근거
- **LeakyReLU(0.5)²**: ReLU²는 음수 입력을 완전히 버림 (50% dead). LeakyReLU(0.5)는 음수 입력의 50%를 통과시키고, 이를 제곱하면 25% 기여. 파라미터 추가 없이 MLP 표현력 증가. NanoGPT speedrun 커뮤니티에서 검증된 기법.
- **BigramHash 6144**: SOTA #1은 2048 버킷 사용. 우리의 이전 SOTA는 10240. 여유 공간(1.16MB)을 활용하여 6144로 확대. 해시 충돌 확률 ~67% 감소.

## 예상 결과
- val_bpb: ~1.1190~1.1225
- 크기: ~15.9MB (16MB 이내)
- 위험: 매우 낮음

## 실제 결과
(학습 후 기록 예정)
