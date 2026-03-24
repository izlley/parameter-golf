# 11L VRL (Value Residual Learning) + LeakyReLU(0.5)² (Exp 76)

## 목적
Value Residual Learning으로 attention sink 방지 + LeakyReLU(0.5)² 결합.

## 베이스
`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

## 변경 사항

| 파라미터 | SOTA #1 | Exp 76 | 근거 |
|----------|---------|--------|------|
| MLP activation | relu² | **LeakyReLU(0.5)²** | dead neuron 제거 |
| VRL | 없음 | **λᵢ * V₀ residual** | attention sink 방지 |

## VRL (Value Residual Learning) 설계
- **논문**: "Value Residual Learning for Alleviating Attention Sink in LLMs"
- **구현**: Layer 0에서 V₀ 캐시 → 이후 레이어에 `λᵢ * V₀` 잔차 추가
- **λᵢ**: per-layer learnable scalar, 초기값 0.1
- **VE 충돌 방지**: Value Embedding이 적용되는 layers 9,10에서는 VRL 비적용
- **효과**: 깊은 레이어에서도 layer 0의 semantic V 정보 유지 → attention 다양성 향상
- **비용**: 11개 스칼라 파라미터 (~44 bytes), V₀ 캐싱은 forward에서 1회 추가 연산

## Value Embedding과의 관계
- **VE**: 토큰 정체성(vocab embedding)을 V에 주입 → 토큰 "무엇인지" 정보
- **VRL**: hidden state 유래 V₀를 후속 레이어에 전달 → 토큰 "어떤 맥락인지" 정보
- **상보적**: VE는 static token identity, VRL은 dynamic context-dependent V₀

## 예상 결과
- val_bpb: ~1.1195~1.1225
- 크기: ~15.55MB (변경 무시)
- 위험: 중간 (V₀ 캐싱 구현, torch.compile 호환성)

## 실제 결과
(학습 후 기록 예정)
