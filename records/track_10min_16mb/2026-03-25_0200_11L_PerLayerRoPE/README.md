# 11L Per-Layer RoPE/NoPE + LeakyReLU(0.5)² (Exp 78)

## 목적
레이어별로 RoPE dims를 다르게 설정하여 얕은 레이어는 위치 정보 강화, 깊은 레이어는 NoPE로 semantic matching 최적화.

## 베이스
`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

## 변경 사항

| 파라미터 | SOTA #1 | Exp 78 | 근거 |
|----------|---------|--------|------|
| MLP activation | relu² | **LeakyReLU(0.5)²** | dead neuron 제거 |
| RoPE dims | 전 레이어 16 (25%) | **Per-layer: 32,32,32,32,16,16,16,16,0,0,0** | 레이어별 최적 위치 인코딩 |

## Per-Layer RoPE 설계
- **Layer 0-3** (구문/문법): `rotary_dim=32` (50% RoPE) — 위치 정보 중요
- **Layer 4-7** (중간 표현): `rotary_dim=16` (25% RoPE) — 기존과 동일
- **Layer 8-10** (의미/추론): `rotary_dim=0` (NoPE) — content-based attention 극대화
- NoPE 레이어에서는 `apply_rotary_emb` 호출 자체를 건너뜀
- 환경변수 `PER_LAYER_ROPE_DIMS`로 커스터마이즈 가능

## 근거
- 얕은 레이어: 토큰 순서/문법 구조 학습 → 위치 정보 필수
- 깊은 레이어: 의미적 유사성 기반 attention → 위치 정보가 오히려 방해 가능
- NoPE 논문들에서 position-free attention이 semantic matching에 유리하다는 결과
- Partial RoPE 자체가 NoPE+RoPE 혼합의 일종 → 레이어 차원으로 확장

## 예상 결과
- val_bpb: ~1.1200~1.1230
- 크기: ~15.55MB (변경 없음, 파라미터 추가 0)
- 위험: 낮음 (기존 Partial RoPE의 자연스러운 확장)

## 실제 결과
(학습 후 기록 예정)
