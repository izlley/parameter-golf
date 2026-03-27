# Exp 158: TTT Multi-site Masking

## 목적
QKV projection output과 attention output에 추가 mask/gate를 적용하여 TTT(Test-Time Training) Supermask의 표현력을 증가시킨다. 기존에는 MLP hidden + Attn output에만 mask/bias를 적용했으나, QKV channel mask와 gate bias를 추가하여 더 세밀한 적응이 가능하도록 한다.

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3)

## 변경 사항
- **기존 4 params/block -> 6 params/block**:
  - `mlp_mask` (hidden_dim): MLP hidden channel mask (기존)
  - `mlp_bias` (hidden_dim): MLP hidden additive bias (기존)
  - `attn_mask` (model_dim): Attention output channel mask (기존)
  - `attn_bias` (model_dim): Attention output additive bias (기존)
  - `qkv_mask` (q_dim + k_dim + v_dim): **NEW** - Q, K, V projection output에 per-channel sigmoid mask 적용
  - `gate_bias` (model_dim): **NEW** - Attention output에 추가 multiplicative sigmoid gate 적용
- **QKV mask**: `c_q`, `c_k`, `c_v` forward를 monkey-patch하여 각 projection output에 channel-wise sigmoid mask 적용 (init=3.0, sigmoid~0.95)
- **Gate bias**: Attention output에 기존 attn_mask/bias 적용 후 추가 multiplicative gate 적용 (init=3.0)
- 복원 로직에 `c_q`, `c_k`, `c_v` forward 복원 추가

## 예상 결과
- val_bpb: 기존 대비 -0.001 ~ -0.003 개선
- step time: +5~10% (추가 sigmoid 연산)
- artifact size: 변동 없음 (TTT mask는 평가 시에만 사용)

## 실제 결과
_(실험 후 기록)_
