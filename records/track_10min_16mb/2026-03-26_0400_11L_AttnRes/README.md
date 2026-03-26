# Exp 103: Attention Residuals (Full AttnRes)

## Purpose
U-Net skip connections (encoder/decoder split)은 cross-layer 정보 전달에 한계가 있음:
- 고정된 대칭 구조 (layer 0↔10, 1↔9, ...)
- skip_weights는 학습되지만 연결 토폴로지는 고정
- 실제 어떤 레이어의 정보가 가장 유용한지 동적으로 결정 불가

Attention Residuals (Kimi Team, 2026)로 U-Net skip을 대체하여
각 레이어가 모든 이전 레이어 출력에 대해 학습된 softmax attention으로
최적의 입력을 자동 구성하도록 함.

## Base
Exp 93: GatedAttn PerHead (`2026-03-25_1800_11L_GatedAttn_PerHead`, val_bpb=1.1198)

## Approach
- **Full AttnRes**: 각 block `l`이 `sources[0..l-1]`에 대해 softmax attention 수행
  - `sources[0]` = embedding output
  - `sources[i+1]` = block i output
- **AttnResOp**: per-layer pseudo_query (dim=512, zero init) + RMSNorm (no learnable weight)
  - Zero init → 학습 초기에는 uniform averaging (= standard residual stream에 근사)
  - 학습 진행에 따라 각 레이어가 최적 source 조합을 학습
- **U-Net skip 제거**: encoder/decoder split, skip_weights 제거
- Block 내부 residual connection (resid_mix, attn_scale, mlp_scale)은 유지
- torch.compile 호환: 고정 크기 source buffer + bool mask로 dynamic shape 회피

## Key Changes vs Exp 93
- `AttnResOp` class 추가 (pseudo_query + RMSNorm no weight)
- `GPT.__init__`: `attn_res_ops` ModuleList 추가, encoder/decoder split 제거
- `GPT.forward`, `forward_logits`: U-Net skip → AttnRes loop
- `CONTROL_TENSOR_NAME_PATTERNS`에 `pseudo_query` 추가
- Optimizer: attn_res pseudo_query를 scalar_params에 등록
- Env var: `ATTN_RES=1` (default enabled)

## Expected Results
- SW baseline: ~1.1180-1.1200 (U-Net skip 대비 동등 또는 개선)
- Size: ~15.55MB (pseudo_query 11x512 = 5,632 params, 무시 가능)
- 학습 시간: 약간 증가 (AttnRes softmax 연산 O(L^2*d) 추가, L=11로 작아 영향 미미)
