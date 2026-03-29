# Experiment 192: PR#1019 Base + NorMuon + SupermaskTTT

## 목적
PR#1019 (abaybektursun)의 Parameter Banking 기반 코드에 두 가지 검증된 기법을 적용하여 성능을 극대화한다:
1. NorMuon: Newton-Schulz 직교화 후 neuron-wise L2 정규화로 학습 안정성 향상
2. SupermaskTTT: 추론 시 per-block supermask를 학습하여 validation BPB를 추가 개선

## 베이스
PR#1019 (abaybektursun) - XSA All Layers, Parameter Banking, GPTQ int6 quantization with LZMA compression

## 변경 사항

### Change 1: NorMuon
- Muon optimizer의 `step()` 메서드에서 `zeropower_via_newtonschulz5()` 호출 후 neuron-wise L2 normalization 추가
- 3D 텐서 (batched banks): `F.normalize(update, dim=-1)` (각 행 벡터 정규화)
- 2D 텐서: `F.normalize(update, dim=1)` (각 행 벡터 정규화)

### Change 2: SupermaskTTT
- `MuonLiteTTT` optimizer: SGD+momentum 기반 경량 TTT 옵티마이저
- Per-block supermask parameters: MLP hidden mask (1536) + MLP bias (1536) + Attn mask (512) + Attn bias (512) = 4096/block
- Total TTT params: 11 blocks x 4096 = 45,056 parameters
- Mask init: 3.0 (sigmoid ~= 0.953), bias init: 0.0
- MLP monkey-patch: squared activation 후 sigmoid mask 적용 + bias 추가
- Attn monkey-patch: attention output에 sigmoid mask 적용 + bias 추가
- LR schedule: warmup 5% -> constant 65% -> cosine 30%
- TTT hyperparameters: lr=0.002, epochs=3, chunk_tokens=32768, momentum=0.9, grad_clip=1.0, batch_seqs=32

## 예상 결과
- val_bpb: ~1.108-1.112 (NorMuon으로 training 개선 + TTT로 추가 ~0.002-0.005 BPB 향상)
- step_avg: ~41-43ms (NorMuon overhead 미미)
- artifact size: ~15.9MB (LZMA + selective pruning으로 16MB 제약 충족)
- TTT eval time: 추가 ~60-120s

## 실제 결과
_(실험 후 기록 예정)_
