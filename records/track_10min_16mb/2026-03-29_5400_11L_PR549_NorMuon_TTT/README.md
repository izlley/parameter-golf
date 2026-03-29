# Experiment 194: PR#549 base + NorMuon + SupermaskTTT

## 목적
PR#549 SOTA 모델에 NorMuon 최적화와 SupermaskTTT를 결합하여 val_bpb를 개선한다.
- NorMuon: Newton-Schulz 후 neuron-wise L2 정규화로 학습 안정성 향상
- SupermaskTTT: 전체 파라미터를 동결하고 경량 mask+bias만 학습하여 효율적인 TTT 수행

## 베이스
PR#549 (Merged SOTA) - Parameter Banking + full-parameter TTT

## 변경 사항

### Change 1: NorMuon
- Muon optimizer의 `step()` 메서드에서 `zeropower_via_newtonschulz5()` 호출 후 neuron-wise L2 normalization 추가
- 3D tensor (banked params): `F.normalize(update, dim=-1)`
- 2D tensor: `F.normalize(update, dim=1)`

### Change 2: SupermaskTTT (replaces full-parameter TTT)
- `MuonLiteTTT` optimizer: sign-based update with momentum, lightweight
- 모든 모델 가중치를 동결하고 블록별 mask+bias 파라미터만 학습
  - MLP: hidden activation (leaky_relu().square() 이후)에 mask+bias 적용
  - Attention: output에 mask+bias 적용
- Score-first legality: Phase 1에서 scoring, Phase 2에서 mask 학습
- monkey-patching으로 Parameter Banking forward에 mask 적용
- 학습 후 원본 forward 복원

### TTT Hyperparameters
- ttt_lr: 0.002
- ttt_epochs: 3
- MuonLiteTTT: momentum=0.9

## 예상 결과
- val_bpb: ~0.6680 (NorMuon 개선 + SupermaskTTT 효율적 적응)
- step time: 베이스와 동일 (NorMuon은 normalize만 추가)
- artifact size: 16MB 이하 (mask는 inference 시에만 사용, 저장 불필요)

## 실제 결과
_(실험 후 작성)_
