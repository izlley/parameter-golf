# Exp 160: TTT LoRA + Mask Hybrid

## 목적
Channel mask의 표현력 한계를 rank-1 LoRA adapter로 보완. Mask = "어떤 채널을 쓸까", LoRA = "출력을 어떻게 보정할까"

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3, 1.1181 bpb)

## 변경 사항
1. 각 block에 rank-1 LoRA 추가: MLP proj + Attn proj
2. params/block: 4 → 8 (mask+bias 4개 + LoRA A,B 4개)
3. 총 TTT 파라미터: ~45K → ~90K
4. ttt_lr: 0.005 → 0.003 (안정성)
5. LoRA init: randn * 0.01 (near-zero start)

## 예상 결과
- val_bpb: ~1.1155-1.1170 (weight 공간 미세 조정 효과)
- eval 시간: ~520-550s (+5~10% LoRA matmul overhead)

## 실제 결과
_(실험 후 기록)_
