# Exp 195: PR#1019 Base + NorMuon + GatedAttention + SupermaskTTT

## 목적
abaybektursun의 PR#1019 (Parameter Banking + Full GPTQ + XSA all) 위에
우리의 3가지 핵심 기법을 모두 적용한 종합 실험.

## 베이스
abaybektursun PR#1019 (1.1147 bpb, Parameter Banking, GPTQ, XSA all 11 layers)
→ Exp 192 (PR#1019 + NorMuon + TTT) 코드에서 GatedAttn 활성화

## 변경 사항
1. **NorMuon**: Parallel Muon NS5 후 `F.normalize(update, dim=-1)` 추가
   - 3D bank tensor: dim=-1, 2D tensor: dim=1
2. **GatedAttention**: `GATED_ATTENTION` 기본값 0 → 1 활성화
   - per-head sigmoid gate on attention output
3. **SupermaskTTT**: MuonLiteTTT + 채널 mask/bias 학습
   - Parameter Banking 구조에 맞춰 monkey-patch 적용
   - ttt_lr=0.002, ttt_epochs=3, chunk=32768

## 예상 결과
- val_bpb: ~1.1100-1.1130 (NorMuon + GatedAttn + TTT 종합 효과)
- step_avg: ~87-90ms (Parameter Banking 효과)
- artifact size: ~16MB (Full GPTQ + LZMA)

## 실제 결과
_(실행 후 기록)_
