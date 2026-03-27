# Exp 145: NorMuon + SupermaskTTT V3 (TTT A)

## 목적
NorMuon 베이스에 SupermaskTTT V3를 적용하여 절대 최고 bpb 탐색. 크기 제한 무시, 품질 한계 실험.

## 베이스
Exp 115 (SupermaskTTT V3, val_bpb 1.1196/TTT 1.1196) + Exp 121 (NorMuon 기법)

## 변경 사항
- Muon optimizer에 NorMuon 적용: NS5 후 `g = F.normalize(g, dim=1)` 추가
- TTT 설정은 V3 최적 config 유지 (lr=0.005, epochs=3)

## 예상 결과
- val_bpb (sliding): ~1.1180 (NorMuon 효과)
- val_bpb (TTT): ~1.1175-1.1180 (TTT 추가 개선)
- 크기: ~17.1MB (초과 -- 크기 제한 무시)
- step_avg: ~95ms
