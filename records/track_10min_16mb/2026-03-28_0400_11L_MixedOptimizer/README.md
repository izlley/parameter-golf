# Exp 144: NorMuon + Mixed Optimizer per Layer

## 목적
초기 레이어(0-4)에 standard Muon, 후기 레이어(5-10)에 NorMuon 적용. 초기 레이어의 압축 친화적 weight 분포를 유지하면서 후기 레이어의 품질을 보장.

## 베이스
Exp 121 (NorMuon, val_bpb 1.1183, 17,158,779B — 크기 초과)

## 변경 사항
- Layers 0-4: Standard Muon (F.normalize 미적용) → 압축 친화적 weight 분포
- Layers 5-10: NorMuon (F.normalize 적용) → 품질 우선
- Non-block params (mtp_heads, bigram.proj, ve_shared.proj): Standard Muon
- Muon.step()에서 per-parameter `_normuon` 플래그로 분기

## 예상 결과
- val_bpb: ~1.119-1.121 (NorMuon 일부 적용으로 소폭 악화)
- 크기: ~16.5-17.0MB (초기 레이어 압축률 개선)
- step_avg: ~95ms (변경 없음)

## 실제 결과
_(실행 후 기록)_
