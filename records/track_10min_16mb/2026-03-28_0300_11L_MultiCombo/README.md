# Exp 143: NorMuon + Multi-technique Combo

## 목적
여러 약한 크기 절감 기법을 동시 적용: EWQ(layers 0-3) + Pruning 8% + Bigram 4096. 각각의 bpb 비용은 작지만 크기 절감은 합산.

## 베이스
Exp 131 (NorMuon+EWQ, val_bpb 1.1296, 15,554,069B)

## 변경 사항
- EWQ int5: layers 0-5 -> **layers 0-3** (4개 레이어만)
- Magnitude pruning: **8%** (post-training, EMA 후)
- BigramHash: 8192 -> **4096** (해시 테이블 축소)

## 예상 결과
- val_bpb: ~1.122-1.125 (약한 기법 합산)
- 크기: ~14.5-15.5MB (16MB 여유 있게 통과)
- step_avg: ~95ms

## 실제 결과
_(예정)_
