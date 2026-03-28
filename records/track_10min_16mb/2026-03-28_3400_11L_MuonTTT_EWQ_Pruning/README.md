# Exp 174: MuonTTT + EWQ + Pruning 12% 결합

## 목적 (Purpose)
검증된 두 기법 EWQ(Exp 165, -2.0MB) + Pruning(Exp 166, -0.4MB)을 결합.
예상 총 크기 절감 ~2.4MB → 18.1MB → ~15.7MB로 16MB 통과.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **EWQ**: Early layers (0-5) int5 양자화, late layers (6-10) int6 유지
  - CastedLinear QAT에 int5 분기 (_ewq_int5 flag)
  - quantize_int5_per_row 함수 추가
  - mixed_quantize_int6에서 레이어별 int5/int6 분기
- **Pruning 12%**: EMA 적용 후, 양자화 전에 magnitude pruning
  - 2D weight 중 하위 12% 값을 0으로 설정
  - zstd 압축률 향상
- **결합 순서**: EMA → Pruning → EWQ Quantization → zstd

## 예상 결과
- TTT val_bpb: ~1.1185-1.1200 (EWQ +0.007 + Pruning +0.002 합산, 일부 중복)
- artifact size: ~15.5-15.8MB ✅ (EWQ -2.0MB + Pruning -0.4MB)
- step_avg: ~95ms (변경 없음)

## 실제 결과
(실행 후 기록)
