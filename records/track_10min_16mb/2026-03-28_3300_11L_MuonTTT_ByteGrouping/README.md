# Exp 173: MuonTTT + Byte-Grouping Compression

## 목적 (Purpose)
int6 양자화 후 zstd 압축 전에 byte-grouping을 적용하여 무손실 크기 절감.
High/low nibble을 분리하여 더 homogeneous한 스트림 생성 → zstd 엔트로피 코딩 효율 향상.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **Byte-Grouping**: quantized int8 tensor를 byte 단위로 재배열
  - 방법 A: high nibble / low nibble 분리 (각각 독립 zstd 압축)
  - 방법 B: int6 4개 → 3 bytes 비트 패킹 (padding bit 제거)
  - 방법 C: delta encoding (인접 weight 차이 저장) + zstd
- **Decompression 코드**: artifact 내 코드에 역변환 로직 포함
- **학습 변경 없음**: 순수 serialization 최적화

## 예상 결과
- TTT val_bpb: 1.1160 (변동 없음 — 무손실)
- artifact size: ~17.0-17.5MB (0.5-1.0MB 절감)
- 주의: Exp 152(RawBinary6bitPack)에서 raw packing이 오히려 크기 증가했던 전례 있음

## 실제 결과
(실행 후 기록)
