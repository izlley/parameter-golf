# Exp 154: NorMuon + Raw Binary 6-bit Packing + SupermaskTTT V3

## 목적
NorMuon+TTT의 best bpb(1.1181)를 유지하면서 raw binary + 6-bit packing 직렬화로 18.2MB -> 16MB 이하 달성

## 베이스
Exp 145 (11L_SupermaskTTT_V4 / NorMuon + SupermaskTTT V3, 1.1181 bpb, 18.2MB)

## 변경 사항
- `torch.save` -> raw binary serialization (struct-based, no pickle overhead)
- int6 quantized weights: 6-bit packing (4 values = 3 bytes, 25% savings vs int8)
- `import struct`, `import json` 추가
- `pack_int6_to_bytes` / `unpack_int6_from_bytes`: int6 값을 6비트로 팩킹/언팩킹
- `serialize_raw_binary` / `deserialize_raw_binary`: torch.save 대신 raw binary format 사용
- zstd 압축은 기존과 동일하게 유지

## 예상 결과
- val_bpb: 1.1181 (동일, 직렬화만 변경이므로 학습/추론에 영향 없음)
- step_avg: 동일
- artifact size: ~14-15MB (18.2MB에서 6-bit packing + raw binary overhead 제거로 감소)

## 실제 결과
_(실행 후 기록)_
