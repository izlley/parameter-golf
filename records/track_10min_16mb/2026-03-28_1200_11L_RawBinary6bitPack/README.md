# Exp 152: NorMuon + Raw Binary 6-bit Packing

## 목적
torch.save의 pickle/zip 오버헤드 제거 + int6 값을 6-bit로 패킹하여 NorMuon 모델의 artifact 크기를 절감한다.
NorMuon은 uniform weight distribution으로 인해 zstd 압축 효율이 낮아 17.2MB가 되는 문제가 있다.
raw binary format과 6-bit packing으로 ~2MB+ 절감을 목표로 한다.

## 베이스
Exp 121 (11L_NorMuon) - val_bpb 1.1183, artifact 17.2MB

## 변경 사항
1. `import struct`, `import json` 추가
2. `pack_int6_to_bytes()` / `unpack_int6_from_bytes()`: int6 값을 6비트로 패킹 (4값 -> 3바이트)
3. `serialize_raw_binary()` / `deserialize_raw_binary()`: torch.save 대신 raw binary 직렬화
   - JSON 메타데이터 + 바이너리 텐서 데이터
   - int6 quantized 텐서(.q)는 6-bit packed format으로 저장 (dtype_code=3)
   - 나머지 텐서는 numpy raw bytes로 저장
4. main()에서 serialization/deserialization을 새 함수로 교체

## 예상 결과
- val_bpb: 1.1183 (동일, 모델 변경 없음)
- step_avg: 동일 (training 변경 없음)
- artifact size: ~14-15MB (torch.save 오버헤드 제거 + 6-bit packing으로 ~25% int6 텐서 크기 절감)

## 실제 결과
_(실험 실행 후 기록)_
