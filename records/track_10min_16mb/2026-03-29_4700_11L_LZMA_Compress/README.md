# Exp 187: NorMuon + LZMA Compression

## 목적
NorMuon의 uniform weight 분포에서 zstd보다 더 나은 압축률을 가진 LZMA 압축으로 교체하여 artifact 크기를 16MB 이하로 줄인다.
abaybektursun이 LZMA를 사용하여 ~15.95MB 달성한 것을 참고.

## 베이스
Exp 162 (11L_TTT_MuonOptimizer / NorMuon + MuonLiteTTT, 1.1160 bpb, 18.1MB)

## 변경 사항
1. `import lzma` 추가
2. `_COMPRESSOR = "lzma"` 로 변경 (zstd 대신)
3. 압축: `lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)` 사용
4. 해제: `lzma.decompress(data)` 사용
5. 학습/모델은 변경 없음 — 직렬화 압축만 변경

## 배경
- zstd는 LZ77 기반으로 반복 패턴에 강하지만, NorMuon의 uniform 분포에서는 효과 저하
- LZMA는 더 큰 사전(dictionary)과 range coding으로 entropy에 가까운 압축 가능
- abaybektursun이 LZMA preset=9로 ~15.95MB 달성

## 예상 결과
- val_bpb: 1.1160 (동일, 압축만 변경)
- step_avg: 동일
- artifact size: ~16-17MB (zstd 18.1MB에서 개선 기대)

## 실제 결과
_(실행 후 기록)_
