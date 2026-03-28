# Exp 181: MuonTTT + N-gram Backoff (LRU Cache Limit)

## 목적 (Purpose)
Exp 170 N-gram Backoff의 메모리/속도 병목 해결 — Fix 3만 적용.
NgramCache에 max_tokens=500K 제한을 두어 dict 크기를 bounded하게 유지.
최근 토큰만 유지하는 LRU 방식으로 메모리 폭발 방지.

## 베이스
Exp 170 (11L MuonTTT NgramBackoff, 3가지 fix 모두 적용 버전에서 Fix 3만 분리)

## 변경 사항
- **LRU cache 제한**: max_tokens=500,000 (기본값)
  - 토큰 히스토리가 limit 초과 시 최근 500K만 유지
  - count dict를 전체 재빌드 (_rebuild_counts)
- **TTT에서 N-gram 사용**: 원본과 동일하게 TTT scoring에서 N-gram blending 적용
- **원본 Python list NgramCache**: numpy 미사용 (원본과 동일)

## 예상 결과
- TTT eval 속도: 초반 빠르나, rebuild 시 일시적 느려짐
- 메모리 사용: bounded (500K tokens × 5 orders × dict entries)
- TTT val_bpb: ~1.1160 (최근 데이터만 사용하므로 원본 대비 약간 다를 수 있음)
- artifact size: 18.1MB (base와 동일)

## 실제 결과
(실행 후 기록)
