# Exp 170: MuonTTT + N-gram Backoff Eval

## 목적 (Purpose)
Eval-time N-gram backoff 적용으로 bpb 개선. Artifact 크기 비용 0.
이미 scored된 토큰들로 n-gram 확률을 구축하여 모델 출력과 블렌딩.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **N-gram Backoff**: eval_val_sliding_ttt 내에서 scored 토큰 기반 n-gram cache 유지
  - Multi-order backoff (2-gram ~ 7-gram)
  - `p_final = (1-alpha) * p_model + alpha * p_ngram`
  - alpha는 n-gram confidence에 따라 동적 조정
- **학습 변경 없음**: 모델 자체는 Exp 162와 동일
- **코드만 추가**: eval 함수에 n-gram 로직 추가 (code bytes 증가 ~2-3KB)

## 예상 결과
- TTT val_bpb: ~1.10-1.11 (bpb -0.01~0.07 개선, 문헌 보고 범위 넓음)
- artifact size: 18.1MB (변동 없음) — 크기 축소와 결합 필요
- eval time: 증가 예상 (n-gram lookup overhead)

## 실제 결과
(실행 후 기록)
