# Exp 180: MuonTTT + N-gram Backoff (Sliding Window Only)

## 목적 (Purpose)
Exp 170 N-gram Backoff의 속도 병목 해결 — Fix 2만 적용.
TTT eval에서 N-gram을 완전 비활성화하고, final sliding window eval에서만 적용.
TTT 속도를 base와 동일하게 유지하면서 N-gram 효과 확인.

## 베이스
Exp 170 (11L MuonTTT NgramBackoff, 3가지 fix 모두 적용 버전에서 Fix 2만 분리)

## 변경 사항
- **TTT에서 N-gram 비활성화**: eval_val_sliding_ttt에서 원래 cross_entropy 방식 복원
- **Sliding window eval에서 N-gram 적용**: eval_val_sliding에 ngram 파라미터 추가
  - 각 window scoring 후 ngram_cache 업데이트 (legal: already scored)
- **원본 Python list NgramCache**: numpy 미사용 (원본과 동일)
- **LRU 제한 없음**: cache 크기 무제한 (원본과 동일)

## 예상 결과
- TTT eval 속도: base와 동일 (~95ms/step)
- Sliding window eval: N-gram blending 적용 (Python list이므로 느릴 수 있음)
- TTT val_bpb: ~1.1160 (N-gram 없이 TTT만)
- final_sliding_window_bpb: N-gram 효과로 소폭 개선 가능
- artifact size: 18.1MB (base와 동일)

## 실제 결과
(실행 후 기록)
