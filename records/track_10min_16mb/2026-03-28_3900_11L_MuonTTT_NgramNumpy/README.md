# Exp 179: MuonTTT + N-gram Backoff (Numpy Vectorized)

## 목적 (Purpose)
Exp 170 N-gram Backoff의 속도 병목 해결 — Fix 1만 적용.
NgramCache를 Python list → numpy array로 전환하여 연산 속도 개선.
TTT eval에서도 N-gram blending 사용 (원본과 동일).

## 베이스
Exp 170 (11L MuonTTT NgramBackoff, 3가지 fix 모두 적용 버전에서 Fix 1만 분리)

## 변경 사항
- **numpy 기반 NgramCache**: `[0] * V` → `np.zeros(V, dtype=int32)`
  - `sum(arr)` → `arr.sum()` (numpy vectorized)
  - `torch.tensor(arr)` per-position → `np.log()` + `torch.from_numpy()` 일괄 변환
  - unigram 업데이트: `np.add.at(self.unigram, ids_arr, 1)`
- **TTT에서 N-gram 사용**: 원본과 동일하게 TTT scoring에서 N-gram blending 적용
- **LRU 제한 없음**: cache 크기 무제한 (원본과 동일)

## 예상 결과
- TTT eval 속도: Exp 170 대비 개선 (numpy 연산), 하지만 dict 크기 증가는 여전
- TTT val_bpb: ~1.1160 (N-gram 효과 포함)
- artifact size: 18.1MB (base와 동일, N-gram은 eval-only)

## 실제 결과
(실행 후 기록)
