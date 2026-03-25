# Exp 102: abaybektursun Base + GatedAttn + Bigram8192 + TTT

## Purpose
TTT가 abaybektursun 아키텍처에서 일관되게 성공(4/4, -0.002~0.003 bpb)하는 점을 활용.
우리의 best 기법(GatedAttn PerHead, Bigram8192)을 abaybektursun 베이스에 적용하고 TTT로 마무리.

## Base
abaybektursun's LeakyReLU + LegalTTT + ParallelMuon (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)

## Approach
- abaybektursun 베이스 (Parallel Muon, Parameter Banks, LeakyReLU²)
- GatedAttn PerHead 활성화 (gated_attention=1)
- BigramHash vocab 2048→8192 (해시 충돌 감소)
- Legal Score-First TTT 활성화 (ttt_enabled=1)
- 목표: SOTA 1.1194 bpb 갱신

## Key Changes
- `GATED_ATTENTION` default: "0" → "1"
- `BIGRAM_VOCAB_SIZE` default: 2048 → 8192
- `TTT_ENABLED` default: "0" → "1"

## Expected Results
- SW baseline: ~1.1200 (GatedAttn + Bigram8192 개선)
- TTT 적용 시: 1.1170~1.1190 (SOTA 갱신 가능)
- Size: abaybektursun 베이스 ~15.5MB + bigram 추가 → ~16.0MB
