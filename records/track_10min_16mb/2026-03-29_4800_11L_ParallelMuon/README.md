# Experiment 188: 11L ParallelMuon (Parameter Banking)

## 목적
Reduce optimizer step time by restructuring per-parameter Newton-Schulz iterations into batched operations. Instead of calling `zeropower_via_newtonschulz5` sequentially for each 2D parameter, group parameters by shape into contiguous 3D banks and apply Newton-Schulz via `torch.bmm`. This is based on the approach from abaybektursun (PR#399).

## 베이스
Experiment 187 (4700_11L_LZMA_Compress)

## 변경 사항
- **New `ParallelMuon` optimizer class**: Groups all 2D matrix parameters by their (rows, cols) shape
- **Batched Newton-Schulz (`_batched_newton_schulz`)**: Processes all matrices in a shape group simultaneously using `torch.bmm` instead of sequential `@` matmuls
- **Shape-grouped processing**: Parameters with the same shape are stacked into (batch, rows, cols) tensors for batch processing
- **NorMuon normalization preserved**: `F.normalize` applied per-matrix after batched NS5
- **Replaced `Muon` with `ParallelMuon`** in optimizer setup (main function)
- No changes to model architecture, hyperparameters, or training loop

## 예상 결과
- **val_bpb**: Same as base (no quality change expected - mathematically equivalent optimizer)
- **step_avg**: Lower than base due to batched Newton-Schulz (fewer kernel launches, better GPU utilization)
- **artifact size**: Same as base (no model changes)

## 실제 결과
_(실험 실행 후 기록)_
