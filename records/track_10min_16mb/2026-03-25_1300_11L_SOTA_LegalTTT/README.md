# Exp 89: SOTA (Bigram8192) + Legal TTT

## Base
Exp 79 (Bigram8192), val_bpb 1.1213

## Changes
Added Legal score-first TTT from Exp 80 (LegalTTT) to the current SOTA model.
All existing SOTA code is preserved unchanged. Only TTT hyperparameters, the
`eval_val_sliding_ttt` function, and the TTT eval call at the end of `main()`
were added.

## Key TTT Parameters

| Parameter          | Value  | Description                          |
|--------------------|--------|--------------------------------------|
| ttt_enabled        | True   | Enable TTT evaluation                |
| ttt_lr             | 0.002  | SGD learning rate for TTT updates    |
| ttt_epochs         | 3      | Training epochs per chunk            |
| ttt_chunk_tokens   | 32768  | Tokens per TTT chunk                 |
| ttt_freeze_blocks  | 2      | Number of early blocks to freeze     |
| ttt_momentum       | 0.9    | SGD momentum                         |
| ttt_batch_seqs     | 32     | Batch size (sequences) for TTT train |
| ttt_grad_clip      | 1.0    | Gradient clipping norm               |

## Expected Result
~1.1180 val_bpb (TTT typically gives approximately -0.003 improvement over
the base sliding-window evaluation).
