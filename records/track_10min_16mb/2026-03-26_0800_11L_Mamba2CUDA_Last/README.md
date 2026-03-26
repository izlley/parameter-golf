# Exp 107: 11L CUDA-Accelerated Mamba2 (Layer 10)

## Purpose
Test whether CUDA-accelerated Mamba2 (mamba_ssm Triton kernels) can bring step_avg down to competitive levels (~95-100ms), making the Mamba2 hybrid architecture viable within the 10-minute training budget.

## Approach
- Based on Exp 106 (Mamba2 Last2) / Exp 90 (Mamba2 layer 10, 1.1208 bpb)
- Key changes:
  - **Pure PyTorch → mamba_ssm CUDA kernels**: `_segsum`/`_ssd_chunked` removed, `mamba_ssm.modules.mamba2.Mamba2` used directly. Internally uses Triton SSD kernel + `causal_conv1d` CUDA kernel.
  - **MAMBA_LAYERS**: "10" (single layer — best config from prior experiments)
  - **MAMBA_EXPAND**: 2 → 1 (d_inner halved: 1024→512, ~50% fewer Mamba params)
  - **MAMBA_CHUNK_SIZE**: 64 → 256 (fewer chunks = less inter-chunk overhead)
  - **VE_LAYERS**: "8,9" (standard, no collision with Mamba at 10)
  - **torch.compile**: fullgraph=True → fullgraph=False (CUDA op compatibility)
  - **LD_LIBRARY_PATH**: Auto-set for `/opt/conda/envs/cuda128/lib`
- Architecture: 11L (10 Attn + 1 Mamba2), 512dim, 8 heads (4 KV), GatedAttn PerHead

## Dependencies
- `mamba_ssm>=2.3.1` (Triton SSD kernels)
- `causal_conv1d` (fused causal conv1d CUDA kernel)
- `LD_LIBRARY_PATH` must include CUDA 12.x runtime

## Expected Results
- step_avg: ~95-100ms (vs ~120-150ms Pure PyTorch Mamba2)
- With more training steps (6000+ vs 4000-5000), expect better final bpb
- expand=1 reduces Mamba param count → better fit within 16MiB budget
- Target: < 1.1220 bpb with size < 16MiB
