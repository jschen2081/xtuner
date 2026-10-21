"""NPU equivalent of route_weight_rows_backward.

Replaces the Triton kernel with pure PyTorch ops. The computation is
element-wise row scaling — no custom kernel needed:

  grad_expert[row] = (grad_weighted[row] * route_weight[row]).to(bf16)
  grad_route[row]  = sum_over_hidden(grad_weighted[row] * expert_output[row])

Both are computed in FP32 end to end (only the grad_expert output is rounded
to bf16). This matches the all2all baseline, whose merge backward keeps FP32
for both `w` and the route-weight reduction (dP agrees with a bare FP32
reference to ~1e-7). The earlier bf16 intermediate rounding, copied from the
Triton kernel, cost 2.8e-3 on dX and 1.7e-3 on dP, and its hardware
round-half-away ties disagree with torch's RNE on ~1/256 of the products.

★ Memory-lean chunked implementation: peak transient memory is one chunk's
rows (~MBs) instead of N full-width fp32 copies — the unchunked version
allocated ~5 [N, H] fp32 temporaries and OOM'd backward at EP=16/PACK=4096
(606MB over budget).
"""

import torch
from torch import Tensor

# Chunk rows so each transient is ~16MB (fp32 H=6144: 16384 rows/chunk upper
# bound tuned for [NvS_total, 6144] shapes; smaller H scales up safely).
_TARGET_CHUNK_BYTES = 16 * 1024 * 1024


def route_weight_rows_backward(
    grad_weighted: Tensor,
    expert_output: Tensor,
    route_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Differentiate fused BF16 row scaling without a full FP32 activation.

    Args:
        grad_weighted: [num_tokens, H] bf16, contiguous.
        expert_output: [num_tokens, H] bf16, contiguous.
        route_weights: [num_tokens] fp32, contiguous.

    Returns:
        grad_expert: [num_tokens, H] bf16 — grad_weighted * route_weight (rounded to bf16).
        grad_route: [num_tokens] fp32 — sum(grad_weighted * expert_output) per row,
            accumulated in fp32.
    """
    assert grad_weighted.dtype is torch.bfloat16 and grad_weighted.is_contiguous()
    assert expert_output.dtype is torch.bfloat16 and expert_output.is_contiguous()
    assert route_weights.dtype is torch.float32 and route_weights.is_contiguous()
    assert grad_weighted.shape == expert_output.shape
    assert route_weights.shape == grad_weighted.shape[:1]

    num_tokens, H = grad_weighted.shape

    rw = route_weights  # fp32, unquantized

    grad_expert = torch.empty_like(grad_weighted)
    grad_route = torch.empty(num_tokens, dtype=torch.float32, device=grad_weighted.device)

    # ★ 分块: 每块中间量只驻留一份 [chunk, H] fp32 (~16MB), 块间释放
    row_bytes = H * 4  # fp32 per row (largest transient)
    chunk = max(1, _TARGET_CHUNK_BYTES // row_bytes)

    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        gw = grad_weighted[start:end]                       # bf16 view
        eo = expert_output[start:end]                       # bf16 view

        # grad_expert chunk = (gw * rw).to(bf16), per-row broadcast
        grad_expert[start:end] = (
            gw.to(torch.float32) * rw[start:end].unsqueeze(-1)
        ).to(torch.bfloat16)

        # grad_route chunk = sum over hidden of (gw*eo), fp32 throughout
        product = gw.to(torch.float32) * eo.to(torch.float32)
        grad_route[start:end] = product.sum(dim=-1)

    return grad_expert, grad_route


__all__ = ["route_weight_rows_backward"]
