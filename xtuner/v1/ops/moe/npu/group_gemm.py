import torch
from .gmm import npu_gmm_with_grad_weight_out


def npu_group_gemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    split_sizes: torch.Tensor,
    *,
    grad_weight_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped GEMM via forked MindSpeed Ops.gmm with grad_weight_out support.

    Args:
        x: [num_tokens, in_features] bf16 input activations.
        weights: [num_experts, out_features, in_features] bf16 weights.
            For MoonEP-Ascend this is the [E+B, O, I] heap view (home+dup).
        split_sizes: [num_experts] int tensor — tokens per expert group.
        grad_weight_out: optional [num_experts, out_features, in_features]
            bf16 tensor. When provided, the weight gradient from backward
            is computed via C++ npu_gmm (group_type=2) and copied into this
            tensor (MoonEP-Ascend XT heap slot). One copy, no hook overhead,
            dx skipped (MoonEP combine handles activation grad).

    Returns:
        [num_tokens, out_features] bf16 output.

    Backward compatibility:
        ``grad_weight_out=None`` (default) is identical to the original
        implementation — MindSpeed's autograd handles dx and dw normally.
    """
    group_list = torch.cumsum(split_sizes, dim=0).to(x.device)
    weights_t = weights.transpose(1, 2)  # [E, I, O] — npu_gmm expects this

    if grad_weight_out is not None:
        # grad_weight_out shape: [E, O, I] — need [E, I, O] to match weights_t
        gwo_t = grad_weight_out.transpose(1, 2)
        return npu_gmm_with_grad_weight_out(
            x, weights_t, group_list=group_list,
            grad_weight_out=gwo_t, original_weight=None,
        )
    else:
        from .gmm import npu_gmm
        return npu_gmm(x, weights_t, bias=None, group_list=group_list,
                       group_type=0, gemm_fusion=False, original_weight=None)
