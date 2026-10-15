import torch
import os

def npu_group_gemm(
    x: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    split_sizes: torch.Tensor,
    *,
    grad_weight_out: torch.Tensor | None = None,
) -> torch.Tensor:
    if os.environ.get("DISPATCHER", "all2all") == "moonep":
        return npu_group_gemm_moonep(x, weights, split_sizes, grad_weight_out=grad_weight_out)
    else:
        return npu_group_gemm_mindspeed(x, weights, split_sizes)

def npu_group_gemm_mindspeed(x: torch.Tensor, weights: torch.Tensor, split_sizes: torch.Tensor) -> torch.Tensor:
    from mindspeed.core.fusions.grouped_matmul import Ops
    weights = weights.transpose(1, 2)
    out = Ops.gmm(x, weights, split_sizes, trans_b=False)
    return out

def npu_group_gemm_moonep(
    x: torch.Tensor,
    weights: torch.Tensor | list[torch.Tensor],
    split_sizes: torch.Tensor,
    *,
    grad_weight_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped GEMM via MoonEP Ops.gmm (自研, 去 mindspeed) with grad_weight_out.

    Args:
        x: [num_tokens, in_features] bf16 input activations.
        weights: [num_experts, out_features, in_features] bf16 weights —
            MoonEP-Ascend [E+B, O, I] 堆视图 (home+dup);
            ★ ZEROCOPY: home+dup 窗口内物理连续 → full_view 单段视图
            (免 cat 物化 -2.4GB@GLM)。
        split_sizes: [num_experts] int tensor — tokens per expert group.
        grad_weight_out: optional [num_experts, out_features, in_features]
            bf16 tensor — 权重梯度直写堆槽 (X8)。
    """
    group_list = torch.cumsum(split_sizes, dim=0).to(x.device)
    # 单段 [E, I, O] (兼容非 ZEROCOPY 路径)
    weights_t = weights.transpose(1, 2)  # [E, I, O] — npu_gmm expects this
    if grad_weight_out is not None:
        # ★ P3B (2026-09-20): grad_weight_out = [E,O,I] 堆槽直传 (X8 backward
        #   grad_t@x → [E,O,I] 直写, 无 transpose)。
        from .gmm import npu_gmm_with_grad_weight_out
        return npu_gmm_with_grad_weight_out(
            x, weights_t, group_list=group_list,
            grad_weight_out=grad_weight_out, original_weight=None,
        )
    else:
        from .gmm import npu_gmm
        return npu_gmm(x, weights_t, bias=None, group_list=group_list,
                       group_type=0)
