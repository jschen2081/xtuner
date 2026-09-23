import torch
import os
from .gmm import npu_gmm_with_grad_weight_out, npu_gmm_moep


def npu_group_gemm(
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
            ★ ZEROCOPY: 也接受 [home_view, dup_view] 两段列表 (免 cat 物化
            -2.4GB@GLM) → 走 npu_gmm_moep TensorList 逐专家配对。
        split_sizes: [num_experts] int tensor — tokens per expert group.
        grad_weight_out: optional [num_experts, out_features, in_features]
            bf16 tensor — 权重梯度直写堆槽 (X8)。
    """
    group_list = torch.cumsum(split_sizes, dim=0).to(x.device)
    if os.environ.get("MOONEP_DBG_DWSHAPE", "0") == "1":
        print(f"[CNT-dbg] rank={torch.distributed.get_rank()} "
              f"split_sizes.shape={tuple(split_sizes.shape)} "
              f"split_sizes.sum={int(split_sizes.sum()) if hasattr(split_sizes,'sum') else 'NA'} "
              f"split_sizes[:8]={list(split_sizes[:8].tolist()) if hasattr(split_sizes,'tolist') else 'NA'} "
              f"split_sizes[8:16]={list(split_sizes[8:16].tolist()) if hasattr(split_sizes,'tolist') else 'NA'} "
              f"split_sizes[16:24]={list(split_sizes[16:24].tolist()) if hasattr(split_sizes,'tolist') else 'NA'} "
              f"split_sizes[-4:]={list(split_sizes[-4:].tolist()) if hasattr(split_sizes,'tolist') else 'NA'}", flush=True)
    if isinstance(weights, (list, tuple)):
        # ★ ZEROCOPY (2026-09-23): [home_view, dup_view] → 逐专家配对
        #   (home epn 行 + dup B 行 = 本 rank 专家数; x 按 split_sizes 切段)
        w_list = [row.transpose(-1, -2) for seg in weights for row in seg.unbind(0)]  # [O,I]→[I,O] view
        x_list = x.split([int(s) for s in split_sizes.tolist()])
        assert len(w_list) == len(x_list), \
            f"moep pairing mismatch: {len(w_list)} weights vs {len(x_list)} x-segments"
        # 直写堆: grad_weight_out 形状 [E,O,I] 与输出 [ΣM, N] 无关 —
        # 输出 out 用普通分配 (输出 ZEROCOPY 直写 combine 段后续优化)
        return npu_gmm_moep(x_list, w_list, group_list=group_list,
                            out=None, grad_weight_out=grad_weight_out)
    # 单段 [E, I, O] (兼容非 ZEROCOPY 路径)
    weights_t = weights.transpose(1, 2)  # [E, I, O] — npu_gmm expects this
    if grad_weight_out is not None:
        # ★ P3B (2026-09-20): grad_weight_out = [E,O,I] 堆槽直传 (X8 backward
        #   grad_t@x → [E,O,I] 直写, 无 transpose)。
        return npu_gmm_with_grad_weight_out(
            x, weights_t, group_list=group_list,
            grad_weight_out=grad_weight_out, original_weight=None,
        )
    else:
        from .gmm import npu_gmm
        return npu_gmm(x, weights_t, bias=None, group_list=group_list,
                       group_type=0, group_list_type=0)
