"""Forked MindSpeed GMM with grad_weight_out support — ★ MoonEP-Ascend 自研版.

2026-09-23: 彻底移除 mindspeed 依赖 — npu_gmm /
npu_gmm_backward 全部由 MoonEP-Ascend ops/gmm 实现 (moonep_ops._gmm,
底层直调 CANN aclnnGroupedMatmulV4)。接口语义与原版兼容。

``npu_gmm_with_grad_weight_out``: dw 经 MoonEP npu_gmm_backward 计算并写入
调用方输出 tensor (home add_ + dup copy_), dx 保持激活梯度链 (X8 bridge)。
"""

import torch
import os

try:
    import moonep_ops
except ImportError:
    raise ImportError(
        "npu_gmm requires MoonEP-Ascend ops (moonep_ops); "
        "ensure MoonEP-Ascend is built and on PYTHONPATH"
    )

__all__ = ["npu_gmm", "npu_gmm_v2", "npu_gmm_with_grad_weight_out",
]


# ═══════════════════════════════════════════════════════════════════════
# ★ MoonEP-Ascend GMMFunction (2026-09-23 重写) — 直调 moonep_ops._gmm
# ═══════════════════════════════════════════════════════════════════════

class GMMFunction(torch.autograd.Function):
    """MoonEP 版 grouped GEMM autograd (mindspeed 依赖移除).

    forward: moonep_ops.npu_gmm (单 x/weight + group_list, 兼容原语义)。
    backward: moonep_ops.npu_gmm_backward → (dx, dw)。
    """

    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        del original_weight  # MoonEP 版无 original_weight 语义
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        del gemm_fusion  # MoonEP 版不支持 gemm_fusion (XTuner 恒 False)
        bias_t = bias if bias is not None else None
        if group_list_data_type == 0:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(x, weight, group_list)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type
        return moonep_ops.npu_gmm(x, weight, bias_t, group_list, group_type,
                                  group_list_type)

    @staticmethod
    def backward(ctx, grad_outputs):
        saved = ctx.saved_tensors
        x, weight = saved[0], saved[1]
        group_list = ctx.group_list
        dx, dw = moonep_ops.npu_gmm_backward(grad_outputs, x, weight,
                                             group_list, ctx.group_list_type)
        return None, dx, dw, None, None


# ═══════════════════════════════════════════════════════════════════════
# XTuner addition: public API with grad_weight_out (X8 堆直写)
# ═══════════════════════════════════════════════════════════════════════

class _GMMWithGradWeightOut(torch.autograd.Function):
    """GMM wrapper that writes dw into a caller-supplied tensor.

    Forward: moonep_ops.npu_gmm.
    Backward: dx + dw via moonep_ops.npu_gmm_backward, dw written into
    grad_weight_out as home add_ + dup copy_.

    ★ grad_weight_first (2026-10-15): 首个 producer 时 home 已由 dispatcher
    在 forward 阶段 zero — GEMM dw 直写整块 gwo (home+dup) 语义等价
    (0 + dw == add_; 覆写 == copy_), 省掉 dw 全量物化 + add_ + copy_
    (~8.4GB/层 附加访存 → 零)。非首个 producer 回退 add_/copy_ 路径
    (home 累加不可由覆写型 GEMM 表达)。
    """

    @staticmethod
    def forward(ctx, x, weight, group_list, group_list_type, grad_weight_out,
                grad_weight_first=False):
        outputs = moonep_ops.npu_gmm(x, weight, None, group_list, 0,
                                     group_list_type)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type
        ctx.grad_weight_out = grad_weight_out
        ctx.grad_weight_first = bool(grad_weight_first)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        group_list_type = ctx.group_list_type
        gwo = ctx.grad_weight_out

        if x.shape[0] == 0:
            return None, None, None, None, None, None

        if getattr(ctx, "grad_weight_first", False):
            # 首个 producer: home 已 zero (dispatcher forward 阶段) → GEMM
            # dw 直写整块 gwo (转置 view [E,I,O] over [E,O,I] 堆布局):
            # 零 dw 物化 / 零 add_ / 零 copy_。dw 返回值即 gwo view, 弃用。
            dx, _dw = moonep_ops.npu_gmm_backward(
                grad_output, x, weight, group_list, group_list_type,
                dw_out=gwo.transpose(1, 2))
            return dx, None, None, None, None, None

        # 回退路径 (非首 producer): home 累加 → dw 物化 + add_ + dup copy_。
        # home/dup 拆分: gwo = [epn+B, O, I], epn==B → b = shape[0]//2。
        b = gwo.shape[0] // 2

        # dx + dw 一次取: MoonEP npu_gmm_backward → (dx, dw), dw 形状 ==
        # weight.sizes() = [E,I,O] (npu_gmm 权重约定 [E,I,O],
        # group_gemm.py: weights.transpose(1,2) 喂入)。
        dx, dw = moonep_ops.npu_gmm_backward(grad_output, x, weight,
                                             group_list, group_list_type)
        # dw [E,I,O] → 堆槽 [E,O,I] 非连续 view (零物化)
        dw = dw.transpose(1, 2)

        # 写堆: home 跨 micro-batch 共享累加器 → add_; dup slot-local → copy_。
        # 不变量: dup copy_ 覆写语义依赖 planning _dedup_dst 的「每目标行
        # 唯一写者」— 多写者则须改 add_, 直写前提消失。
        gwo[:b].add_(dw[:b])
        gwo[b:].copy_(dw[b:])

        # dx 重启用: X8 接线后 _MoonEPExpertGradBridge.backward 由 dx 流回
        # expert-block 输入触发 → start_gradient_completion → reduce_grad_bf16。
        return dx, None, None, None, None, None


def npu_gmm_param_verification(x, weight, *, bias=None, group_list=None,
                               group_type=0, group_list_type=0):
    """参数校验 (与原版语义一致)。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}.")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight)}.")
    check_optional_tensor(bias, x.device, "bias")
    if group_list is not None and not isinstance(group_list, torch.Tensor):
        raise TypeError(f"group_list must be a torch.Tensor or None, got {type(group_list)}.")
    check_optional_tensor(group_list, x.device, "group_list")
    if group_list is not None and group_list.dtype != torch.int64:
        raise TypeError(f"group_list must be int64, got {group_list.dtype}!")


def check_optional_tensor(tensor, device, name):
    if tensor is not None and tensor.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but found at "
            f"least two devices, {device}(arg0) and {tensor.device}({name})!"
        )


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0,
            gemm_fusion=False, original_weight=None):
    """Grouped GEMM forward (MoonEP 自研, 兼容原 npu_gmm 语义)。"""
    del gemm_fusion  # MoonEP 版不支持 (XTuner 恒 False)
    npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list,
                               group_type=group_type, group_list_type=0)
    group_args = (group_list, group_type, False, 0,
                  1 if isinstance(group_list, (torch.Tensor, type(None))) else 0)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm_v2(x, weight, *, bias=None, group_list=None, group_type=0,
               gemm_fusion=False, original_weight=None):
    """group_list_type=1 变体 — MoonEP 版等价走 npu_gmm (group_list Tensor)。"""
    del gemm_fusion
    npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list,
                               group_type=group_type, group_list_type=1)
    group_args = (group_list, group_type, False, 1,
                  1 if isinstance(group_list, (torch.Tensor, type(None))) else 0)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm_with_grad_weight_out(x, weight, *, group_list,
                                 grad_weight_out=None, original_weight=None,
                                 grad_weight_first=False):
    """GMM with optional grad_weight_out (MoonEP-Ascend path).

    grad_weight_out=None → 等同 npu_gmm (backward 兼容)。
    提供时 backward 的 dw 直写 grad_weight_out (home add_ + dup copy_)。
    """
    if grad_weight_out is None:
        return npu_gmm(x, weight, bias=None, group_list=group_list,
                       group_type=0, group_list_type=0)
    npu_gmm_param_verification(x, weight, bias=None, group_list=group_list,
                               group_type=0, group_list_type=0)
    return _GMMWithGradWeightOut.apply(x, weight, group_list, 0,
                                       grad_weight_out, grad_weight_first)
