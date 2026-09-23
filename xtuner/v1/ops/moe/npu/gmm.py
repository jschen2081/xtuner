"""Forked MindSpeed GMM with grad_weight_out support — ★ MoonEP-Ascend 自研版.

2026-09-23: 彻底移除 mindspeed 依赖 — npu_gmm / npu_gmm_moep /
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
           "npu_gmm_moep"]


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
    """

    @staticmethod
    def forward(ctx, x, weight, group_list, group_list_type, grad_weight_out):
        outputs = moonep_ops.npu_gmm(x, weight, None, group_list, 0,
                                     group_list_type)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type
        ctx.grad_weight_out = grad_weight_out
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        group_list_type = ctx.group_list_type
        gwo = ctx.grad_weight_out

        if x.shape[0] == 0:
            return None, None, None, None, None

        # home/dup 拆分: gwo = [epn+B, O, I], epn==B → b = shape[0]//2。
        b = gwo.shape[0] // 2

        # dx + dw 一次取: MoonEP npu_gmm_backward → (dx, dw), dw 形状 ==
        # weight.sizes() = [E,I,O] (npu_gmm 权重约定 [E,I,O],
        # group_gemm.py: weights.transpose(1,2) 喂入)。
        dx, dw = moonep_ops.npu_gmm_backward(grad_output, x, weight,
                                             group_list, group_list_type)
        # dw [E,I,O] → 堆槽 [E,O,I] 非连续 view (零物化, 8.6GB/层 临时消除)
        dw = dw.transpose(1, 2)
        if int(os.environ.get("MOONEP_DBG_DWSHAPE", "0")):
            print(f"[X8-dbg] rank={torch.distributed.get_rank()} "
                  f"dw.shape={tuple(dw.shape)} gwo.shape={tuple(gwo.shape)} "
                  f"b={b} x.shape={tuple(x.shape)}", flush=True)
            print(f"[X8-grad] rank={torch.distributed.get_rank()} "
                  f"dw.norm={dw.float().norm().item():.6f} "
                  f"dw[:b].norm={dw[:b].float().norm().item():.6f} "
                  f"dw[b:].norm={dw[b:].float().norm().item():.6f}", flush=True)

        # 写堆: home 跨 micro-batch 共享累加器 → add_; dup slot-local → copy_。
        gwo[:b].add_(dw[:b])
        gwo[b:].copy_(dw[b:])

        # dx 重启用: X8 接线后 _MoonEPExpertGradBridge.backward 由 dx 流回
        # expert-block 输入触发 → start_gradient_completion → reduce_grad_bf16。
        return dx, None, None, None, None


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


def npu_gmm_moep(x, weights, *, group_list=None, out=None, bias=None,
                 group_type=0, gemm_fusion=False, grad_weight_out=None):
    """MoonEP ZEROCOPY grouped GEMM (TensorList weight + out 直写堆).

    Args:
        x: TensorList, 每专家一组输入 [M_i, K] bf16 (aclnn TensorList 配对,
           与 weights 等长; dispatch 输出已按专家行排序, 调用方按组切)。
        weights: TensorList, 每专家独立权重 [K, N] bf16 — [home_view,
           dup_view] 或逐专家, 免 torch.cat 物化。
        out: 可选预分配 MoonEP 堆输出视图 [ΣM, N] — ACLNN 直写 (ZEROCOPY)。
        grad_weight_out: 可选 [E,O,I] 堆槽 — backward 的 dw 直写 (X8)。
    """
    if not isinstance(x, (list, tuple)):
        x = [x]
    if not isinstance(weights, (list, tuple)):
        raise TypeError("npu_gmm_moep weights must be a list/tuple of tensors")
    if len(x) != len(weights):
        raise ValueError(f"npu_gmm_moep x({len(x)}) and weights({len(weights)}) "
                         f"must be paired")
    if bias is not None:
        raise ValueError("npu_gmm_moep bias not supported (MoonEP 无 bias)")
    if grad_weight_out is not None:
        return _GMMMoepGradOut.apply(*x, *weights, group_list, out,
                                     grad_weight_out, len(x))
    return _GMMMoepFunction.apply(*x, *weights, group_list, out,
                                  len(x))


class _GMMMoepFunction(torch.autograd.Function):
    """npu_gmm_moep autograd (无 gwo): forward 直写 + backward dx。

    ★ apply 参数须为 Tensor (list 参数不被 autograd 追踪 → 展平传入)。
    """

    @staticmethod
    def forward(ctx, *args):
        x_len = int(args[-1])
        out = args[-2]
        group_list = args[-3]
        x_list = list(args[:x_len])
        weights = list(args[x_len:-3])
        outputs = moonep_ops.npu_gmm_moep(x_list, weights, None, group_list, 0, 0,
                                          out if out is not None else None)
        ctx.has_gl = group_list is not None
        if group_list is not None:
            ctx.save_for_backward(*x_list, *weights, group_list)
        else:
            ctx.save_for_backward(*x_list, *weights)
        ctx.x_len = x_len
        ctx.weights_len = len(weights)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x_list = saved[:ctx.x_len]
        weights = saved[ctx.x_len:ctx.x_len + ctx.weights_len]
        group_list = saved[-1] if ctx.has_gl else None
        x = torch.cat(x_list, 0)
        w = torch.stack(weights, 0)      # [E,K,N]
        dx, _ = moonep_ops.npu_gmm_backward(grad_output, x, w, group_list, 0)
        dx_segs = dx.split([xi.size(0) for xi in x_list], 0)   # 按段切回
        return (*dx_segs, *[None] * (ctx.weights_len + 3))     # x 段 + w 段 + gl + out + len


class _GMMMoepGradOut(torch.autograd.Function):
    """npu_gmm_moep + X8: backward dw 直写 grad_weight_out (home add_ + dup copy_)."""

    @staticmethod
    def forward(ctx, *args):
        x_len = int(args[-1])
        grad_weight_out = args[-2]
        out = args[-3]
        group_list = args[-4]
        x_list = list(args[:x_len])
        weights = list(args[x_len:-4])
        outputs = moonep_ops.npu_gmm_moep(x_list, weights, None, group_list, 0, 0,
                                          out if out is not None else None)
        ctx.has_gl = group_list is not None
        if group_list is not None:
            ctx.save_for_backward(*x_list, *weights, group_list)
        else:
            ctx.save_for_backward(*x_list, *weights)
        ctx.x_len = x_len
        ctx.weights_len = len(weights)
        ctx.grad_weight_out = grad_weight_out
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x_list = saved[:ctx.x_len]
        weights = saved[ctx.x_len:ctx.x_len + ctx.weights_len]
        group_list = saved[-1] if ctx.has_gl else None
        gwo = ctx.grad_weight_out

        x = torch.cat(x_list, 0)
        w = torch.stack(weights, 0)      # [E,K,N]
        dx, dw = moonep_ops.npu_gmm_backward(grad_output, x, w, group_list, 0)
        dw = dw.transpose(1, 2)          # [E,O,I]
        b = gwo.shape[0] // 2
        gwo[:b].add_(dw[:b])
        gwo[b:].copy_(dw[b:])
        dx_segs = dx.split([xi.size(0) for xi in x_list], 0)   # 按段切回
        return (*dx_segs, *[None] * (ctx.weights_len + 4))     # x 段 + w 段 + gl + out + gwo + len


def npu_gmm_with_grad_weight_out(x, weight, *, group_list,
                                 grad_weight_out=None, original_weight=None):
    """GMM with optional grad_weight_out (MoonEP-Ascend path).

    grad_weight_out=None → 等同 npu_gmm (backward 兼容)。
    提供时 backward 的 dw 直写 grad_weight_out (home add_ + dup copy_)。
    """
    if grad_weight_out is None:
        return npu_gmm(x, weight, bias=None, group_list=group_list,
                       group_type=0, gemm_fusion=False,
                       original_weight=original_weight)
    npu_gmm_param_verification(x, weight, bias=None, group_list=group_list,
                               group_type=0, group_list_type=0)
    return _GMMWithGradWeightOut.apply(x, weight, group_list, 0,
                                       grad_weight_out)
