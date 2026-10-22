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

    ★ grad_weight_first (2026-10-15): 首个 producer 的 GEMM dw 直写覆写整块
    gwo (home+dup, 空组行写 0) — 覆写 == add_ (home 初值) 且 == copy_ (dup),
    省掉 dw 全量物化 + add_ + copy_ (~8.4GB/层 附加访存 → 零), 同时免掉
    dispatcher 的 home 预清零 (2026-10-22)。
    非首个 producer 在 MOONEP_X8_DW=1 (默认) 下回退 add_/copy_ 路径
    (home 累加不可由覆写型 GEMM 表达);
    MOONEP_X8_DW=2 (P0) 下非首 producer 走 dup 半区直写 + home 物化累加
    (dup 直写 == copy_, producer 局部等价)。
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

        # MOONEP_X8_DW: 0 = 全物化回退; 1 = 首 producer 整块直写 (P1, 默认);
        #               2 = P0 — 首 producer 整块直写 + 非首 producer
        #               dup 半区直写 + home 半区物化累加。
        dw_mode = os.environ.get("MOONEP_X8_DW", "1")
        # B1 (2026-10-22): dw 直写改走「交换 A/B + 连续 out」, 消除
        # aclnnGroupedMatmulV4 为转置 view out 插入的内部 Transpose 物化。
        # 置 0 回退旧写法 (转置 view out), 用于同库 A/B 对拍。
        dw_swap = os.environ.get("MOONEP_GMM_DW_SWAP", "1") != "0"

        if getattr(ctx, "grad_weight_first", False) and dw_mode != "0":
            # 首个 producer: GEMM dw 直写整块 gwo, 覆写含空组行的全部 2B 行。
            # dw 按 (x^T @ grad)^T == grad^T @ x 反向组织, 直接产出堆槽布局
            # [E,O,I] 的连续 out — 省掉「写转置 view → CANN 内部 Transpose
            # 物化」那一步 (实测 40×(0.57+0.26) ms/步, 两条路径逐比特一致)。
            # 零 dw 物化 / 零 add_ / 零 copy_。
            if dw_swap:
                wt = weight.transpose(-1, -2)
                dx = moonep_ops.npu_gmm(grad_output, wt, None, group_list, 0,
                                        group_list_type)
                moonep_ops.npu_gmm_moep([grad_output.transpose(-1, -2)], [x],
                                        None, group_list, 2, group_list_type,
                                        gwo)
            else:  # A/B 对拍用: 改动前原样 (转置 view out, CANN 内部插 Transpose)
                dx, _dw = moonep_ops.npu_gmm_backward(
                    grad_output, x, weight, group_list, group_list_type,
                    dw_out=gwo.transpose(1, 2))
            return dx, None, None, None, None, None

        if dw_mode == "2":
            # ★ P0 (非首 producer): dup 半区 GEMM 直写 + home 半区物化累加。
            #   home 跨 micro-batch 共享累加 → GEMM 物化 dw_home + add_;
            #   dup slot-local, 每行都被本 producer 的 copy_ 写满 →
            #   GEMM 直写 == copy_ (含空组行写 0, 无残留风险), producer 局部
            #   等价, 不依赖跨 producer 不变量。
            #   切分: x 的组序与堆行序一致 (先 home 组后 dup 组),
            #   boundary = group_list[b-1] (home 组 token 总数)。
            #   代价: 一次 .item() 同步取 boundary (见 env 注释, G>1 才走来)。
            b = gwo.shape[0] // 2
            boundary = int(group_list[b - 1].item())
            # dx: 与 npu_gmm_backward 同式 (grad @ W^T, group_type=0 全 M 一组)
            wt = weight.transpose(-1, -2)
            dx = moonep_ops.npu_gmm(grad_output, wt, None, group_list, 0,
                                    group_list_type)
            # home dw: 物化 + add_ (跨 micro-batch 累加器, 不可覆写)
            # ★ B1: 同首 producer, 交换 A/B 让 dw_home 直接是 [b,O,I] 连续,
            #   add_ 走连续读写 (免 stride 读 + 免内部 Transpose 物化)。
            if boundary > 0:
                if dw_swap:
                    dw_home = moonep_ops.npu_gmm(
                        grad_output[:boundary].transpose(-1, -2), x[:boundary],
                        None, group_list[:b], 2, group_list_type)
                else:  # A/B 对拍用: 旧写法, dw_home 是 [b,I,O] 再转置 add_
                    dw_home = moonep_ops.npu_gmm(
                        x[:boundary].transpose(-1, -2), grad_output[:boundary],
                        None, group_list[:b], 2, group_list_type).transpose(1, 2)
                gwo[:b].add_(dw_home)
            # dup dw: GEMM 直写 dup 半区 (空组行写 0 — 自洁, 等价 copy_)
            if x.shape[0] - boundary > 0:
                gl_dup = (group_list[b:] - boundary).contiguous()
                if dw_swap:
                    moonep_ops.npu_gmm_moep(
                        [grad_output[boundary:].transpose(-1, -2)],
                        [x[boundary:]], None, gl_dup, 2, group_list_type,
                        gwo[b:])
                else:  # A/B 对拍用: 旧写法, 转置 view out
                    moonep_ops.npu_gmm_moep(
                        [x[boundary:].transpose(-1, -2)],
                        [grad_output[boundary:]], None, gl_dup, 2,
                        group_list_type, gwo[b:].transpose(1, 2))
            else:
                gwo[b:].zero_()
            return dx, None, None, None, None, None

        # 回退路径 (dw_mode == "0", 或 "1" 下非首 producer):
        # home 累加 → dw 物化 + add_ + dup copy_。
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
