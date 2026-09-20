"""Forked MindSpeed GMM with grad_weight_out support.

Copy of ``mindspeed/ops/gmm.py`` + a thin wrapper ``_GMMWithGradWeightOut``
that computes dw via the C++ ``npu_gmm`` forward op (group_type=2, same as
the C++ backward's internal implementation) and copies into a caller-
supplied output tensor.  This avoids the hook + extra autograd chain
overhead of the register_hook approach.  dx is computed via the official
``npu_gmm_backward`` so the activation-grad chain stays live — required
once X8 is wired into the dispatcher (triggers _MoonEPExpertGradBridge →
reduce_grad_bf16 + FSDP .grad writeback).

The original MindSpeed code (GMMFunction, npu_gmm, npu_gmm_v2) is preserved
unchanged for backward compatibility.  The new code path is only activated
when ``grad_weight_out`` is explicitly passed.

Original copyright: (c) 2025, Huawei Technologies Co., Ltd.
"""

import torch
from torch.library import impl

try:
    from mindspeed.op_builder import GMMOpBuilder, GMMV2OpBuilder
    from mindspeed.op_builder.builder import AS_LIBRARY
except ImportError:
    raise ImportError(
        "npu_gmm requires mindspeed; ensure torch_npu + mindspeed are installed"
    )

try:
    from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
except ImportError:
    npu_groupmatmul_add_fp32 = None


__all__ = ["npu_gmm", "npu_gmm_v2", "npu_gmm_with_grad_weight_out"]


def check_optional_tensor(tensor, device, name):
    if not isinstance(tensor, (torch.Tensor, type(None))):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor)}.")
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but found at least two devices, "
            f"{device}(arg0) and {tensor.device}({name})!")


# ============================================================================
# Original MindSpeed GMMFunction — unmodified (backward compatible path)
# ============================================================================

class GMMFunction(torch.autograd.Function):
    builder = GMMOpBuilder()
    builder2 = GMMV2OpBuilder()

    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        bias = [] if bias is None else [bias]
        if group_list_type == 0:
            outputs = GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
        elif group_list_type == 1:
            outputs = GMMFunction.builder2.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
        if group_list_data_type == 0:
            ctx.save_for_backward(x, weight, original_weight)
            ctx.group_list = group_list
        else:
            ctx.save_for_backward(x, weight, group_list, original_weight)
        ctx.gemm_fusion = gemm_fusion
        ctx.group_list_type = group_list_type
        ctx.group_list_data_type = group_list_data_type
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        if ctx.group_list_data_type == 0:
            x, weight, original_weight = ctx.saved_tensors
            group_list = ctx.group_list
        else:
            x, weight, group_list, original_weight = ctx.saved_tensors

        if ctx.gemm_fusion:
            if ctx.group_list_type == 0:
                dx, _, dbias = GMMFunction.builder.load().npu_gmm_backward_fusion([grad_outputs], [weight], group_list,
                                                                    ctx.group_list_type)
                if npu_groupmatmul_add_fp32 is not None:
                    npu_groupmatmul_add_fp32(x, grad_outputs, group_list, original_weight.main_grad)
            elif ctx.group_list_type == 1:
                dx, _, dbias = GMMFunction.builder2.load().npu_gmm_backward_fusion([grad_outputs], [weight], group_list,
                                                                    ctx.group_list_type)
                group_list_v2 = torch.cumsum(group_list, dim=0)
                if npu_groupmatmul_add_fp32 is not None:
                    npu_groupmatmul_add_fp32(x, grad_outputs, group_list_v2, original_weight.main_grad)

            dbias = None if len(dbias) == 0 else dbias[0]

            if hasattr(original_weight, 'grad_added_to_main_grad'):
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.shape,
                        dtype=x.dtype,
                        device=x.device,
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.shape,
                        dtype=x.dtype,
                        device=x.device,
                        requires_grad=False,
                    )
                original_weight.grad_added_to_main_grad = True
            else:
                grad_weight = None

            return None, dx[0], grad_weight, dbias, None
        else:
            if ctx.group_list_type == 0:
                dx, dw, dbias = GMMFunction.builder.load().npu_gmm_backward([grad_outputs], [x], [weight], group_list,
                                                                    ctx.group_list_type)
            elif ctx.group_list_type == 1:
                dx, dw, dbias = GMMFunction.builder2.load().npu_gmm_backward([grad_outputs], [x], [weight], group_list,
                                                                    ctx.group_list_type)
            dbias = None if len(dbias) == 0 else dbias[0]
            return None, dx[0], dw[0], dbias, None


# ============================================================================
# XTuner addition: thin wrapper for grad_weight_out (MoonEP-Ascend path)
# ============================================================================

class _GMMWithGradWeightOut(torch.autograd.Function):
    """GMM wrapper that writes dw into a caller-supplied tensor.

    Forward: calls C++ npu_gmm (same as MindSpeed).
    Backward: dw + dx via C++ npu_gmm_backward (returns (dx, dw, dbias)),
              dw reshaped to [E,O,I] (= weight.sizes, no layout scramble)
              and written into grad_weight_out as home add_ + dup copy_
              (home is a shared cross-micro-batch accumulator).  dx keeps
              the activation-grad chain live for _MoonEPExpertGradBridge.

    ★ P3B 修正 (2026-09-20): 旧实现走 forward op group_type=2 取 dw
    (grad^T@x), 但 aclnnGroupedMatmulV4 在 group_type=2 且 x 非分离时强制
    x 须转置 → 直传被拒 (RuntimeError); 且该路径产 [E,I,O] 经 reshape
    进堆 = o/i scramble。改用 npu_gmm_backward 一次取 dx+dw (生产路径,
    内部已 _foreach_transpose, dw 天然 [E,O,I])。
    """

    @staticmethod
    def forward(ctx, x, weight, group_list, group_list_type, grad_weight_out):
        bias = []
        if group_list_type == 0:
            outputs = GMMFunction.builder.load().npu_gmm(
                [x], [weight], bias, group_list, 0, group_list_type)
        else:
            outputs = GMMFunction.builder2.load().npu_gmm(
                [x], [weight], bias, group_list, 0, group_list_type)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type
        ctx.grad_weight_out = grad_weight_out
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        group_list_type = ctx.group_list_type
        gwo = ctx.grad_weight_out

        if x.shape[0] == 0:
            return None, None, None, None, None

        # home/dup 拆分: gwo = [epn+B, O, I], epn==B (home 与 dup 均为
        # experts_per_rank) → b = shape[0]//2 即 home 行数。
        b = gwo.shape[0] // 2

        # dw + dx 一次取: 官方 npu_gmm_backward 返回 (dx, dw, dbias), dw 形状
        # == weight.sizes()=[E,O,I] (生产路径, 无布局错位)。★不走 forward op
        # group_type=2 (aclnnGroupedMatmulV4 在 group_type=2 且 x 非分离时强制
        # x 须转置 → grad^T@x 直传 x 不转置会被拒; 且该路径产 [E,I,O] 经 reshape
        # 进堆 = o/i scramble — 旧 P3B 方案证伪)。npu_gmm_backward 内部已含正确
        # 转置 (_foreach_transpose), dw 天然 [E,O,I], reshape(gwo.shape) no-op。
        if group_list_type == 0:
            dx_list, dw_list, _ = GMMFunction.builder.load().npu_gmm_backward(
                [grad_output], [x], [weight], group_list, group_list_type)
        else:
            dx_list, dw_list, _ = GMMFunction.builder2.load().npu_gmm_backward(
                [grad_output], [x], [weight], group_list, group_list_type)
        dx = dx_list[0]
        # npu_gmm 权重约定 [E,I,O] (group_gemm.py: weights.transpose(1,2) 喂入),
        # 故 npu_gmm_backward 的 dw 形状 = weight.sizes() = [E,I,O]; 堆槽 gwo 为
        # [E,O,I] (home/dup = out,in) → transpose(1,2) 回堆布局。
        # ★显存优化 (2026-09-20): 旧 .contiguous().reshape() 物化 [E,O,I] 临时
        # tensor = 8.6GB/层 瞬时峰值 (proj0 5760+proj1 2880)。去 contiguous,
        # 直接用 transpose view 喂 add_/copy_ — 后者按 stride 读, bitwise 不变,
        # 省整个 [E,O,I] 临时物化 (npu_gmm_backward 的 [E,I,O] C++ 输出不可避免,
        # 但不再额外叠一份 [E,O,I])。注: add_/copy_ 对非连续 source 在 torch_npu
        # 经 GW-01 + XT-DISPATCH-X8 实测 bitwise 正确 (与 contiguous 路径一致)。
        dw = dw_list[0].transpose(1, 2)  # [E,O,I] 非连续 view (零物化)

        # 写堆: home 是跨 micro-batch 共享累加器 (见 return_expert_gradients 的
        # target[:b].add_ 语义), 必须 add_ 累加; dup 是 slot-local, copy_ 覆盖。
        # 旧 gwo.copy_(dw) 对 home 是覆盖非累加 → 多 micro-batch 丢累加 → 改。
        gwo[:b].add_(dw[:b])
        gwo[b:].copy_(dw[b:])

        # dx 重启用: X8 接线后 _MoonEPExpertGradBridge.backward 由 dx 流回
        # expert-block 输入触发 → start_gradient_completion → reduce_grad_bf16
        # (EP dup→home 精确和) + FSDP .grad 回写。若 dx=None 则 bridge 不触发,
        # 梯度完成链断裂 (reduce_grad 不跑、home 缺远端 dup、.grad 不回写)。
        # 返回: dx 给 x; weight/group_list/group_list_type/gwo 的 grad=None
        # (dw 已直写堆槽, 不走 autograd)。
        return dx, None, None, None, None


# ============================================================================
# Original MindSpeed public API — unmodified
# ============================================================================

def npu_gmm_param_verification(x, weight, *, bias=None, group_list=None, group_type=0, group_list_type=0):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"arg0 must be a torch.Tensor, got {type(x)}.")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"arg1 must be a torch.Tensor, got {type(weight)}.")
    if not isinstance(bias, (torch.Tensor, type(None))):
        raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias)}.")
    if (group_list_type == 0):
        if not (
            isinstance(group_list, (torch.Tensor, type(None)))
            or (isinstance(group_list, list) and all(isinstance(x, int) for x in group_list))
        ):
            raise TypeError(f"group_list must be a List of int64, torch.Tensor or None, got {type(group_list)}.")
    else:
        if not (isinstance(group_list, (torch.Tensor, type(None)))):
            raise TypeError(f"group_list must be a torch.Tensor or None, got {type(group_list)}.")
    if isinstance(group_list, torch.Tensor):
        if len(group_list.shape) > 1:
            raise ValueError(f"If group_list is not None, it must be an one-dimensional tensor, "
                             f"got dimension of group_list: {len(group_list.shape)}!")
        if group_list.dtype != torch.int64:
            raise TypeError(f"group_list must be a List of int64, got group_list type: {type(group_list)}, "
                            f"dtype: {group_list.dtype}!")
    if not isinstance(group_type, (int, type(None))):
        raise TypeError(f"group_type must be an int or None, got {type(group_type)}.")
    x_device = x.device
    device_warning = "Expected all tensors to be on the same device, but found at least two devices"
    if weight.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {weight.device}(arg1)!")
    if bias is not None and bias.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {bias.device}(bias)!")
    if isinstance(group_list, torch.Tensor) and group_list.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {group_list.device}(group_list)!")


def _npu_gmm_common(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, group_list_type=0, gemm_fusion=False):
    support_dtype = [torch.float16, torch.bfloat16, torch.float32]
    if weight.dtype not in support_dtype:
        raise TypeError(f"Only support non quant case, but got weight dtype {weight.dtype}.")
    npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list, group_type=group_type,
                               group_list_type=group_list_type)
    if group_list_type == 0:
        return torch.ops.mindspeed.npu_gmm(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
    elif group_list_type == 1:
        return torch.ops.mindspeed.npu_gmm_v2(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
    else:
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}.")


@impl(AS_LIBRARY, "npu_gmm.List", "PrivateUse1")
@impl(AS_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")
def _npu_gmm(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    if isinstance(group_list, (torch.Tensor, type(None))):
        group_list_data_type = 1
    else:
        group_list_data_type = 0
    group_args = (group_list, group_type, gemm_fusion, 0, group_list_data_type)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=0, gemm_fusion=gemm_fusion)


@impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "PrivateUse1")
def _npu_gmm_v2(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    group_args = (group_list, group_type, gemm_fusion, 1, 1)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm_v2(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=1, gemm_fusion=gemm_fusion)


# ============================================================================
# XTuner addition: public API with grad_weight_out
# ============================================================================

def npu_gmm_with_grad_weight_out(x, weight, *, group_list, grad_weight_out=None, original_weight=None):
    """GMM with optional grad_weight_out (MoonEP-Ascend path).

    When grad_weight_out is None, identical to npu_gmm (backward compatible).
    When provided, backward computes dw via C++ npu_gmm forward (group_type=2)
    and writes into grad_weight_out (home add_ + dup copy_) — one copy, no hook
    overhead.  dx is computed via npu_gmm_backward to keep the activation-grad
    chain live (required to trigger _MoonEPExpertGradBridge → reduce_grad_bf16
    + FSDP .grad writeback when X8 is wired into the dispatcher).
    """
    if grad_weight_out is None:
        return npu_gmm(x, weight, bias=None, group_list=group_list, group_type=0,
                       gemm_fusion=False, original_weight=original_weight)

    npu_gmm_param_verification(x, weight, bias=None, group_list=group_list,
                               group_type=0, group_list_type=0)
    group_list_type = 0
    return _GMMWithGradWeightOut.apply(x, weight, group_list, group_list_type, grad_weight_out)
