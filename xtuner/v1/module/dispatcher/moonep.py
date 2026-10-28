"""MoonEP's model-scoped XTuner integration.

The backend import remains lazy so unrelated dispatchers do not require
MoonEP. ``MoonEPModelRuntime`` owns model resources and ``MoonEPDispatcher``
owns one routed layer's static policy. One dispatch/combine call is a pure
data ``_MoonEPLayerCallState`` record advanced by the module-level transaction
functions (``dispatch_forward``, ``prepare_experts``, ``combine_forward``,
``combine_backward``, gradient completion, and the layer-Join handoff). The
private VMM workspace remains the deep module for physical expert layout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import TypedDict, override

# ★ Ascend: route_weight_rows_backward NPU branch (pure PyTorch, no Triton).
# GPU path uses a Triton kernel; NPU uses an equivalent torch implementation.
# ★★ 必须按设备分流而非 import 成败: cuda.route_weight 的 import 永远成功,
#     Triton kernel 只在首次调用时加载 libcuda — NPU 上 import 能过但调用即崩。
try:
    import torch.npu  # noqa: F401  (torch_npu 注入 npu 后端后此属性存在)

    _ON_NPU = True
except (ImportError, AttributeError):
    _ON_NPU = False

if _ON_NPU:
    # ★ P3A: 接线 MoonEP AscendC route_weight 算子 (moonep_ops.route_weight,
    #   commit cf69813, 单遍 fused AIV, 5.20x 带宽). 签名与 torch 分块实现一致;
    #   回退到 xtuner 纯 PyTorch 分块实现 (OOM-safe chunked) 当 moonep_ops 未构建.
    try:
        from moonep_ascend.ops.route_weight import route_weight_rows_backward
    except (ImportError, AttributeError):
        from xtuner.v1.ops.moe.npu.route_weight import route_weight_rows_backward
else:
    from xtuner.v1.ops.moe.cuda.route_weight import route_weight_rows_backward
from xtuner.v1.utils import log_rank0

from .base import ExpertWeightLayout, GenericDispatcher, PostDispatchResult, ProjectionPair
from .fsdp_vmm_landing import (
    accumulate_fsdp_unsharded_expert_gradients,
    fsdp_current_unsharded_expert_parameters,
    install_fsdp_vmm_landing,
    uninstall_fsdp_vmm_landing,
)
from .moonep_workspace import _ExpertVMMWorkspace
from .moonep_ascend_workspace import _ExpertAscendWorkspace


_INTEGRATION_API_VERSION = 3
_MOONEP_IMPORT_ERROR: ImportError | None

try:
    import moonep_ascend as _moonep_backend
except ImportError as exc:
    _moonep_backend = None
    _MOONEP_IMPORT_ERROR = exc
else:
    _MOONEP_IMPORT_ERROR = None


def require_moonep_backend() -> Any:
    """Validate the optional MoonEP-mod package when MoonEP is selected."""
    if _moonep_backend is None:
        raise RuntimeError("dispatcher='moonep' requires the MoonEP-mod integration package") from _MOONEP_IMPORT_ERROR

    source = getattr(_moonep_backend, "__file__", "<unknown>")
    if getattr(_moonep_backend, "XTUNER_INTEGRATION_API_VERSION", None) != _INTEGRATION_API_VERSION:
        raise RuntimeError(
            f"incompatible MoonEP integration API; expected {_INTEGRATION_API_VERSION}; loaded module: {source}"
        )
    return _moonep_backend


@dataclass(frozen=True)
class _MoonEPLayer:
    """One physical routed layer's identity, stored once.

    ``ordinal`` is the FSDP execution-order position; the home generation is
    issued from it by the workspace, so it is not stored here.
    """

    fqn: str
    projections: tuple[nn.Module, nn.Module]
    ordinal: int


@dataclass(frozen=True)
class _MoonEPResources:
    """The only non-``None`` state after ``install_after_fsdp``.

    The call state borrows this record instead of the runtime; ``buffer_for``
    and ``enqueue`` are its two methods. ``_buffer_box`` is a one-slot mutable
    box holding ``(Buffer, Fixed-S)`` because the activation Buffer is still
    built lazily on the first forward once Fixed-S is known.
    """

    workspace: _ExpertVMMWorkspace
    landing: ExpertLandingAdapter
    comm_stream: torch.cuda.Stream
    ep_group: dist.ProcessGroup
    num_experts: int
    experts_per_rank: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    gradient_slots: int
    num_sms: int
    ip_port: str
    _buffer_box: list
    # ★ 2026-09-23: Ascend DirectVMM 延迟 install 载体 (buffer_for 的 self 是
    #   resources, frozen dataclass 不能动态赋 → 声明字段, 构造时传入)。
    _deferred_landing_install: Any = None

    def buffer_for(self, tokens_per_rank: int) -> Any:
        """Dispatch entry: build the Fixed-S Buffer once, then check S."""
        if not self._buffer_box:
            assert _moonep_backend is not None
            # ★ 每个 EP 组用独立 SHMEM bootstrap 端口。组 ID = 本组 root 的全局
            # rank (dist.get_global_rank(ep_group, 0)) — 组内所有 rank 算出同一值,
            # 不同组不同值。不能用 dist.get_rank(ep_group) (那是组内 rank 0..R-1,
            # 同组不同 rank 会算出不同端口 → client 连错 server)。
            # DeviceMesh EP 维跨节点交错 → 多组 root 同在 master, 端口必须错开。
            group_root_global_rank = dist.get_global_rank(self.ep_group, 0)
            base_port = int(self.ip_port.rsplit(':', 1)[-1])
            host = self.ip_port.rsplit(':', 1)[0]
            shmem_port = base_port + group_root_global_rank
            shmem_ip_port = f"{host}:{shmem_port}"
            buffer = _moonep_backend.BufferXtuner(
                S=tokens_per_rank,
                H=self.hidden_size,
                K=self.top_k,
                E=self.num_experts,
                num_ep_ranks=self.ep_group.size(),
                group=self.ep_group,
                explicitly_destroy=True,
                num_sms=self.num_sms,
                ip_port=shmem_ip_port,
                ffn_hidden_size=self.intermediate_size,
                gradient_slots=self.gradient_slots,
            )
            # ★ Ascend: fill the deferred workspace layout from BufferXtuner's
            # symmetric heap views (xt_*_view). GPU path builds layout in
            # install_after_fsdp via VMM; Ascend defers to here (post-buffer).
            if hasattr(self.workspace, 'build_from_buffer'):
                self.workspace.build_from_buffer(buffer, self.gradient_slots)
                # ★ S3 修: DirectVMM 的 install 延迟至此 (layout 刚建好,
                #   landings 视图可用)。仅一次。
                if getattr(self, "_deferred_landing_install", None):
                    fsdp_root, layers = self._deferred_landing_install
                    # ★ self 是 _MoonEPResources — landing 字段无下划线 (runtime
                    #   的 _landing 私有属性只存在于 runtime 上, 这里用 self.landing)
                    self.landing.install(fsdp_root=fsdp_root,
                                         workspace=self.workspace,
                                         layers=layers)
                    self._deferred_landing_install = None
            self._buffer_box.append((buffer, tokens_per_rank))
        buffer, fixed_s = self._buffer_box[0]
        if tokens_per_rank != fixed_s:
            raise RuntimeError(f"MoonEP fixed S changed: {fixed_s} -> {tokens_per_rank}")
        return buffer

    @property
    def buffer(self) -> Any:
        """Combine/backward entry: the Buffer this call's dispatch built."""
        if not self._buffer_box:
            raise RuntimeError("MoonEP Buffer must be created by dispatch first")
        return self._buffer_box[0][0]

    def expect_tokens_per_rank(self, tokens_per_rank: int) -> None:
        """Reject a changed Fixed-S before the first Dispatcher/VMM op."""
        if self._buffer_box and tokens_per_rank != self._buffer_box[0][1]:
            raise RuntimeError(f"MoonEP fixed S changed: {self._buffer_box[0][1]} -> {tokens_per_rank}")

    def enqueue(
        self,
        operation: Callable[[], Any],
        *,
        inputs: tuple[torch.Tensor | None, ...] = (),
    ) -> tuple[Any, torch.cuda.Event]:
        """Run one MoonEP transaction on XTuner's stream and return ``(result,
        done event)``."""
        caller_stream = torch.cuda.current_stream()
        self.comm_stream.wait_event(caller_stream.record_event())
        for tensor in inputs:
            if tensor is not None:
                tensor.record_stream(self.comm_stream)
        with torch.cuda.stream(self.comm_stream):
            result = operation()
            done = self.comm_stream.record_event()
        if self.comm_stream is not caller_stream:
            # 输出张量在 comm_stream 上分配, 之后由 caller 流消费。不登记
            #   caller 流的话, allocator 在 comm_stream 完成时就认为可复用,
            #   caller 仍在使用 → UAF。(单独 comm 流已移除, 此分支留作防御)
            _record_tensor_streams(result, caller_stream)
        return result, done

    def home_generation(self, layer: _MoonEPLayer) -> int:
        """The workspace owns the physical chunks and issues the generation."""
        return self.workspace.generation_for(layer.ordinal)


class ExpertLandingAdapter:
    """Make one generation's home weights ready and hand back its Parameters.

    ``DirectVMMLanding`` is the production path (FSDP unpacks straight into VMM
    home rows); ``StagingReferenceLanding`` is the numerical-reference path
    (FSDP lands normal storage, then copies into the VMM home rows). Both are
    the two Adapters of one Seam, so the runtime and transaction functions
    only call ``prepare`` and never branch on a ``staging_reference`` flag.
    """

    def install(
        self, *, fsdp_root: nn.Module, workspace: _ExpertVMMWorkspace, layers: tuple[_MoonEPLayer, ...]
    ) -> None:
        raise NotImplementedError

    def prepare(self, *, layer: _MoonEPLayer, generation: int) -> tuple[nn.Parameter, nn.Parameter]:
        raise NotImplementedError

    def uninstall(self) -> None:
        raise NotImplementedError


class DirectVMMLanding(ExpertLandingAdapter):
    """FSDP's final per-parameter unpack lands directly in the VMM home
    rows."""

    def __init__(self) -> None:
        self._fsdp_params: tuple[Any, ...] = ()

    @override
    def install(
        self, *, fsdp_root: nn.Module, workspace: _ExpertVMMWorkspace, layers: tuple[_MoonEPLayer, ...]
    ) -> None:
        self._fsdp_params = install_fsdp_vmm_landing(
            fsdp_root=fsdp_root,
            targets=tuple(
                (layer.fqn, layer.projections, workspace.landing(workspace.generation_for(layer.ordinal)))
                for layer in layers
            ),
        )

    @override
    def prepare(self, *, layer: _MoonEPLayer, generation: int) -> tuple[nn.Parameter, nn.Parameter]:
        # "Ready" is one check: FSDP has already materialized the weight in
        # the VMM landing, so there is no copy.
        del generation
        return fsdp_current_unsharded_expert_parameters(layer.projections)

    @override
    def uninstall(self) -> None:
        if self._fsdp_params:
            uninstall_fsdp_vmm_landing(self._fsdp_params)
            self._fsdp_params = ()


class StagingReferenceLanding(ExpertLandingAdapter):
    """FSDP lands normal storage first, then this copies it into the VMM home
    rows."""

    def __init__(self) -> None:
        log_rank0.warning(
            "moonep_staging_reference=True copies complete BF16 home expert "
            "weights after every FSDP AllGather; it is a numerical reference, "
            "not the production performance path."
        )
        self._workspace: _ExpertVMMWorkspace | None = None

    @override
    def install(
        self, *, fsdp_root: nn.Module, workspace: _ExpertVMMWorkspace, layers: tuple[_MoonEPLayer, ...]
    ) -> None:
        # No FSDP binding is installed: that is what distinguishes the two
        # Adapters. The copy happens in ``prepare``.
        del fsdp_root, layers
        self._workspace = workspace

    @override
    def prepare(self, *, layer: _MoonEPLayer, generation: int) -> tuple[nn.Parameter, nn.Parameter]:
        assert self._workspace is not None
        parameters: list[nn.Parameter] = []
        for linear, landing in zip(layer.projections, self._workspace.landing(generation), strict=True):
            weight = cast(torch.Tensor, linear.weight)
            if not isinstance(weight, nn.Parameter):
                raise RuntimeError(f"{layer.fqn} staging expected an unsharded expert Parameter")
            source = weight.to_local() if isinstance(weight, DTensor) else weight
            if source.dtype is not torch.bfloat16 or source.numel() != landing.numel():
                raise RuntimeError(f"{layer.fqn} staging expected an unsharded BF16 expert weight")
            with torch.no_grad():
                landing.copy_(source.view_as(landing))
            parameters.append(weight)
        return parameters[0], parameters[1]

    @override
    def uninstall(self) -> None:
        self._workspace = None


def build_landing_adapter(staging_reference: bool) -> ExpertLandingAdapter:
    if staging_reference:
        return StagingReferenceLanding()
    return DirectVMMLanding()


class MoonEPModelRuntime:
    """Own the lifecycle and ordered layer registry for one model/EP group.

    Construction takes no CUDA resources. ``build_dispatcher`` registers one
    physical routed layer per call in construction order; ``install_after_fsdp``
    allocates the workspace, cross-checks registration order against FSDP
    execution order, and installs the landing Adapter.
    """

    def __init__(
        self,
        *,
        ep_group: dist.ProcessGroup,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        intra_layer_micro_batch: int,
        staging_reference: bool,
        num_sms: int = 64,
        ip_port: str = "tcp://127.0.0.1:8766",
        tokens_per_rank: int | None = None,
        home_generations: int = 2,
    ) -> None:
        # Config-level capability validation (backend version, EP geometry,
        # dtype, grouped-GEMM backend, ...) lives in ``moonep_capability`` and
        # runs at meta model build. Keep only the optional-backend gate here so
        # direct construction still fails fast.
        require_moonep_backend()

        self._ep_group = ep_group
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._num_experts = num_experts
        self._top_k = top_k
        self._num_sms = num_sms
        self._gradient_slots = intra_layer_micro_batch
        self._ip_port = ip_port
        # ★ 2026-09-23 (DirectVMM S 前置): trainer 算好的 Fixed-S (已除 SP)。
        #   install_after_fsdp 用它预建 BufferXtuner → DirectVMM install 早于
        #   FSDP 首次 all_gather (Ascend 路径 landings 来自 buffer home 视图)。
        self._tokens_per_rank = tokens_per_rank
        # ★ 2026-09-23 (Home 单代): workspace allocate 的 home_generations
        #   (MOONEP_HOME_GENERATIONS env, 默认 2)。单代省一半 home 显存。
        self._home_generations = home_generations
        self._landing = build_landing_adapter(staging_reference)

        # Physical routed layers in registration (construction) order.
        self._layers: list[_MoonEPLayer] = []
        self._resources: _MoonEPResources | None = None
        self._closed = False

    def bind_layer(
        self,
        *,
        layer_fqn: str,
        projections: tuple[nn.Module, nn.Module],
    ) -> MoonEPDispatcher:
        """Register one physical routed layer and return its Dispatcher."""
        if any(layer.fqn == layer_fqn for layer in self._layers):
            raise ValueError(f"duplicate MoonEP routed layer: {layer_fqn}")
        layer = _MoonEPLayer(fqn=layer_fqn, projections=projections, ordinal=len(self._layers))
        self._layers.append(layer)
        return MoonEPDispatcher(runtime=self, layer=layer)

    def validate_before_fsdp(self, fsdp_config: Any) -> None:
        # The build-time FSDP policy checks moved to ``moonep_capability``.
        # This boundary stays because the Protocol needs it and a future
        # backend may have its own FSDP preconditions.
        del fsdp_config

    def install_after_fsdp(self, *, fsdp_root: nn.Module, execution_order: list[str]) -> None:
        """Allocate execution resources after native FSDP has been
        installed."""
        if self._resources is not None:
            raise RuntimeError("MoonEP FSDP resources are already installed")
        if not self._layers:
            raise TypeError("MoonEP requires at least one physical routed-expert layer")

        # Registration order vs FSDP execution order, checked once in the only
        # place that can see both. ``moe.py`` hands over the ordered list
        # rather than an adapter reading FSDP private structure.
        registered = [layer.fqn for layer in self._layers]
        if registered != execution_order:
            raise RuntimeError(
                f"MoonEP registration order does not match FSDP execution order: {registered} != {execution_order}"
            )

        # ★ DIAG: trace install_after_fsdp step by step
        def _diag(msg):
            pass  # (排查期诊断打印已移除, 保留签名以最小化 diff)

        workspace = _ExpertAscendWorkspace.allocate(
            projection_shapes=(
                (2 * self._intermediate_size, self._hidden_size),
                (self._hidden_size, self._intermediate_size),
            ),
            num_experts=self._num_experts,
            ep_group=self._ep_group,
            gradient_slots=self._gradient_slots,
            home_generations=self._home_generations,
            buffer=None,  # ★ Ascend: workspace built lazily in buffer_for
        )

        # ★ 2026-10-28 独立 comm 流已移除。Keep MoonEP collectives in FSDP's
        #   device-side launch order: a separate high-priority stream forms an
        #   orthogonal progress wave with HCCL and stalls at MoonEP's rank
        #   barriers, and on Ascend it also deadlocks outright —
        #   ``_combine_xt`` reads ``plan.resolve_nsegs()`` back to the host, and
        #   that D2H waits on the dedicated stream while the compute stream is
        #   still waiting for the same combine (measured: ACL stream synchronize
        #   failed 507015, then SIGABRT on every rank). ``enqueue`` therefore
        #   degrades to a no-op stream switch, which is what all the overlap
        #   work below is built on top of.
        comm_stream = torch.cuda.current_stream()

        try:
            if hasattr(workspace, "_layout") and workspace._layout is None:
                # ★ Ascend deferred layout: workspace 的 layout 依赖
                #   BufferXtuner (landings 来源)。DirectVMM install 必须早于
                #   FSDP 首次 all_gather → tokens_per_rank 已知时在 resources
                #   创建后预建 buffer (layout 填充) + 立即 install; 未知时
                #   (旧路径) 延迟到 buffer_for (首次 dispatch, 有 S 校验兜底)。
                if self._tokens_per_rank is not None:
                    self._prebuild_required = True
                else:
                    self._deferred_landing_install = (fsdp_root, tuple(self._layers))
            else:
                self._landing.install(fsdp_root=fsdp_root, workspace=workspace,
                                      layers=tuple(self._layers))
        except Exception:
            workspace.destroy()
            raise
        self._resources = _MoonEPResources(
            workspace=workspace,
            landing=self._landing,
            comm_stream=comm_stream,
            ep_group=self._ep_group,
            num_experts=self._num_experts,
            experts_per_rank=self._num_experts // self._ep_group.size(),
            top_k=self._top_k,
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
            gradient_slots=self._gradient_slots,
            num_sms=self._num_sms,
            ip_port=self._ip_port,
            _buffer_box=[],
            _deferred_landing_install=getattr(self, "_deferred_landing_install", None),
        )
        self._deferred_landing_install = None
        # ★ 2026-09-23 (DirectVMM S 前置): tokens_per_rank 已知 → 立即预建
        #   buffer (build_from_buffer 填 workspace layout) + DirectVMM install。
        #   此处早于任何 FSDP all_gather (模型刚 fully_shard, 训练未开始)。
        if getattr(self, "_prebuild_required", False):
            self._prebuild_required = False
            self._resources.buffer_for(self._tokens_per_rank)
            self._resources.landing.install(
                fsdp_root=fsdp_root, workspace=workspace,
                layers=tuple(self._layers))
            log_rank0.info(
                f"[MoonEP] DirectVMM installed early (tokens_per_rank={self._tokens_per_rank})")

    @property
    def resources(self) -> _MoonEPResources:
        """One place decides "is MoonEP installed"."""
        if self._closed:
            raise RuntimeError("MoonEP runtime was closed")
        if self._resources is None:
            raise RuntimeError("MoonEP FSDP resources must be installed before forward")
        return self._resources

    def close(self) -> None:
        """Release Buffer before VMM workspace at a coordinated boundary."""
        if self._closed:
            return
        if self._resources is not None:
            resources = self._resources
            resources.comm_stream.synchronize()
            for buffer, _ in resources._buffer_box:
                buffer.destroy()
            resources.landing.uninstall()
            resources.workspace.destroy()
            self._resources = None
        self._layers.clear()
        self._closed = True


# Dispatcher shape legend:
#   S: source tokens on this EP rank, K: routed experts per token,
#   NvS: MoonEP's padded VM-group rows, E: global experts, B=E/R: home
#   experts per EP rank, H: hidden size.


class MoonEPPreDispatchResult(TypedDict):
    """Stage 1: device-resident router-space inputs normalized for MoonEP."""

    hidden_states: torch.Tensor  # [S, H], BF16 source-token order.
    topk_ids: torch.Tensor  # [S, K], contiguous int32 global expert IDs.
    tokens_per_expert: torch.Tensor  # [E], contiguous int32 source histogram.
    # The call state is opaque control state for the remaining five stages;
    # it never crosses into the compiled tensor-only expert block.
    _moonep_call: _MoonEPLayerCallState


class MoonEPDispatchResult(TypedDict):
    """Stage 2: global dispatch outputs.

    The call state travels on ``MoonEPPreDispatchResult`` only; every later
    stage already receives ``pre_dispatched`` and reads ``_moonep_call`` there.
    """

    hidden_states: torch.Tensor  # [NvS, H], BF16 physical VM-group order.
    topk_weights: torch.Tensor  # [NvS], FP32 weights in the same row order.
    # [E+B], int32 padded group ends; stays on device and is non-differentiable.
    cu_seqlens: torch.Tensor


class MoonEPPostDispatchResult(PostDispatchResult):
    """Stage 3: tensor-only local ``[2B]`` expert-compute bundle.

    ``hidden_states`` is ``[NvS, H]``; ``tokens_per_expert`` is device int32
    ``[2B]`` for home then duplicate groups; ``expert_weight_layout`` holds the
    projection-paired ``[2B, O_p, I_p]`` call-local weight aliases.
    """


class MoonEPPreCombineResult(TypedDict):
    """Stage 4: expert outputs before route scaling and reverse transport."""

    hidden_states: torch.Tensor  # [NvS, H], physical VM-group order.


class MoonEPCombineResult(TypedDict):
    """Stage 5: fused route-scaled output restored to source-token order."""

    hidden_states: torch.Tensor  # [S, H].


class MoonEPPostCombineResult(TypedDict):
    """Stage 6: final tensor bundle returned through the generic interface."""

    hidden_states: torch.Tensor  # [S, H].


def _async_parallel_enabled() -> bool:
    """Whether MoonEP communication is submitted asynchronously and overlapped.

    ``MOONEP_ASYNC_PARALLEL=1`` is the single switch for the communication /
    compute overlap that in-layer micro-batching depends on. It turns on three
    cooperating effects, none of which pays off alone (measured on Qwen3-30B
    EP8 ILMB=2: either half alone leaves the step at ~14.3 s, both together
    reach 13.1 s):

    * the routed dispatch is launched without the host-side
      ``aclrtSynchronizeStream``. That sync costs ~11 ms of host time per
      micro-batch (the kernel spins on the remote MAGIC flag for most of it),
      and while the host sits inside it the next micro-batch's prepare/GEMM
      cannot be submitted;
    * the weight prefetch runs on its own stream instead of the compute stream,
      so the transfer proceeds while dispatch and GEMM are being launched;
    * the prefetch waits for this call's own GEMM (see
      :func:`run_deferred_prefetch`). The expert home section holds one set of B
      borrow slots per generation shared by every in-layer micro-batch, so a
      later micro-batch's prefetch would otherwise overwrite the experts an
      earlier micro-batch's GEMM is still reading — a deterministic
      wrong-weight read (measured: borrow slot 2 holds expert 105 for
      micro-batch 0, expert 98 for micro-batch 1), not a tolerance-level
      difference.

    Read per call rather than cached so a single-process A/B does not need a
    restart; the environment is consulted once per routed layer.
    """
    return os.environ.get("MOONEP_ASYNC_PARALLEL", "0") == "1"


def _record_tensor_streams(result: Any, stream: Any) -> None:
    """Walk an enqueue result and pin every tensor to the consumer stream.

    The transaction result is a tensor, a tuple of tensors, or a tuple holding
    a ``(tensors..., event)`` pair. Registering the consumer stream keeps the
    caching allocator from recycling a comm-stream allocation while the caller
    stream still reads it.
    """
    if isinstance(result, torch.Tensor):
        result.record_stream(stream)
    elif isinstance(result, (tuple, list)):
        for item in result:
            _record_tensor_streams(item, stream)


@dataclass
class _MoonEPLayerGradients:
    """One FSDP call's initialization state, never shared across replay calls.

    Storage belongs to the workspace. Only the first backward producer clears it; forward/checkpoint replay must not
    touch a different call's live H.
    """

    initialized: bool = False
    # ★★ B6 (2026-10-27) ILMB: "first producer" 必须按 **梯度槽** 判, 不能按层判。
    #   intra_layer_micro_batch > 1 时每个微批占一个独立槽 (B4), 每层只写同一槽
    #   一次 → 该槽在层的首次使用就是它的 first producer。按层判会让第 2 个微批
    #   的槽既不被覆写也不被预清零, GEMM 的 add_ 累积到上一步/上一层的残留上
    #   (实测专家梯度 ≈ 48 层累加、norm ×51)。G=1 时与旧的按层判等价。
    gmm_written: set[int] = field(default_factory=set)
    reduced: set[int] = field(default_factory=set)


@dataclass(eq=False)
class _MoonEPLayerCallState:
    """One routed-layer call's pure lifecycle state for the transaction
    functions.

    The record borrows the installed ``_MoonEPResources`` and the ``_MoonEPLayer``
    identity, plus the call-local plan/event/weight/gradient handles that the
    module-level transaction functions read and advance. It owns no behavior
    and never references ``MoonEPDispatcher``. Identity, not field equality,
    distinguishes two calls, so instances stay hashable by ``id``.
    """

    resources: _MoonEPResources
    layer: _MoonEPLayer
    generation: int
    grad_slot: int
    layer_gradients: _MoonEPLayerGradients

    # One MoonEP communication plan and its device-side dependency chain. Each
    # event is recorded once its named producer has been enqueued.
    plan: Any | None = None
    dispatch_done: Any | None = None
    dispatch_event: Any | None = None
    weights_ready: Any | None = None
    combine_done: Any | None = None

    # Borrowed local [2B, O_p, I_p] weight aliases for this call.
    local_weights: ProjectionPair | None = None

    # Deferred weight prefetch (``MOONEP_ASYNC_PARALLEL``): the home borrow
    # slots hold one set of B rows per generation, and each micro-batch's
    # prefetch fills them with the experts its own routing chose. Running the
    # prefetch inside ``dispatch`` therefore lets micro-batch i+1 rewrite the
    # slots micro-batch i's GEMM is still reading. With the prefetch deferred
    # it is issued right before this call's own GEMM instead, so it is ordered
    # behind the previous micro-batch's GEMM.
    prefetch_buffer: Any | None = None
    prefetch_deferred: bool = False

    # Current FSDP unsharded home Parameters [B, O_p, I_p] receive the returned
    # BF16 home gradients after both local projections complete.
    home_parameters: tuple[nn.Parameter, nn.Parameter] | None = None
    # Completed home views and the event covering the pair reduction.
    gradient_completion: tuple[ProjectionPair, Any] | None = None

    # X8 (grad_weight_out direct-to-heap dW) active for this call. When True,
    # ``prepare_experts`` injects per-slot heap gradient views as
    # ``grad_weight_out`` into the expert GroupedLinear forward; dW is written
    # directly into the heap (home add_ + dup copy_) and the bridge receives
    # ``None`` dW (weights passed as non-leaf). The bridge still fires via the
    # re-enabled dx to trigger ``reduce_grad_bf16`` + FSDP ``.grad`` writeback.
    x8_active: bool = False


# --- Transaction functions --------------------------------------------------
#
# Each function covers one complete device-side sequence for a single call and
# advances ``_MoonEPLayerCallState`` in place. The autograd Functions and the
# Dispatcher are the only callers; the pair backward must reuse the forward
# plan and gradient slot recorded on the call state.


def finish_combine(state: _MoonEPLayerCallState, combined: torch.Tensor, *, async_op: bool) -> torch.Tensor:
    """Establish the final device dependency for an async combine."""
    if async_op:
        assert state.combine_done is not None
        state.combine_done.wait()
    return combined


def run_deferred_prefetch(state: _MoonEPLayerCallState) -> None:
    """Fill this call's home borrow slots, right before its own GEMM consumes them.

    ``prepare_experts`` is the first weight consumer of a call, so issuing the
    transfer here orders it behind the previous micro-batch's GEMM. That
    ordering is what makes in-layer interleave correct: the borrow slots are
    shared by every micro-batch of the layer, so a prefetch that ran inside
    ``dispatch`` would let micro-batch i+1 overwrite the experts micro-batch i
    is still reading.

    The transfer goes to the dedicated prefetch stream rather than the compute
    stream. ``start_prefetch`` folds the launching stream's current point into
    its completion event, which is exactly the "previous GEMM has retired" edge
    this ordering needs, so the transfer runs alongside the following launches
    instead of blocking the host.
    """
    if not state.prefetch_deferred:
        return
    buffer = state.prefetch_buffer
    plan = state.plan
    state.prefetch_deferred = False
    state.prefetch_buffer = None
    assert buffer is not None and plan is not None
    workspace = state.resources.workspace
    generation = state.generation
    with torch.profiler.record_function("MoonEP::deferred_prefetch"):
        done = workspace.start_prefetch(
            buffer=buffer,
            plan=plan,
            generation=generation,
            stream=buffer.prefetch_stream,
        )
        state.local_weights = workspace.local_weights(generation=generation)
        # This call's GEMM is the transfer's only consumer.
        torch.cuda.current_stream().wait_event(done)


def prepare_experts(state: _MoonEPLayerCallState, dispatched: MoonEPDispatchResult) -> MoonEPPostDispatchResult:
    """Wait at the first weight consumer and expose the tensor-only layout."""
    resources = state.resources
    assert state.weights_ready is not None

    run_deferred_prefetch(state)

    with torch.profiler.record_function("MoonEP::prepare_experts"):
        # This inserts a device dependency; it never waits on the host.
        state.weights_ready.wait()
        # ``dispatch`` completes on its own comm stream; join it here, at the
        # first consumer of the routed tokens, rather than back in
        # ``dispatch_forward`` where the wait would also gate the following
        # micro-batch's dispatch and destroy the overlap.
        if state.dispatch_event is not None:
            state.dispatch_event.wait()
        # Both halves of the ``[E+B] -> [2B]`` contract now live in the
        # workspace: the caller cannot receive counts it must still fix up.
        hidden_states, local_counts = resources.workspace.local_compute_view(
            hidden_nvsh=dispatched["hidden_states"],
            cu_seqlens=dispatched["cu_seqlens"],
        )

    local_weights = state.local_weights
    state.local_weights = None
    assert local_weights is not None
    # Join activation and both weight edges so staging precedes upstream
    # activation backward; a weight-only hook cannot establish that dependency.
    # The bridge makes the aliases require grad (via hidden_states), so grouped
    # GEMM returns their dW without a leaf ``nn.Parameter`` wrapper.
    hidden_states, w13, w2 = _MoonEPExpertGradBridge.apply(hidden_states, local_weights[0], local_weights[1], state)

    grad_weight_out = None
    if state.x8_active:
        # X8: inject per-slot heap gradient views so _GMMWithGradWeightOut writes
        # dW directly into the heap (home add_ + dup copy_). The bridge still
        # fires via the re-enabled dx; its backward receives None dW (X8 owns it)
        # and triggers reduce_grad_bf16 + FSDP .grad writeback.
        workspace = state.resources.workspace
        grad_slot = state.grad_slot
        # The home prefix is a single accumulator shared across every micro-
        # batch of one FSDP call (same storage for every slot).
        # ★ 2026-10-22: first producer 的 GEMM dw 直写会覆写整块堆的每一行
        #   (空组行写 0 — ops/tests 探针验证 garbage-heap == zero-heap), 因此
        #   home 无需预清零: producer 1 覆写其余 add_ 累加即为正确和。仅
        #   MOONEP_X8_DW=0 (全物化回退, home 走 add_) 仍需预清零。
        first_producer = grad_slot not in state.layer_gradients.gmm_written
        if first_producer:
            if os.environ.get("MOONEP_X8_DW", "1") == "0":
                for proj in (0, 1):
                    workspace.home_grad_view(grad_slot, proj).zero_()
            state.layer_gradients.gmm_written.add(grad_slot)
            state.layer_gradients.initialized = True
        grad_weight_out = (
            workspace.grad_view(grad_slot, 0),
            workspace.grad_view(grad_slot, 1),
        )

    return MoonEPPostDispatchResult(
        hidden_states=hidden_states,
        tokens_per_expert=local_counts,
        expert_weight_layout=ExpertWeightLayout(
            trainable_weights=(w13, w2),
            grad_weight_out=grad_weight_out,
            grad_weight_first=first_producer if state.x8_active else False,
        ),
    )


def dispatch_forward(
    state: _MoonEPLayerCallState,
    source_hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    source_route_weights: torch.Tensor,
    *,
    async_op: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch on a fresh plan and start both projection weight prefetches."""
    resources = state.resources
    buffer = resources.buffer_for(source_hidden.shape[0])

    def dispatch_and_prefetch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Landing.prepare makes this generation's home weights ready: a check
        # for the direct Adapter, the staging copy for the reference Adapter.
        # Its call position (before dispatch's device barrier) is part of this
        # sequence, not an implicit precondition of a mode branch.
        state.home_parameters = resources.landing.prepare(layer=state.layer, generation=state.generation)

        def start_prefetch(plan: Any) -> Any:
            """Launch the weight prefetch on the dedicated stream.

            Called by ``dispatch`` once the plan is final but before its kernel
            is launched, which is the only window where the transfer can run
            concurrently with dispatch. It reads ``plan.experts_to_copy`` and
            the home weights — both ready before that kernel — and writes the
            home window, which dispatch never touches.
            """
            return resources.workspace.start_prefetch(
                buffer=buffer,
                plan=plan,
                generation=state.generation,
                stream=buffer.prefetch_stream,
            )

        async_parallel = _async_parallel_enabled()
        # In-layer micro-batching needs the prefetch to follow this call's own
        # GEMM; a single micro-batch has no neighbour to clash with.
        defer_prefetch = async_parallel and resources.gradient_slots > 1
        dispatched = buffer.dispatch(
            source_hidden,
            route_weights_sk=source_route_weights,
            topk_experts_sk=topk_ids,
            tokens_per_expert=tokens_per_expert,
            async_finish=async_parallel,
            zero_copy=False,
            # ``None`` keeps the serial path: dispatch, then prefetch.
            prefetch_hook=start_prefetch if (async_parallel and not defer_prefetch) else None,
            # ★★ B6 (2026-10-27) ILMB: 层内多微批各占一个梯度槽, 同槽号隔离
            #   对称堆的 hidden/weights/status 份 (Ascend 专用, 其他实现忽略)。
            mb_idx=state.grad_slot,
        )
        if async_parallel:
            hidden_nvsh, route_weights_nvs, cu_seqlens, plan, dispatch_event = dispatched
        else:
            hidden_nvsh, route_weights_nvs, cu_seqlens, plan = dispatched
            dispatch_event = None
        assert route_weights_nvs is not None and cu_seqlens is not None
        state.plan = plan
        if dispatch_event is not None:
            # ``dispatch`` ran on the buffer's own comm stream. Do **not** fold
            # its completion into this stream here: a ``wait_event`` gates every
            # later launch on the current stream, so the compute of micro-batch
            # i would queue behind micro-batch i+1's dispatch as well and the
            # two could never overlap. ``prepare_experts`` joins this event at
            # the first consumer instead, which is the only place that needs
            # the dispatched tokens to have landed.
            state.dispatch_event = dispatch_event
            state.dispatch_done = dispatch_event
        else:
            state.dispatch_event = None
            state.dispatch_done = torch.cuda.current_stream().record_event()
        # Only local weights are returned now; the gradient slot views are
        # created inside ``return_expert_gradients`` when they are needed.
        if defer_prefetch:
            # Leave the home borrow slots untouched until this call's GEMM is
            # about to run: a later micro-batch's dispatch would otherwise
            # refill them ahead of the earlier micro-batch's read.
            state.prefetch_buffer = buffer
            state.prefetch_deferred = True
            state.local_weights = None
        elif async_parallel:
            # The transfer already ran concurrently on the prefetch stream;
            # dispatch joined it before returning, so the aliases are safe to
            # hand out directly.
            state.local_weights = resources.workspace.local_weights(generation=state.generation)
        else:
            state.local_weights = resources.workspace.prefetch_weights(
                buffer=buffer,
                plan=plan,
                generation=state.generation,
            )
        return hidden_nvsh, route_weights_nvs, cu_seqlens

    with torch.profiler.record_function("MoonEP::dispatch_forward"):
        result, state.weights_ready = resources.enqueue(
            dispatch_and_prefetch,
            inputs=(source_hidden, source_route_weights, topk_ids, tokens_per_expert),
        )
    assert state.dispatch_done is not None
    if not async_op:
        state.dispatch_done.wait()
    return result


def dispatch_backward(
    state: _MoonEPLayerCallState,
    grad_hidden_nvsh: torch.Tensor,
    grad_route_weights_nvs: torch.Tensor,
) -> ProjectionPair:
    """Combine activation and route-weight gradients on the saved plan."""
    resources = state.resources
    buffer = resources.buffer
    grad_hidden_nvsh = grad_hidden_nvsh.contiguous()
    grad_route_weights_nvs = grad_route_weights_nvs.contiguous()

    def combine_gradients() -> tuple[torch.Tensor, torch.Tensor]:
        grad_hidden, grad_route_weights, no_event = buffer.combine(
            plan=state.plan,
            hidden_nvsh=grad_hidden_nvsh,
            route_weights_nvs=grad_route_weights_nvs,
            async_finish=False,
            zero_copy=False,
            mb_idx=state.grad_slot,
        )
        assert grad_route_weights is not None and no_event is None
        return grad_hidden, grad_route_weights

    with torch.profiler.record_function("MoonEP::dispatch_backward"):
        result, done = resources.enqueue(
            combine_gradients,
            inputs=(grad_hidden_nvsh, grad_route_weights_nvs),
        )
        done.wait()
    return result


def combine_forward(
    state: _MoonEPLayerCallState,
    expert_output: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    async_op: bool,
) -> torch.Tensor:
    """Fuse route scaling into the combine boundary on the saved plan."""
    resources = state.resources
    buffer = resources.buffer

    def combine_output() -> torch.Tensor:
        output, gathered_weights, no_event = buffer.combine(
            plan=state.plan,
            hidden_nvsh=expert_output,
            hidden_scales_nvs=route_weights,
            route_weights_nvs=None,
            async_finish=False,
            zero_copy=False,
            mb_idx=state.grad_slot,
        )
        assert gathered_weights is None and no_event is None
        return output

    with torch.profiler.record_function("MoonEP::combine_forward"):
        output, state.combine_done = resources.enqueue(
            combine_output,
            inputs=(expert_output, route_weights),
        )
    if not async_op:
        state.combine_done.wait()
    return output


def combine_backward(state: _MoonEPLayerCallState, grad_output: torch.Tensor) -> tuple[torch.Tensor, Any]:
    """Replay duplicated weights on the saved plan and return weighted grad."""
    resources = state.resources
    buffer = resources.buffer
    grad_output = grad_output.contiguous()

    def dispatch_gradient_and_prefetch() -> tuple[torch.Tensor, torch.cuda.Event]:
        # FSDP pre-backward has restored this generation; same Adapter call,
        # same sequence.
        replay_home_parameters = resources.landing.prepare(layer=state.layer, generation=state.generation)
        if state.home_parameters is None:
            raise RuntimeError("MoonEP backward has no forward home Parameters")
        if any(
            replay is not forward
            for replay, forward in zip(replay_home_parameters, state.home_parameters, strict=True)
        ):
            raise RuntimeError("MoonEP backward observed a different FSDP unsharded Parameter")

        def start_prefetch(plan: Any) -> Any:
            """Backward's weight replay, launched to run alongside dispatch."""
            return resources.workspace.start_prefetch(
                buffer=buffer,
                plan=plan,
                generation=state.generation,
                stream=buffer.prefetch_stream,
            )

        async_parallel = _async_parallel_enabled()
        grad_weighted, no_weights, no_cu, reused_plan = buffer.dispatch(
            grad_output,
            plan=state.plan,
            async_finish=False,
            zero_copy=False,
            prefetch_hook=start_prefetch if async_parallel else None,
            mb_idx=state.grad_slot,
        )
        # ★ Ascend buffer.dispatch 无条件返回 cu_seqlens (plan 的视图, 非 None),
        #   权重在 plan 复用模式下也不产出 — 断言只锁 plan 身份 (数据面正确性:
        #   grad_weighted 按 state.plan 的 dst 打包, 复用同一 plan 即正确)。
        assert no_weights is None and reused_plan is state.plan
        gradient_dispatch_done = torch.cuda.current_stream().record_event()
        if not async_parallel:
            resources.workspace.prefetch_weights(buffer=buffer, plan=state.plan, generation=state.generation)
        return grad_weighted, gradient_dispatch_done

    with torch.profiler.record_function("MoonEP::combine_backward"):
        (grad_weighted, gradient_dispatch_done), replay_done = resources.enqueue(
            dispatch_gradient_and_prefetch,
            inputs=(grad_output,),
        )
        # Route-scale backward overlaps weight replay but cannot read the
        # dispatched gradient before this device event.
        gradient_dispatch_done.wait()
    return grad_weighted, replay_done


def start_gradient_completion(state: _MoonEPLayerCallState, gradients: ProjectionPair | None) -> None:
    """Hand the allocation-return dW to the workspace for the home return.

    ``gradients`` is None on the X8 path (dW already written into the heap by
    ``_GMMWithGradWeightOut``); the workspace then skips add_/copy_/zero_ and
    only runs ``reduce_grad_bf16`` + returns the home views.
    """
    if state.gradient_completion is not None:
        raise RuntimeError("MoonEP gradient completion was started twice")
    resources = state.resources
    x8 = gradients is None

    # On the X8 path the dW was written to the heap grad views by the grouped
    # GEMM backward on the compute stream; pass those views as enqueue inputs
    # so the comm stream (reduce_grad_bf16) waits for the compute-stream writes.
    if x8:
        workspace = resources.workspace
        grad_slot = state.grad_slot
        inputs: tuple[torch.Tensor, ...] = (
            workspace.grad_view(grad_slot, 0),
            workspace.grad_view(grad_slot, 1),
        )
    else:
        inputs = gradients  # type: ignore[assignment]

    with torch.profiler.record_function("MoonEP::gradient_handoff"):
        # The workspace owns the ``B`` split, the home-prefix zero-or-add, the
        # duplicate-suffix copy, and the EP exact-sum reduction. ``initialize``
        # is the call-local flag the Dispatcher owns (ADR-0027). On X8 dW was
        # written by the GEMM (first producer overwrites the whole slot, later
        # producers add_ it), so return_expert_gradients skips zero/add_/copy_
        # and only reduces.
        home_grads, done = resources.enqueue(
            lambda: resources.workspace.return_expert_gradients(
                buffer=resources.buffer,
                plan=state.plan,
                gradients=gradients,
                grad_slot=state.grad_slot,
                initialize=state.grad_slot not in state.layer_gradients.reduced,
                x8=x8,
            ),
            inputs=inputs,
        )
    state.layer_gradients.reduced.add(state.grad_slot)
    state.layer_gradients.initialized = True
    state.gradient_completion = (home_grads, done)


def finish_gradient_completion(
    state: _MoonEPLayerCallState,
) -> tuple[tuple[nn.Parameter, nn.Parameter], ProjectionPair]:
    """Wait on the device event; the layer Join owns the single handoff."""
    completion = state.gradient_completion
    if completion is None:
        raise RuntimeError("MoonEP gradient completion was not started")
    home_grads, done = completion
    done.wait()

    if state.home_parameters is None:
        raise RuntimeError("MoonEP gradient completion has no home Parameters")
    home_parameters = state.home_parameters

    state.home_parameters = None
    state.gradient_completion = None
    return home_parameters, home_grads


class _MoonEPExpertGradBridge(torch.autograd.Function):
    """Consume both dWs before releasing the expert activation gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        call_state: _MoonEPLayerCallState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.call_state = call_state
        ctx.x8_active = call_state.x8_active
        # ★ 2026-10-22: w13/w2 的 dW 在 X8 下由 _GMMWithGradWeightOut 直写堆,
        #   这里必然收到 None。默认的 materialize_grads 会为每个未定义梯度
        #   分配并清零一份全尺寸 zeros_like (2B,O,I) — 每步 40×(268MB+134MB)
        #   纯浪费。关掉后 backward 直接拿到 None (X8 分支本就不看 dw)。
        ctx.set_materialize_grads(False)
        return hidden_states, w13, w2

    @staticmethod
    def backward(
        ctx: Any, grad_hidden: torch.Tensor, dw13: torch.Tensor | None, dw2: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, None, None]:
        # X8: dW was written directly into the heap by _GMMWithGradWeightOut,
        # so autograd delivers None here. The bridge still fires (via the
        # re-enabled dx) to trigger reduce_grad_bf16 + FSDP .grad writeback.
        gradients: ProjectionPair | None = None if ctx.x8_active else (dw13, dw2)
        start_gradient_completion(cast(_MoonEPLayerCallState, ctx.call_state), gradients)
        # dW is now owned by MoonEP; do not also accumulate it on anchor leaves.
        return grad_hidden, None, None, None


class _MoonEPLayerGradJoin(torch.autograd.Function):
    """Join every microbatch before the native FSDP input backward hook."""

    @staticmethod
    def forward(
        ctx: Any,
        call_states: tuple[_MoonEPLayerCallState, ...],
        *layer_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.call_states = call_states
        return layer_inputs

    @staticmethod
    def backward(ctx: Any, *grad_inputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        # Event.wait inserts a dependency into the current CUDA stream; it does
        # not block the Python host or poll event readiness.
        # ★★ B6 (2026-10-27) ILMB: each microbatch owns its own gradient slot, and
        # the home prefix is slot-local (``get_xt_grad_view`` offsets by
        # ``slot_idx * per_slot``), so the call states no longer alias one shared
        # home. Publish every pair and let ``accumulate_fsdp_unsharded_expert_gradients``
        # sum them with ``add_``; publishing only the last would drop all but one
        # microbatch. With G=1 the loop runs once and behaviour is unchanged.
        for call_state in ctx.call_states:
            call_parameters, call_grads = finish_gradient_completion(call_state)
            accumulate_fsdp_unsharded_expert_gradients(call_parameters, call_grads)
        return (None, *grad_inputs)


class _DispatchAutograd(torch.autograd.Function):
    """Bridge the dispatch/combine pair into PyTorch autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        source_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        source_route_weights: torch.Tensor,
        call_state: _MoonEPLayerCallState,
        async_op: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.call_state = call_state
        hidden_nvsh, route_weights_nvs, cu_seqlens = dispatch_forward(
            call_state,
            source_hidden,
            topk_ids,
            tokens_per_expert,
            source_route_weights,
            async_op=async_op,
        )
        ctx.mark_non_differentiable(cu_seqlens)
        return hidden_nvsh, route_weights_nvs, cu_seqlens

    @staticmethod
    def backward(
        ctx: Any,
        grad_hidden_nvsh: torch.Tensor,
        grad_route_weights_nvs: torch.Tensor,
        grad_cu_seqlens: None,
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, None, None]:
        del grad_cu_seqlens
        grad_hidden, grad_route_weights = dispatch_backward(
            cast(_MoonEPLayerCallState, ctx.call_state),
            grad_hidden_nvsh,
            grad_route_weights_nvs,
        )
        return grad_hidden, None, None, grad_route_weights, None, None


def _route_weights_full_slots(
    route_weights: torch.Tensor,
    dst: torch.Tensor,
    nvs: int,
    group: Any,
) -> torch.Tensor:
    """Assemble the complete slot-major route weights for this rank's segment.

    ``route_weights`` is token-major (``[S*K]`` flattened, row ``t*K+k``), and each entry's
    slot lives at ``decode(dst[t*K+k])`` in the **global** packed space
    ``[0, world * nvs)`` -- a bijection per the plan. ``grad_weighted`` / ``expert_output``
    have one row per slot of this rank's segment, and those slots mostly hold *other* ranks'
    tokens (measured ~7% of a rank's own ``dst`` entries land in its own segment at EP=16).
    The missing weights live on the tokens' home ranks, so every rank scatters its own
    token-major weights into global packed positions and the expert-parallel group sums
    them; each global slot is written by exactly one rank, so the sum is exact.

    Args:
        route_weights (torch.Tensor): ``[S*K]`` token-major weights of this rank's tokens.
        dst (torch.Tensor): ``[S*K]`` global packed destinations (``dest * nvs + offset``).
        nvs (int): slots per rank, including per-segment padding.
        group (Any): the expert-parallel process group.

    Returns:
        torch.Tensor: ``[nvs]`` float32 weights indexed by this rank's local slot.
    """
    flat_dst = dst.reshape(-1)
    d_abs = torch.where(flat_dst < 0, -flat_dst - 1, flat_dst).to(torch.int64)
    flat_w = route_weights.reshape(-1).to(torch.float32)
    world = dist.get_world_size(group)
    full = torch.zeros(world * nvs, dtype=torch.float32, device=flat_w.device)
    full.index_copy_(0, d_abs, flat_w)
    dist.all_reduce(full, op=dist.ReduceOp.SUM, group=group)
    rank = dist.get_rank(group)
    return full[rank * nvs : (rank + 1) * nvs]


def _grad_route_slots_to_tokens(
    grad_route_slot: torch.Tensor,
    dst: torch.Tensor,
    nvs: int,
    group: Any,
) -> torch.Tensor:
    """Gather a slot-indexed route-weight gradient back to token-major order.

    ``grad_route_slot`` is ``[nvs]`` and holds, on each rank, the values of **that rank's
    own segment** of the global packed space. Each token-major entry ``i = t*K+k`` of this
    rank's tokens points at a global slot ``decode(dst[i])`` that generally lives on another
    rank, so the value is pulled from the owner: the per-rank segments are gathered into the
    global array and indexed by ``dst``.

    A plain ``all_reduce`` over per-rank partial arrays is **not** equivalent: every rank's
    ``[S*K]`` array is indexed by its *own* tokens, and the ranks hold different tokens, so
    the index spaces do not align.

    Args:
        grad_route_slot (torch.Tensor): ``[nvs]`` gradient indexed by this rank's local slot.
        dst (torch.Tensor): ``[S*K]`` global packed destinations from the saved plan.
        nvs (int): slots per rank, including per-segment padding.
        group (Any): the expert-parallel process group.

    Returns:
        torch.Tensor: ``[S*K]`` float32 gradient in token-major order.
    """
    seg = grad_route_slot.reshape(-1).to(torch.float32).contiguous()
    world = dist.get_world_size(group)
    gathered = torch.empty(world * nvs, dtype=torch.float32, device=seg.device)
    dist.all_gather_into_tensor(gathered, seg, group=group)
    flat_dst = dst.reshape(-1)
    d_abs = torch.where(flat_dst < 0, -flat_dst - 1, flat_dst).to(torch.int64)
    return gathered[d_abs]


class _CombineAutograd(torch.autograd.Function):
    """Bridge fused combine and saved-plan dispatch into autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        expert_output: torch.Tensor,
        route_weights: torch.Tensor,
        call_state: _MoonEPLayerCallState,
        async_op: bool,
    ) -> torch.Tensor:
        ctx.call_state = call_state
        ctx.save_for_backward(expert_output, route_weights)
        return combine_forward(call_state, expert_output, route_weights, async_op=async_op)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        grad_weighted, replay_done = combine_backward(cast(_MoonEPLayerCallState, ctx.call_state), grad_output)
        expert_output, route_weights = ctx.saved_tensors
        _cs = cast(_MoonEPLayerCallState, ctx.call_state)
        # ★ 2026-10-12 修复 (MoE 块反向梯度系统性偏低根因, 两条独立路径):
        #   route_weights 是 token-major ([S,K] 展平, 行=t*K+k), 而
        #   grad_weighted/expert_output 的行是本 rank 段的**槽序** (全局打包空间
        #   [0, world*NvS) 中本 rank 的 [rank*NvS, (rank+1)*NvS) 段)。
        #   plan.dst 是 (token, expert) → 全局槽的双射 (实测 hit=nslot, coll=0),
        #   且本 rank 段里大多是**别的 rank** 的 token (EP=16 实测只有 ~7% 的
        #   dst 条目落在本 rank 段内)。所以:
        #   1) grad_expert 的逐行权重必须跨 rank 组装完整槽序数组 (本 rank 只
        #      掌握自己 token 的权重); 旧的按 loff 直接 index_copy 会把别的
        #      rank 段的槽也写成本 rank token 的权重。
        #   2) grad_route_weights 按槽序产出, 却要作为 token-major route_weights
        #      的梯度返回 —— 必须按 dst 从槽拥有者处 gather 回 token 序。
        _plan = _cs.plan
        _dst = _plan.dst if _plan is not None else None
        if _dst is not None:
            _grp = _cs.resources.ep_group
            _nvs = int(_plan.NvS)
            _rw = _route_weights_full_slots(route_weights, _dst, _nvs, _grp)
        else:
            _rw = route_weights
        grad_expert, grad_route_weights = route_weight_rows_backward(
            grad_weighted,
            expert_output,
            _rw,
        )
        if _dst is not None:
            grad_route_weights = _grad_route_slots_to_tokens(
                grad_route_weights,
                _dst,
                _nvs,
                _grp,
            )
        # The next autograd node immediately reads duplicated weights.
        replay_done.wait()
        return grad_expert, grad_route_weights, None, None


class MoonEPDispatcher(
    GenericDispatcher[
        MoonEPPreDispatchResult,
        MoonEPDispatchResult,
        MoonEPPostDispatchResult,
        MoonEPPreCombineResult,
        MoonEPCombineResult,
        MoonEPPostCombineResult,
    ]
):
    """Adapt one routed layer to XTuner's six-stage dispatcher interface.

    This class owns only layer-static policy. Every dispatch creates a fresh
    ``_MoonEPLayerCallState`` that the module-level transaction functions
    advance with the call's plan, event, weight, and gradient state.
    """

    def __init__(
        self,
        *,
        runtime: MoonEPModelRuntime,
        layer: _MoonEPLayer,
    ) -> None:
        super().__init__(
            n_routed_experts=runtime._num_experts,
            process_group=runtime._ep_group,
        )
        self._runtime = runtime
        self._layer = layer
        self._next_gradient_slot = 0

    def _new_call_state(self, layer_gradients: _MoonEPLayerGradients) -> _MoonEPLayerCallState:
        """Allocate the next call-local slot and call-state token."""
        resources = self._runtime.resources
        grad_slot = self._next_gradient_slot
        self._next_gradient_slot = (grad_slot + 1) % resources.gradient_slots
        # X8 is default-on for the Ascend MoonEP heap path (dW direct-to-heap).
        # ``XTUNER_MOE_X8=0`` reverts to the autograd-dW bridge path (A/B compare).
        # Gate on Ascend workspace: the ``grad_view`` accessor is Ascend-only.
        x8_active = (
            os.environ.get("XTUNER_MOE_X8", "1") == "1"
            and isinstance(resources.workspace, _ExpertAscendWorkspace)
        )
        return _MoonEPLayerCallState(
            resources=resources,
            layer=self._layer,
            generation=resources.home_generation(self._layer),
            grad_slot=grad_slot,
            layer_gradients=layer_gradients,
            x8_active=x8_active,
        )

    @override
    def prepare_layer_inputs(
        self,
        layer_inputs: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[object | None]]:
        """Create one call-local Join and one plan/duplicate slot per
        branch."""
        gradients = _MoonEPLayerGradients()
        call_states = tuple(self._new_call_state(gradients) for _ in layer_inputs)
        if len(call_states) > self._runtime.resources.gradient_slots:
            raise ValueError("MoonEP layer width exceeds the gradient slot ring")
        # No-grad original forwards build no backward node and never clear H.
        return list(_MoonEPLayerGradJoin.apply(call_states, *layer_inputs)), list(call_states)

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        layer_state: object | None = None,
        async_op: bool = False,
    ) -> MoonEPPreDispatchResult:
        del topk_weights, async_op
        if layer_state is None:
            raise RuntimeError("MoonEP dispatch_preprocess requires layer_state from prepare_layer_inputs")
        if not isinstance(layer_state, _MoonEPLayerCallState):
            raise TypeError("MoonEP layer_state must be a _MoonEPLayerCallState")
        layer_state.resources.expect_tokens_per_rank(hidden_states.shape[0])
        return MoonEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(dtype=torch.int32).contiguous(),
            tokens_per_expert=tokens_per_expert.to(dtype=torch.int32).contiguous(),
            _moonep_call=layer_state,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPDispatchResult:
        if decoding:
            raise NotImplementedError("MoonEP fixed-S training dispatch does not implement decoding")
        # Create the activation autograd edge and start weight prefetch.
        hidden_nvsh, topk_weights_nvs, cu_seqlens = _DispatchAutograd.apply(
            pre_dispatched["hidden_states"],
            pre_dispatched["topk_ids"],
            pre_dispatched["tokens_per_expert"],
            topk_weights.to(dtype=torch.float32).contiguous(),
            pre_dispatched["_moonep_call"],
            async_op,
        )
        return MoonEPDispatchResult(
            hidden_states=hidden_nvsh,
            topk_weights=topk_weights_nvs,
            cu_seqlens=cu_seqlens,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        async_op: bool = False,
    ) -> MoonEPPostDispatchResult:
        del async_op
        return prepare_experts(pre_dispatched["_moonep_call"], dispatched)

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPPreCombineResult:
        del pre_dispatched, dispatched, post_dispatched, async_op, decoding
        return MoonEPPreCombineResult(hidden_states=hidden_states)

    @override
    def combine(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        pre_combined: MoonEPPreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPCombineResult:
        del post_dispatched, decoding
        # Create the fused route-scaled combine autograd edge.
        return MoonEPCombineResult(
            hidden_states=_CombineAutograd.apply(
                pre_combined["hidden_states"],
                dispatched["topk_weights"],
                pre_dispatched["_moonep_call"],
                async_op,
            )
        )

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        pre_combined: MoonEPPreCombineResult,
        combined: MoonEPCombineResult,
        async_op: bool = False,
    ) -> MoonEPPostCombineResult:
        del post_dispatched, pre_combined
        return MoonEPPostCombineResult(
            hidden_states=finish_combine(
                pre_dispatched["_moonep_call"],
                combined["hidden_states"],
                async_op=async_op,
            )
        )


__all__ = [
    "MoonEPDispatcher",
    "MoonEPModelRuntime",
    "MoonEPPreDispatchResult",
    "MoonEPDispatchResult",
    "MoonEPPostDispatchResult",
    "MoonEPPreCombineResult",
    "MoonEPCombineResult",
    "MoonEPPostCombineResult",
    "require_moonep_backend",
]
