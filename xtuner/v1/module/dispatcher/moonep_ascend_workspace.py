"""Ascend (cann-shmem) variant of the MoonEP expert workspace.

This module replaces the GPU-only VMM allocation path
(``_ExpertVMMWorkspace.allocate`` → ``nvl_dist_alloc`` / ``nvl_dist_map`` /
CUDA IPC FD exchange) with a lightweight adapter that pulls zero-copy views
from ``BufferXtuner``'s symmetric heap (``xt_*_view``).  The runtime methods
inherited from ``_ExpertVMMWorkspace`` — ``prefetch_weights``,
``return_expert_gradients``, ``local_compute_view``, ``landing``,
``generation_for`` — consume only ``_WorkspaceLayout`` and are fully
shape-compatible with the Ascend views.

Layout correspondence (``_WorkspaceLayout`` ↔ ``xt_*_view``):

  landings[gen][proj]               = xt_weight_home_view(gen, proj)           [B, O_p, I_p]
  global_weights[gen][proj]         = cat(home, duplicate)                     [E+B, O_p, I_p]
  local_weights[gen][proj]           = cat(home, duplicate)                     [2B, O_p, I_p]
  local_grad_outputs[slot][proj]     = cat(grad_home, grad_duplicate)           [2B, O_p, I_p]
  distributed_duplicate_grads[slot][proj] = placeholder (kernel self-fetches)  [R, B, O_p, I_p]
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence, cast

import torch
import torch.distributed as dist
from typing_extensions import TypedDict

from .moonep_workspace import _ExpertVMMWorkspace, _WorkspaceLayout


class _ExpertAscendWorkspace(_ExpertVMMWorkspace):
    """Ascend workspace: views from BufferXtuner symmetric heap, no VMM/IPC.

    Inherits all runtime methods from ``_ExpertVMMWorkspace`` unchanged —
    they read only ``_WorkspaceLayout`` via ``self._views``, which is
    constructed here from ``BufferXtuner.xt_*_view`` instead of VMM mappings.
    """

    # ---------------------------------------------------------------------------
    # Deferred layout: build_from_buffer() fills _layout after buffer creation.
    # ---------------------------------------------------------------------------

    def __init__(self, *, layout, ep_group, ep_rank, num_experts,
                 experts_per_rank, home_generations):
        # Allow deferred layout (None) — filled by build_from_buffer().
        self._layout = layout
        self._ep_group = ep_group
        self._ep_rank = ep_rank
        self._num_experts = num_experts
        self._experts_per_rank = experts_per_rank
        self._home_generations = home_generations
        # If layout is deferred, set _gradient_slots from gradient_slots param;
        # otherwise infer from layout (same as parent).
        if layout is not None:
            self._gradient_slots = len(layout["local_grad_outputs"])
        else:
            self._gradient_slots = 0  # updated in build_from_buffer()
        self._destroyed = False

    def build_from_buffer(self, buffer: Any, gradient_slots: int) -> None:
        """Fill the deferred layout from BufferXtuner's symmetric heap views.

        Called once from ``buffer_for()`` after the Buffer is created.
        Constructs ``_WorkspaceLayout`` from ``xt_*_view`` and sets
        ``_gradient_slots``.
        """
        if self._layout is not None:
            raise RuntimeError("Ascend workspace layout already built")
        if self._destroyed:
            raise RuntimeError("Ascend workspace has been destroyed")

        ep_size = dist.get_world_size(self._ep_group)
        ep_rank = self._ep_rank
        home_generations = self._home_generations

        # Build _WorkspaceLayout from xt_*_view (same logic as allocate()).
        landings: list[list[torch.Tensor]] = []
        global_weights: list[list[torch.Tensor]] = []
        local_weights: list[list[torch.Tensor]] = []

        for gen in range(home_generations):
            gen_landings: list[torch.Tensor] = []
            gen_globals: list[torch.Tensor] = []
            gen_locals: list[torch.Tensor] = []
            for proj in range(2):
                home_view = buffer.xt_weight_home_view(gen, proj)
                # ★ 整段 [epn+B] 堆视图 (C++ from_blob 从堆基址直建) —
                # grouped GEMM 按专家 id 索引行需 [E+B] 连续; torch.cat 会
                # 分配堆外 tensor (prefetch data_ptr 落堆校验拒)。
                full_view = buffer.xt_weight_full_view(gen, proj)
                gen_landings.append(home_view)
                gen_globals.append(full_view)
                gen_locals.append(full_view)
            landings.append(gen_landings)
            global_weights.append(gen_globals)
            local_weights.append(gen_locals)

        local_grad_outputs: list[list[torch.Tensor]] = []
        distributed_duplicate_grads: list[list[torch.Tensor]] = []

        for slot in range(gradient_slots):
            slot_grads: list[torch.Tensor] = []
            slot_dist: list[torch.Tensor] = []
            for proj in range(2):
                grad_home = buffer.xt_grad_home_view(slot, proj)
                grad_dup = buffer.xt_grad_duplicate_view(slot, proj)
                # ★ 同款 C++ 整段视图 [epn+B] (禁 cat/as_strided — storage 边界)
                grad_full = buffer.xt_grad_full_view(slot, proj)
                slot_grads.append(grad_full)
                # ★ 显存炸弹修复 (2026-09-20): 旧实现 .expand(...).contiguous()
                # 物化 [R,B,O,I] = R× dup (R=2/epn=20/H=HP=6144 → 17GB/rank
                # 永久占用; R=16 → 138GB 必 OOM)。但 reduce_grad_bf16 对此参数
                # 仅 assert len(tuple)==2, 数据全不读 (kernel 自取对称堆)。
                # 改为保留 expand view (零分配, shape [R,B,O,I] 契约不变)。
                slot_dist.append(
                    grad_dup.unsqueeze(0).expand(ep_size, *grad_dup.shape)
                )
            local_grad_outputs.append(slot_grads)
            distributed_duplicate_grads.append(slot_dist)

        keepalives = (
            *sum(landings, []),
            *sum(global_weights, []),
            *sum(local_weights, []),
            *sum(local_grad_outputs, []),
        )

        self._layout = {
            "landings": tuple(tuple(p) for p in landings),
            "global_weights": tuple(tuple(p) for p in global_weights),
            "local_weights": tuple(tuple(p) for p in local_weights),
            "local_grad_outputs": tuple(tuple(p) for p in local_grad_outputs),
            "distributed_duplicate_grads": tuple(tuple(p) for p in distributed_duplicate_grads),
            "keepalives": keepalives,
        }
        self._gradient_slots = gradient_slots

    @classmethod
    def allocate(
        cls,
        *,
        projection_shapes: Sequence[tuple[int, int]],
        num_experts: int,
        ep_group: dist.ProcessGroup,
        gradient_slots: int,
        home_generations: int = 2,
        buffer: Any = None,
    ) -> _ExpertAscendWorkspace:
        """Build the workspace layout from BufferXtuner's symmetric heap views.

        Unlike the GPU ``_ExpertVMMWorkspace.allocate`` which allocates VMM
        chunks, exchanges IPC FDs, and maps cross-rank views via
        ``nvl_dist_map``, this method pulls zero-copy views directly from the
        already-allocated cann-shmem symmetric heap owned by BufferXtuner.

        Args:
            projection_shapes: ((2*Hp, H), (H, Hp)) — w13 / w2 projection shapes.
            num_experts: total routed experts E.
            ep_group: EP process group (must be initialized).
            gradient_slots: N gradient slots (= intra_layer_micro_batch).
            home_generations: G home generations (always 2).
            buffer: BufferXtuner instance (must be created and workspace-ready).
        """
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
        experts_per_rank = num_experts // ep_size

        if buffer is None:
            # Deferred mode: layout built later by ``build_from_buffer()``.
            # Used when install_after_fsdp runs before the first forward
            # (Buffer is lazily created in buffer_for once Fixed-S is known).
            return cls(
                layout=None,  # deferred — filled by build_from_buffer()
                ep_group=ep_group,
                ep_rank=ep_rank,
                num_experts=num_experts,
                experts_per_rank=experts_per_rank,
                home_generations=home_generations,
            )
        # ★ 2026-09-23 Home 单代开关: 允许 1 (单代省一半 home) / 2 (双代)
        if home_generations not in (1, 2):
            raise ValueError(f"Ascend workspace requires home_generations 1 or 2, got {home_generations}")
        if len(projection_shapes) != 2:
            raise ValueError("MoonEP requires fused w1/w3 and w2 projections (projection_shapes len=2)")

        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
        experts_per_rank = num_experts // ep_size

        # ------------------------------------------------------------------
        # Build _WorkspaceLayout from xt_*_view.
        #
        # Projection indexing: proj=0 → w13 [B, 2Hp, H], proj=1 → w2 [B, H, Hp]
        # ------------------------------------------------------------------

        # landings[G][P], each [B, O_p, I_p] — FSDP AllGather targets.
        landings: list[list[torch.Tensor]] = []
        # global_weights[G][P], each [E+B, O_p, I_p] — prefetch addresses.
        global_weights: list[list[torch.Tensor]] = []
        # local_weights[G][P], each [2B, O_p, I_p] — grouped GEMM compute alias.
        local_weights: list[list[torch.Tensor]] = []

        for gen in range(home_generations):
            gen_landings: list[torch.Tensor] = []
            gen_globals: list[torch.Tensor] = []
            gen_locals: list[torch.Tensor] = []
            for proj in range(2):
                home_view = buffer.xt_weight_home_view(gen, proj)       # [B, O_p, I_p]
                dup_view = buffer.xt_weight_duplicate_view(gen, proj)   # [B, O_p, I_p]
                gen_landings.append(home_view)
                # ★ 2026-09-23 ZEROCOPY (MOONEP_GMM_ZEROCOPY=1 默认): global/
                #   local weights 传 [home, dup] 两段列表 (免 torch.cat 物化
                #   [E+B] 副本 -2.4GB@GLM); group_gemm 识别列表 → npu_gmm_moep
                #   TensorList 配对。=0 回退 cat 单段 (兼容/对比)。
                if int(os.environ.get("MOONEP_GMM_ZEROCOPY", "1")) == 1:
                    gen_globals.append([home_view, dup_view])   # [home B + dup B]
                    gen_locals.append([home_view, dup_view])    # [2B, ...] 同构
                else:
                    gen_globals.append(torch.cat([home_view, dup_view], dim=0))
                    gen_locals.append(torch.cat([home_view, dup_view], dim=0))
            landings.append(gen_landings)
            global_weights.append(gen_globals)
            local_weights.append(gen_locals)

        # local_grad_outputs[N][P], each [2B, O_p, I_p] — [home, duplicate] grad.
        # distributed_duplicate_grads[N][P], each [R, B, O_p, I_p] —
        #   On Ascend the kernel self-fetches peer duplicate slots via
        #   symmetric heap addressing; this is a non-empty contract check
        #   placeholder (shape [R, B, ...] for API compatibility).
        local_grad_outputs: list[list[torch.Tensor]] = []
        distributed_duplicate_grads: list[list[torch.Tensor]] = []

        for slot in range(gradient_slots):
            slot_grads: list[torch.Tensor] = []
            slot_dist: list[torch.Tensor] = []
            for proj in range(2):
                grad_home = buffer.xt_grad_home_view(slot, proj)           # [B, O_p, I_p]
                grad_dup = buffer.xt_grad_duplicate_view(slot, proj)       # [B, O_p, I_p]
                slot_grads.append(torch.cat([grad_home, grad_dup], dim=0))  # [2B, O_p, I_p]
                # ★ 显存炸弹修复 (2026-09-20): 旧 .expand(...).contiguous() 物化
                # [R,B,O,I] = R× dup (R=2→17GB/rank; R=16→138GB 必 OOM)。
                # reduce_grad_bf16 仅 assert len==2, 数据不读 (kernel 自取堆)。
                # 改为 expand view (零分配, shape 契约 [R,B,O,I] 不变)。
                slot_dist.append(
                    grad_dup.unsqueeze(0).expand(ep_size, *grad_dup.shape)
                )
            local_grad_outputs.append(slot_grads)
            distributed_duplicate_grads.append(slot_dist)

        # keepalives: hold strong refs to the heap views so the storage isn't
        # reclaimed; the BufferXtuner owns the underlying heap.
        keepalives = (
            *sum(landings, []),
            *sum(global_weights, []),
            *sum(local_weights, []),
            *sum(local_grad_outputs, []),
        )

        layout: _WorkspaceLayout = {
            "landings": tuple(tuple(p) for p in landings),
            "global_weights": tuple(tuple(p) for p in global_weights),
            "local_weights": tuple(tuple(p) for p in local_weights),
            "local_grad_outputs": tuple(tuple(p) for p in local_grad_outputs),
            "distributed_duplicate_grads": tuple(tuple(p) for p in distributed_duplicate_grads),
            "keepalives": keepalives,
        }

        return cls(
            layout=layout,
            ep_group=ep_group,
            ep_rank=ep_rank,
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            home_generations=home_generations,
        )

    # ---------------------------------------------------------------------------
    # Override: destroy() — no VMM/FD teardown; just drop layout refs.
    # The BufferXtuner owns the symmetric heap and handles SHMEM finalize.
    # ---------------------------------------------------------------------------

    def destroy(self) -> None:
        """Release layout references; BufferXtuner owns heap lifecycle."""
        if self._destroyed:
            return
        # No torch.cuda.synchronize() — NPU uses torch.npu.synchronize().
        # BufferXtuner.destroy() handles SHMEM finalize; we only drop views.
        dist.barrier(group=self._ep_group)
        self._layout = None
        self._destroyed = True

    # ---------------------------------------------------------------------------
    # X8 accessors: public hooks for the dispatcher to pull per-slot heap
    # gradient views so it can inject them as ``grad_weight_out`` into the
    # expert GroupedLinear forward calls. ``local_grad_outputs[slot][proj]``
    # is the full [epn+B, O, I] home+dup view (X8 writes home add_ + dup copy_).
    # ---------------------------------------------------------------------------

    def grad_view(self, slot: int, projection: int) -> torch.Tensor:
        """Full [epn+B, O, I] heap gradient slot view (home+dup) for X8."""
        if self._layout is None:
            raise RuntimeError("Ascend workspace layout not built (buffer deferred?)")
        return self._views["local_grad_outputs"][slot][projection]

    def home_grad_view(self, slot: int, projection: int) -> torch.Tensor:
        """Home-only [B, O, I] heap gradient slot view (for the first-mb zero_)."""
        if self._layout is None:
            raise RuntimeError("Ascend workspace layout not built (buffer deferred?)")
        return self._views["local_grad_outputs"][slot][projection][: self._experts_per_rank]

    def __del__(self) -> None:
        if getattr(self, "_destroyed", True) or getattr(self, "_layout", None) is None:
            return
        warnings.warn(
            "MoonEP Ascend workspace was not destroyed explicitly; "
            "resources may leak.",
            ResourceWarning,
        )
        # Keep layout alive for rank-divergence safety (same as GPU path).
        from .moonep_workspace import _UNDISPOSED_WORKSPACE_TENSORS
        _UNDISPOSED_WORKSPACE_TENSORS.append(self._layout)
        self._layout = None
