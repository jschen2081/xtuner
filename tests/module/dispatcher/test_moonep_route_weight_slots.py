"""Regression tests for the MoonEP combine-backward route-weight index repair.

``_CombineAutograd.backward`` consumes token-major ``route_weights`` but produces
slot-major ``grad_weighted`` / ``expert_output`` rows. ``plan.dst`` maps every
``(token, expert)`` entry to a slot in the **global** packed space
(``dest_rank * NvS + offset``) and is a bijection onto it, so the conversion between
the two orders has to go through the global space:

* :func:`_route_weights_full_slots` scatters each rank's own token-major weights
  into global packed positions and all-reduces, so a rank's segment is filled with
  the *home-rank* weights of the (mostly foreign) tokens it actually holds.
* :func:`_grad_route_slots_to_tokens` gathers the per-rank slot gradients back into
  token-major order from the segment that owns each entry's slot.

The earlier code scattered every entry's ``loff`` into the local segment unfiltered
(EP=16: ~93% of a rank's ``dst`` entries point at other ranks' segments, so foreign
entries overwrote local slots) and returned the slot-major ``grad_route`` unchanged,
which left the 13 MoE ``gate.weight`` gradients at 1--4% of the base run.
"""

import os

import pytest
import torch
import torch.distributed as dist

from xtuner.v1.module.dispatcher.moonep import (
    _grad_route_slots_to_tokens,
    _route_weights_full_slots,
)

NVS = 4


@pytest.fixture(scope="module")
def ep_group():
    """Single-rank gloo group standing in for the expert-parallel group."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield dist.group.WORLD
    dist.destroy_process_group()


def _run_as_rank(
    monkeypatch,
    build,
    *,
    rank: int,
    world: int,
    full: torch.Tensor | None = None,
    segs: list[torch.Tensor] | None = None,
):
    """Run ``build`` with the dist collectives faked for a ``world``-rank group.

    The helpers only communicate through ``all_reduce`` (a sum of every rank's
    scattered global array) and ``all_gather_into_tensor`` (a concatenation of
    every rank's segment), so a single process can stand in for rank ``rank`` of
    a ``world``-rank expert-parallel group by pre-computing those results.
    """
    monkeypatch.setattr(dist, "get_world_size", lambda group: world)
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)
    if full is not None:
        monkeypatch.setattr(
            dist, "all_reduce", lambda tensor, op=None, group=None: tensor.copy_(full)
        )
    if segs is not None:
        monkeypatch.setattr(
            dist,
            "all_gather_into_tensor",
            lambda out, tensor, group=None: out.copy_(torch.cat(segs)),
        )
    return build()


class TestRouteWeightsFullSlots:
    def test_identity_dst_keeps_token_order(self, ep_group):
        weights = 10.0 + torch.arange(NVS, dtype=torch.float32)
        dst = torch.arange(NVS, dtype=torch.int32)
        slots = _route_weights_full_slots(weights, dst, NVS, ep_group)
        assert torch.equal(slots, weights)

    def test_permuted_dst_scatters_by_destination(self, ep_group):
        weights = 10.0 + torch.arange(NVS, dtype=torch.float32)
        dst = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
        slots = _route_weights_full_slots(weights, dst, NVS, ep_group)
        assert torch.equal(slots, weights[torch.argsort(dst)])

    def test_dup_entry_decodes_to_the_dup_slot(self, ep_group):
        """``dst < 0`` de-duplicates a slot but still names that slot's position."""
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0])
        dst = torch.tensor([3, -1 - 1, 2, -1 - 3], dtype=torch.int32)
        slots = _route_weights_full_slots(weights, dst, NVS, ep_group)
        assert slots[1].item() == 20.0
        assert slots[3].item() == 40.0
        assert slots[2].item() == 30.0

    def test_dtype_is_float32(self, ep_group):
        weights = torch.arange(NVS, dtype=torch.bfloat16)
        dst = torch.arange(NVS, dtype=torch.int32)
        slots = _route_weights_full_slots(weights, dst, NVS, ep_group)
        assert slots.dtype is torch.float32


class TestRouteWeightsFullSlotsMultiRank:
    def test_foreign_entries_fill_the_segment_from_owners(self, monkeypatch):
        """A rank's segment holds other ranks' tokens — their weights must arrive.

        At EP=16 only ~7% of a rank's own ``dst`` entries land in its own segment,
        so the all-reduce of the globally scattered weights is what makes the
        per-slot weights correct rather than a overwrite of local entries.
        """
        w0 = torch.tensor([10.0, 20.0, 30.0, 40.0])
        d0 = torch.tensor([0, 4, 1, 5], dtype=torch.int32)  # slots 0,1 local; 4,5 rank 1
        w1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        d1 = torch.tensor([6, 2, 7, 3], dtype=torch.int32)  # slots 2,3 local; 6,7 rank 0

        full = torch.zeros(2 * NVS, dtype=torch.float32)
        for w, d in ((w0, d0), (w1, d1)):
            full[torch.where(d < 0, -d - 1, d).to(torch.int64)] = w

        s0 = _run_as_rank(
            monkeypatch,
            lambda: _route_weights_full_slots(w0, d0, NVS, None),
            rank=0,
            world=2,
            full=full,
        )
        s1 = _run_as_rank(
            monkeypatch,
            lambda: _route_weights_full_slots(w1, d1, NVS, None),
            rank=1,
            world=2,
            full=full,
        )
        # slots 0,1 written by rank 0's tokens; slots 2,3 by rank 1's tokens
        assert torch.equal(s0, torch.tensor([10.0, 30.0, 2.0, 4.0]))
        assert torch.equal(s1, torch.tensor([20.0, 40.0, 1.0, 3.0]))

    def test_unwritten_segment_slots_stay_zero(self, monkeypatch):
        """Slots nobody points at must stay zero, not pick up a foreign weight."""
        w0 = torch.tensor([7.0, 8.0, 9.0, 10.0])
        d0 = torch.tensor([4, 5, 6, 7], dtype=torch.int32)  # all in rank 1's segment
        w1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # entry 1 is a dup of slot 0, so slot 1 is pointed at by nobody
        d1 = torch.tensor([0, -1, 2, 3], dtype=torch.int32)

        full = torch.zeros(2 * NVS, dtype=torch.float32)
        for w, d in ((w0, d0), (w1, d1)):
            full[torch.where(d < 0, -d - 1, d).to(torch.int64)] = w

        s0 = _run_as_rank(
            monkeypatch,
            lambda: _route_weights_full_slots(w1, d1, NVS, None),
            rank=0,
            world=2,
            full=full,
        )
        s1 = _run_as_rank(
            monkeypatch,
            lambda: _route_weights_full_slots(w0, d0, NVS, None),
            rank=1,
            world=2,
            full=full,
        )
        assert torch.equal(s0, torch.tensor([2.0, 0.0, 3.0, 4.0]))
        # rank 0's tokens fill rank 1's segment; slot 1 stays zero (nobody points at it)
        assert torch.equal(s1, torch.tensor([7.0, 8.0, 9.0, 10.0]))


class TestGradRouteSlotsToTokens:
    def test_entry_reads_its_decoded_slot(self, ep_group):
        grad_slot = 100.0 + 10.0 * torch.arange(NVS, dtype=torch.float32)
        dst = torch.arange(NVS - 1, -1, -1, dtype=torch.int32)
        out = _grad_route_slots_to_tokens(grad_slot, dst, NVS, ep_group)
        assert torch.equal(out, grad_slot.flip(0))

    def test_dup_entry_reads_the_dup_slot(self, ep_group):
        grad_slot = torch.arange(NVS, dtype=torch.float32)
        dst = torch.tensor([2, -1 - 1, 0, -1 - 3], dtype=torch.int32)
        out = _grad_route_slots_to_tokens(grad_slot, dst, NVS, ep_group)
        assert out[0].item() == 2.0
        assert out[1].item() == 1.0
        assert out[3].item() == 3.0

    def test_shape_and_dtype_follow_dst(self, ep_group):
        grad_slot = torch.ones(NVS, dtype=torch.float32)
        dst = torch.zeros(NVS, dtype=torch.int32)
        out = _grad_route_slots_to_tokens(grad_slot, dst, NVS, ep_group)
        assert out.shape == dst.shape
        assert out.dtype is torch.float32


class TestGradRouteSlotsToTokensMultiRank:
    def test_entry_reads_the_owning_rank_segment(self, monkeypatch):
        """Each token-major entry pulls its value from the segment that owns its slot."""
        seg0 = 100.0 + torch.arange(NVS, dtype=torch.float32)
        seg1 = 200.0 + torch.arange(NVS, dtype=torch.float32)
        d0 = torch.tensor([4, 0, 5, 1], dtype=torch.int32)  # entries 0,2 on rank 1
        d1 = torch.tensor([6, 2, 7, 3], dtype=torch.int32)  # entries 0,2 on rank 0

        out0 = _run_as_rank(
            monkeypatch,
            lambda: _grad_route_slots_to_tokens(seg0, d0, NVS, None),
            rank=0,
            world=2,
            segs=[seg0, seg1],
        )
        out1 = _run_as_rank(
            monkeypatch,
            lambda: _grad_route_slots_to_tokens(seg1, d1, NVS, None),
            rank=1,
            world=2,
            segs=[seg0, seg1],
        )
        assert torch.equal(out0, torch.tensor([200.0, 100.0, 201.0, 101.0]))
        assert torch.equal(out1, torch.tensor([202.0, 102.0, 203.0, 103.0]))


class TestSlotTokenRoundTrip:
    def test_scatter_then_gather_restores_token_order(self, monkeypatch):
        """With a bijective ``dst`` the two conversions are inverses.

        The combine backward relies on this: the router receives exactly one
        contribution per token-major entry, which is what made the 13 MoE
        ``gate.weight`` gradients match the base run to cos = 1.0.
        """
        gen = torch.Generator().manual_seed(0)
        w0 = torch.randn(NVS, generator=gen)
        w1 = torch.randn(NVS, generator=gen)
        # a bijection onto the 2*NVS global slots, split across the two ranks
        d0 = torch.tensor([0, 4, 1, 5], dtype=torch.int32)
        d1 = torch.tensor([6, 2, 7, 3], dtype=torch.int32)

        full = torch.zeros(2 * NVS, dtype=torch.float32)
        for w, d in ((w0, d0), (w1, d1)):
            full[torch.where(d < 0, -d - 1, d).to(torch.int64)] = w

        s0 = _run_as_rank(
            monkeypatch,
            lambda: _route_weights_full_slots(w0, d0, NVS, None),
            rank=0,
            world=2,
            full=full,
        )
        segs = [full[0:NVS], full[NVS:]]
        back0 = _run_as_rank(
            monkeypatch,
            lambda: _grad_route_slots_to_tokens(s0, d0, NVS, None),
            rank=0,
            world=2,
            segs=segs,
        )
        assert torch.allclose(back0, w0)

    def test_scatter_then_gather_restores_token_order_world1(self, ep_group):
        gen = torch.Generator().manual_seed(1)
        weights = torch.randn(NVS, generator=gen)
        perm = torch.randperm(NVS, generator=gen)
        dst = perm.to(torch.int32)
        slots = _route_weights_full_slots(weights, dst, NVS, ep_group)
        back = _grad_route_slots_to_tokens(slots, dst, NVS, ep_group)
        assert torch.allclose(back, weights)
