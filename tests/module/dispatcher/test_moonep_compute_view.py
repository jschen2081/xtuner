import os
import unittest
import warnings
from unittest import mock

import torch

from xtuner.v1.module.dispatcher import moonep_workspace
from xtuner.v1.module.dispatcher.moonep_workspace import (
    _MaskedCopy,
    _MaskedZeroTail,
    _masked_copy_mode,
)


_FAKE_OPS = mock.Mock(spec=["masked_copy"])


def _mode(value: str | None) -> str:
    """Resolve ``_masked_copy_mode`` with ``MOONEP_MASKED_COPY`` pinned."""
    env = {} if value is None else {"MOONEP_MASKED_COPY": value}
    with mock.patch.object(moonep_workspace, "moonep_ops", _FAKE_OPS):
        with mock.patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop("MOONEP_MASKED_COPY", None)
            return _masked_copy_mode()


class TestMaskedCopyMode(unittest.TestCase):
    """The three compute-view implementations are selected by one env var."""

    def test_default_is_inplace(self) -> None:
        assert _mode(None) == "inplace"

    def test_explicit_selection(self) -> None:
        for value, expected in (
            ("0", "mul"),
            ("mul", "mul"),
            ("1", "copy"),
            ("copy", "copy"),
            ("true", "copy"),
            ("inplace", "inplace"),
            ("2", "inplace"),
        ):
            assert _mode(value) == expected, value

    def test_mul_fallback_without_the_kernel(self) -> None:
        with mock.patch.object(moonep_workspace, "moonep_ops", None):
            assert _masked_copy_mode() == "mul"


class TestMulCombineOverlapWarning(unittest.TestCase):
    """``mul`` silently disables the combine overlap, so it has to say so once."""

    def setUp(self) -> None:
        moonep_workspace._mul_overlap_warning_emitted = False

    def _resolve(self) -> str:
        with mock.patch.object(moonep_workspace, "moonep_ops", _FAKE_OPS):
            return _masked_copy_mode()

    def test_warns_with_combine_overlap(self) -> None:
        with mock.patch.dict(
            os.environ, {"MOONEP_MASKED_COPY": "mul", "MOONEP_COMBINE_OVERLAP": "1"}, clear=False
        ):
            with self.assertWarns(RuntimeWarning):
                assert self._resolve() == "mul"
            # Emitted once per process, not once per routed call.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert self._resolve() == "mul"
            assert not caught

    def test_silent_without_combine_overlap(self) -> None:
        with mock.patch.dict(os.environ, {"MOONEP_MASKED_COPY": "mul"}, clear=False):
            os.environ.pop("MOONEP_COMBINE_OVERLAP", None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert self._resolve() == "mul"
            assert not caught

    def test_silent_for_other_modes(self) -> None:
        for value in ("inplace", "copy"):
            with mock.patch.dict(
                os.environ, {"MOONEP_MASKED_COPY": value, "MOONEP_COMBINE_OVERLAP": "1"}, clear=False
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    assert self._resolve() == value
                assert not caught, value



def _device_available() -> bool:
    """Whether an accelerator is present, whichever backend ``torch`` maps to."""
    if torch.cuda.device_count() > 0:
        return True
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available())


@unittest.skipUnless(_device_available(), "requires an Ascend NPU")
class TestComputeViewEquivalence(unittest.TestCase):
    """``mul`` / ``copy`` / ``inplace`` must produce the same tensor, bit for bit."""

    rows = 256
    hidden = 512
    covered = 200

    def _source(self) -> tuple[torch.Tensor, torch.Tensor]:
        src = torch.randn(self.rows, self.hidden, dtype=torch.bfloat16, device="cuda")
        return src, torch.tensor([self.covered], dtype=torch.int32, device="cuda")

    def _mul(self, src: torch.Tensor) -> torch.Tensor:
        row_is_covered = torch.arange(self.rows, device=src.device) < self.covered
        return src * row_is_covered.unsqueeze(-1)

    def test_forward_modes_agree(self) -> None:
        if moonep_workspace.moonep_ops is None:
            self.skipTest("moonep_ops is not installed")
        src, covered = self._source()
        expected = self._mul(src)
        assert torch.equal(_MaskedCopy.apply(src, covered, 16), expected)
        assert torch.equal(_MaskedZeroTail.apply(src.clone(), covered, 16), expected)

    def test_inplace_leaves_the_prefix_untouched(self) -> None:
        if moonep_workspace.moonep_ops is None:
            self.skipTest("moonep_ops is not installed")
        src, covered = self._source()
        prefix = src[: self.covered].clone()
        _MaskedZeroTail.apply(src, covered, 16)
        assert torch.equal(src[: self.covered], prefix)
        assert torch.count_nonzero(src[self.covered :]) == 0

    def test_backward_modes_agree(self) -> None:
        if moonep_workspace.moonep_ops is None:
            self.skipTest("moonep_ops is not installed")
        src, covered = self._source()
        grad = torch.randn(self.rows, self.hidden, dtype=torch.bfloat16, device="cuda")

        copy_root = src.clone().requires_grad_(True)
        (copy_grad,) = torch.autograd.grad(_MaskedCopy.apply(copy_root, covered, 16), copy_root, grad)

        # Production hands ``_MaskedZeroTail`` a dispatch output; an in-place
        # op on a leaf that requires grad is rejected by autograd, so mirror
        # that with a one-op chain.
        inplace_root = src.clone().requires_grad_(True)
        inplace_out = _MaskedZeroTail.apply(inplace_root * 1.0, covered, 16)
        (inplace_grad,) = torch.autograd.grad(inplace_out, inplace_root, grad)

        assert torch.equal(copy_grad, inplace_grad)
        assert torch.count_nonzero(inplace_grad[self.covered :]) == 0

    def test_inplace_backward_does_not_write_into_the_incoming_grad(self) -> None:
        """``grad`` belongs to the engine; a sibling branch may still alias it."""
        if moonep_workspace.moonep_ops is None:
            self.skipTest("moonep_ops is not installed")
        src, covered = self._source()
        root = src.clone().requires_grad_(True)
        out = _MaskedZeroTail.apply(root * 1.0, covered, 16)
        grad = torch.randn(self.rows, self.hidden, dtype=torch.bfloat16, device="cuda")
        reference = grad.clone()

        (grad_in,) = torch.autograd.grad(out, root, grad)

        assert torch.count_nonzero(grad_in[self.covered :]) == 0
        assert torch.equal(grad_in[: self.covered], reference[: self.covered])
        # The caller's buffer must come back untouched.
        assert torch.equal(grad, reference)
