import os
from typing import Tuple

import torch


def npu_token_permute(
    input_act: torch.Tensor,
    indices: torch.Tensor,
    num_topK: int | None = None,
    num_out_tokens: int | None = None,
    num_negative_one_in_indices: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_out_tokens is not None:
        raise NotImplementedError

    if num_negative_one_in_indices is not None:
        raise NotImplementedError

    if num_topK is not None:
        raise NotImplementedError

    from torch_npu import npu_moe_token_permute

    return npu_moe_token_permute(input_act, indices)


def npu_token_unpermute(
    input_act: torch.Tensor, row_id_map: torch.Tensor, probs: torch.Tensor | None = None
) -> torch.Tensor:
    from torch_npu import npu_moe_token_unpermute

    if probs is not None:
        if not os.environ.get("XT_UNPERMUTE_FP32"):
            # ★ 2026-10-21 诊断 (默认关): XT_UNPERMUTE_FP32=1 时保留 fp32 权重, 不量化成 bf16,
            #   用于验证「base 前向 combine 的 bf16 量化是否就是 base-vs-MoonEP L0 差异的来源」。
            probs = probs.to(torch.bfloat16)
    return npu_moe_token_unpermute(input_act, row_id_map, probs=probs)
