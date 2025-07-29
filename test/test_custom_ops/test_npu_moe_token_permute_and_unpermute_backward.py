# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Dict, List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import pytest

from torch_npu import npu_moe_token_permute
from torch_npu import npu_moe_token_unpermute


def permute_with_padded_tokens(tokens, indices):
    """Permute the tokens based on the indices, only used in padding mode."""
    permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))
    return permuted_tokens, indices


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens] or [num_tokens, topk].
        num_out_tokens (int, optional): The effective output token count, when enabling the capacity factor, should equal the number of tokens not dropped. By default, set to None, meaning no tokens are dropped.
        padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity] to denote selected tokens per expert. Defaults to False.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute_with_padded_tokens(
    permuted_tokens: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermutes a padded permuted tokens based on sorted indices and merges the tokens with their corresponding probabilities.
    """
    assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dim()}D."

    probs = probs.view(-1).unsqueeze(-1)
    indices = indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
    assert permuted_tokens.shape == indices.shape

    combined_output = probs * permuted_tokens

    empty_tokens = torch.zeros(
        restore_shape,
        dtype=combined_output.dtype,
        device=combined_output.device,
        requires_grad=True,
    )

    unpermuted_tokens = torch.scatter_add(empty_tokens, 0, indices, combined_output)

    return unpermuted_tokens


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def dtype_tols(te_dtype) -> Dict[str, float]:
    """Estimated tolerances for a datatype

    Based on tolerances for torch.testing.assert_close.

    """
    if te_dtype == torch.float32:
        return dict(rtol=1.0e-6, atol=1.0e-6)
    if te_dtype == torch.float16:
        return dict(rtol=3.0e-3, atol=1.0e-5)
    if te_dtype == torch.bfloat16:
        return dict(rtol=2.0e-2, atol=1.0e-5)
    raise ValueError(f"Unsuppored dtype ({te_dtype})")


class TestNpuFusedPermuteAndUnpermute():

    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_permute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = dtype_tols(dtype)
        token_ori = torch.randn(num_tokens, hidden_size).npu().to(dtype).requires_grad_(True)
        indices_ori = torch.randint(0, num_experts, (num_tokens, topk)).npu()

        token_ori = token_ori.requires_grad_(True)
        token_fused = token_ori.clone().detach().requires_grad_(True)
        indices_fused = indices_ori.clone().detach()

        permuted_tokens_ori, sorted_indices_ori = permute(token_ori, indices_ori)
        permuted_tokens_fused, sorted_indices_fused = npu_moe_token_permute(token_fused, indices_fused)
        permuted_tokens_fused.backward(torch.ones(permuted_tokens_fused.shape).to(torch.bfloat16).npu())

        assert torch.allclose(permuted_tokens_ori, permuted_tokens_fused, **tols)
        # The fusion operator will perform two torch.argsort operations internally
        sorted_indices_ori = torch.argsort(sorted_indices_ori, stable=True).to(sorted_indices_fused.dtype)
        assert torch.equal(sorted_indices_ori, sorted_indices_fused)

    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = dtype_tols(dtype)
        permuted_tokens_ori = torch.randn(num_tokens * topk, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        sorted_indices_ori = torch.argsort(indices.view(-1), stable=True).npu().to(dtype=torch.int32)
        probs_ori = None
        probs_fused = None
        if topk > 1:
            probs_ori = (torch.ones(num_tokens, topk) / topk).npu().to(dtype).requires_grad_(True)
            probs_fused = probs_ori.clone().detach().requires_grad_(True)
        permuted_tokens_fused = permuted_tokens_ori.clone().detach().requires_grad_(True)
        sorted_indices_fused = sorted_indices_ori.clone().detach()

        # The fusion operator will perform two torch.argsort operations internally
        sorted_indices_ori = torch.argsort(sorted_indices_ori, stable=True)
        unpermuted_tokens_ori = unpermute(
            permuted_tokens_ori, sorted_indices_ori, probs=probs_ori)

        unpermuted_tokens_fused = npu_moe_token_unpermute(
            permuted_tokens_fused, sorted_indices_fused, probs=probs_fused)

        unpermuted_tokens_fused.backward(torch.ones(unpermuted_tokens_fused.shape).to(torch.bfloat16).npu())
        assert torch.allclose(unpermuted_tokens_ori, unpermuted_tokens_fused, **tols)

    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_npu_npu_moe_token_permute_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = dtype_tols(dtype)
        tokens = torch.randn(num_tokens, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        probs = None
        if topk > 1:
            probs = (torch.ones_like(indices) / topk).npu().to(dtype)

        permuted_tokens, sorted_indices = npu_moe_token_permute(tokens, indices)
        unpermuted_tokens = npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)

        if topk == 1:
            assert torch.equal(unpermuted_tokens, tokens)
        else:
            assert torch.allclose(unpermuted_tokens, tokens, **tols)
