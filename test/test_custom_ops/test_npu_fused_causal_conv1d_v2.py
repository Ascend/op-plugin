import unittest
import torch
import torch.nn.functional as F
import torch_npu
from typing import Optional
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def causal_conv1d_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    B_size: int = 0,
    conv_mode: int = 0,
    inplace: bool = False,
    residual: bool = False,
) -> tuple:
    """Golden function adapted from golden.py for PTA test use."""
    if x.ndim == 3:
        flattened = True
        bsz, seq_len_3d, dim = x.shape
        x = x.view(-1, dim)
        if query_start_loc is None:
            query_start_loc = torch.arange(
                start=0, end=(bsz + 1) * seq_len_3d, step=seq_len_3d,
                dtype=torch.int32, device=x.device)
    else:
        flattened = False

    cu_seq_len, dim = x.shape
    batch_size = query_start_loc.shape[0] - 1

    width = weight.size(0)
    assert conv_states.size(1) >= width - 1

    apc_enabled = block_idx_last_scheduled_token is not None

    out = torch.ones_like(x)

    for batch_idx in range(batch_size):
        start_idx = query_start_loc[batch_idx].item()
        end_idx = query_start_loc[batch_idx + 1].item()
        seq_len = end_idx - start_idx
        seq_x = x[start_idx:end_idx]

        if apc_enabled:
            seq_completed_offset_token = num_computed_tokens[batch_idx].item() % B_size
            seq_completed_offset = B_size - seq_completed_offset_token
            seq_end_offset = (seq_len - seq_completed_offset) % B_size
            last_full_block_token_index = seq_len - seq_end_offset
            if seq_end_offset == 0:
                last_full_block_token_index -= B_size
            idx_first = block_idx_first_scheduled_token[batch_idx].item()
            idx_last = block_idx_last_scheduled_token[batch_idx].item()
            n_block_to_fill = idx_last - idx_first

            assert cache_indices is not None and cache_indices.ndim == 2
            read_cache_line = cache_indices[batch_idx, initial_state_idx[batch_idx]].item()
            write_cache_line = cache_indices[batch_idx, idx_last].item()
        else:
            if cache_indices is not None:
                read_cache_line = cache_indices[batch_idx].item()
                write_cache_line = cache_indices[batch_idx].item()
            else:
                read_cache_line = batch_idx
                write_cache_line = batch_idx

        if read_cache_line == pad_slot_id:
            continue

        if num_computed_tokens is not None and num_computed_tokens[batch_idx] == 0:
            cached_state = torch.zeros((width - 1, dim), device=x.device, dtype=x.dtype)
            offset = 0
        else:
            if num_accepted_tokens is not None:
                accepted_tokens = num_accepted_tokens[batch_idx].item()
                assert 1 <= accepted_tokens <= seq_len
                offset = accepted_tokens - 1
            else:
                offset = conv_states.size(1) - (width - 1)
            cached_state = conv_states[read_cache_line][:offset + width - 1]

        padded_input = torch.cat([cached_state, seq_x], dim=0)

        cache_len = min(conv_states.size(1), padded_input.size(0))
        conv_states[write_cache_line][-cache_len:] = padded_input[-cache_len:]

        padded_input = padded_input[offset:]

        if apc_enabled:
            for chunk in range(n_block_to_fill):
                boundary_idx = last_full_block_token_index - (n_block_to_fill - chunk - 1) * B_size
                assert boundary_idx > 0
                wc = cache_indices[batch_idx, idx_first + chunk]
                conv_states[wc][-(width - 1):] = padded_input[boundary_idx: boundary_idx + width - 1]

        result = F.conv1d(
            padded_input.transpose(0, 1).unsqueeze(0),
            weight.transpose(0, 1).unsqueeze(1),
            bias=None, stride=1, padding=0, groups=dim
        ).squeeze(0).transpose(0, 1)

        if conv_mode == 1:
            assert num_computed_tokens is not None
            last_reset_idx = width - 1 - num_computed_tokens[batch_idx].item()
            last_reset_idx = min(max(last_reset_idx, 0), seq_len)
            result[:last_reset_idx] = 0

        out[start_idx:end_idx] = result + seq_x if residual else result
        if inplace:
            x[start_idx:end_idx] = out[start_idx:end_idx]

    if inplace:
        return x if not flattened else x.view(bsz, -1, dim), conv_states
    return out if not flattened else out.view(bsz, -1, dim), conv_states


class TestNpuFusedCausalConv1dV2(TestCase):

    @unittest.skip("Skip test_npu_fused_causal_conv1d_v2 now")
    @SupportedDevices(['Ascend950'])
    def test_npu_fused_causal_conv1d_v2_decode(self):
        batch, dim, kernel_width = 4, 128, 3
        seq_len, m_num = 3, 2
        state_len = kernel_width - 1 + m_num
        dtype = torch.float16

        x = torch.randn(batch, seq_len, dim, dtype=dtype)
        weight = torch.randn(kernel_width, dim, dtype=dtype)
        conv_states = torch.randn(batch, state_len, dim, dtype=dtype)
        cache_indices = torch.arange(batch, dtype=torch.int32)
        num_accepted_tokens = torch.tensor([1, 2, 1, 3], dtype=torch.int32)
        num_computed_tokens = torch.tensor([5, 3, 7, 4], dtype=torch.int32)

        golden_x, golden_states = causal_conv1d_golden(
            x.clone().float(), weight.float(), conv_states.clone().float(),
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            conv_mode=1, inplace=True, residual=True,
        )

        x_npu = x.clone().npu()
        conv_states_npu = conv_states.clone().npu()

        torch_npu.npu_fused_causal_conv1d_v2(
            x_npu, weight.npu(), conv_states_npu,
            cache_indices=cache_indices.npu(),
            num_accepted_tokens=num_accepted_tokens.npu(),
            residual_connection=1, pad_slot_id=-1,
            num_computed_tokens=num_computed_tokens.npu(),
            conv_mode="pangu",
        )
        torch.npu.synchronize()

        self.assertRtolEqual(x_npu.cpu(), golden_x.to(dtype))
        self.assertRtolEqual(conv_states_npu.cpu(), golden_states.to(dtype))

    @unittest.skip("Skip test_npu_fused_causal_conv1d_v2 now")
    @SupportedDevices(['Ascend950'])
    def test_npu_fused_causal_conv1d_v2_prefill_2d(self):
        batch, dim, kernel_width = 4, 128, 3
        state_len = kernel_width - 1
        dtype = torch.bfloat16

        seq_lens = [5, 3, 7, 4]
        cu_seq_len = sum(seq_lens)

        x = torch.randn(cu_seq_len, dim, dtype=dtype)
        weight = torch.randn(kernel_width, dim, dtype=dtype)
        conv_states = torch.randn(8, state_len, dim, dtype=dtype)

        starts = [0]
        for sl in seq_lens:
            starts.append(starts[-1] + sl)
        query_start_loc = torch.tensor(starts, dtype=torch.int32)
        cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32)
        num_computed_tokens = torch.zeros(batch, dtype=torch.int32)

        golden_x, golden_states = causal_conv1d_golden(
            x.clone().float(), weight.float(), conv_states.clone().float(),
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            num_computed_tokens=num_computed_tokens,
            conv_mode=1, inplace=True, residual=True,
        )

        x_npu = x.clone().npu()
        conv_states_npu = conv_states.clone().npu()

        torch_npu.npu_fused_causal_conv1d_v2(
            x_npu, weight.npu(), conv_states_npu,
            query_start_loc=query_start_loc.npu(),
            cache_indices=cache_indices.npu(),
            residual_connection=1, pad_slot_id=-1,
            num_computed_tokens=num_computed_tokens.npu(),
            conv_mode="pangu",
        )
        torch.npu.synchronize()

        self.assertRtolEqual(x_npu.cpu(), golden_x.to(dtype))
        self.assertRtolEqual(conv_states_npu.cpu(), golden_states.to(dtype))

    @unittest.skip("Skip test_npu_fused_causal_conv1d_v2 now")
    @SupportedDevices(['Ascend950'])
    def test_npu_fused_causal_conv1d_v2_apc(self):
        batch, dim, kernel_width = 4, 128, 3
        dtype, block_size = torch.bfloat16, 128
        seq_lens = [5, 3, 7, 4]
        cu_seq_len = sum(seq_lens)
        max_query_len = max(seq_lens)
        max_num_blocks = (max_query_len + block_size - 1) // block_size + 1

        x = torch.randn(cu_seq_len, dim, dtype=dtype)
        weight = torch.randn(kernel_width, dim, dtype=dtype)
        conv_states = torch.randn(batch * max_num_blocks, kernel_width - 1, dim, dtype=dtype)

        starts = [0]
        for sl in seq_lens:
            starts.append(starts[-1] + sl)
        query_start_loc = torch.tensor(starts, dtype=torch.int32)

        cache_indices = torch.zeros(batch, max_num_blocks, dtype=torch.int32)
        for i in range(batch):
            for j in range(max_num_blocks):
                cache_indices[i][j] = i * max_num_blocks + j

        block_idx_first = torch.zeros(batch, dtype=torch.int32)
        block_idx_last = torch.tensor(
            [(sl - 1) // block_size for sl in seq_lens], dtype=torch.int32)
        initial_state_idx = torch.zeros(batch, dtype=torch.int32)
        num_computed_tokens = torch.zeros(batch, dtype=torch.int32)

        golden_x, golden_states = causal_conv1d_golden(
            x.clone().float(), weight.float(), conv_states.clone().float(),
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=block_idx_first,
            block_idx_last_scheduled_token=block_idx_last,
            initial_state_idx=initial_state_idx,
            B_size=block_size, conv_mode=1,
            inplace=True, residual=True,
        )

        x_npu = x.clone().npu()
        conv_states_npu = conv_states.clone().npu()

        torch_npu.npu_fused_causal_conv1d_v2(
            x_npu, weight.npu(), conv_states_npu,
            query_start_loc=query_start_loc.npu(),
            cache_indices=cache_indices.npu(),
            residual_connection=1, pad_slot_id=-1,
            max_query_len=max_query_len,
            num_computed_tokens=num_computed_tokens.npu(),
            block_idx_first_scheduled_token=block_idx_first.npu(),
            block_idx_last_scheduled_token=block_idx_last.npu(),
            initial_state_idx=initial_state_idx.npu(),
            block_size=block_size, conv_mode="pangu",
        )
        torch.npu.synchronize()

        self.assertRtolEqual(x_npu.cpu(), golden_x.to(dtype))
        self.assertRtolEqual(conv_states_npu.cpu(), golden_states.to(dtype))


if __name__ == "__main__":
    run_tests()
