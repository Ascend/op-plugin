import unittest
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuFusedCausalConv1d(TestCase):
    def get_golden_fn(self, x, weight, conv_states, query_start_loc, cache_indices,
                   initial_state_mode, residual_connection=0, pad_slot_id=-1):
        x_fp32 = x.float()
        weight_fp32 = weight.float()
        conv_states_fp32 = conv_states.float()
        dtype = x.dtype

        cu_seq_len, dim = x_fp32.shape
        batch_size = query_start_loc.shape[0] - 1
        kernel_width = weight_fp32.size(0)
        state_len = kernel_width - 1

        out = torch.zeros(cu_seq_len, dim, dtype=torch.float32)

        for batch_idx in range(batch_size):
            slot_idx = cache_indices[batch_idx].item()
            if slot_idx == pad_slot_id:
                continue

            start_idx = query_start_loc[batch_idx].item()
            end_idx = query_start_loc[batch_idx + 1].item()
            seq_x = x_fp32[start_idx:end_idx]

            has_state = initial_state_mode[batch_idx].item()

            if has_state == 1:
                cached_state = conv_states_fp32[slot_idx].clone()
            else:
                cached_state = torch.zeros(state_len, dim, dtype=torch.float32)

            padded_input = torch.cat([cached_state, seq_x], dim=0)

            result = F.conv1d(
                padded_input.transpose(0, 1).unsqueeze(0),
                weight_fp32.transpose(0, 1).unsqueeze(1),
                bias=None, stride=1, padding=0, groups=dim,
            ).squeeze(0).transpose(0, 1)

            if has_state == 2:
                result[:state_len] = 0

            if residual_connection == 1:
                out[start_idx:end_idx] = result + x_fp32[start_idx:end_idx]
            else:
                out[start_idx:end_idx] = result

            conv_states_fp32[slot_idx] = padded_input[-state_len:]

        return out.to(dtype), conv_states_fp32.to(dtype)

    def get_golden_update(self, x, weight, conv_state, conv_state_indices,
                   query_start_loc=None, num_accepted_tokens=None,
                   residual_connection=0, pad_slot_id=-1):
        x_fp32 = x.float()
        weight_fp32 = weight.float()
        conv_state_fp32 = conv_state.float()
        dtype = x.dtype

        if x_fp32.ndim == 3:
            batch, seq_len, dim = x_fp32.shape
        else:
            batch = query_start_loc.size(0) - 1
            dim = x_fp32.size(1)

        width = weight_fp32.size(0)
        state_len = conv_state_fp32.size(1)

        out = torch.ones_like(x_fp32)

        for batch_idx in range(batch):
            if conv_state_indices[batch_idx] == pad_slot_id:
                continue

            if x_fp32.ndim == 2:
                start_idx = query_start_loc[batch_idx].item()
                end_idx = query_start_loc[batch_idx + 1].item()
                x_seq = x_fp32[start_idx:end_idx]
            else:
                x_seq = x_fp32[batch_idx]

            state_idx = conv_state_indices[batch_idx]
            current_state = conv_state_fp32[state_idx]

            if num_accepted_tokens is not None:
                offset = num_accepted_tokens[batch_idx].item() - 1
            else:
                offset = 0

            padded_input = torch.cat(
                (current_state[offset: offset + width - 1], x_seq), dim=0)

            result = F.conv1d(
                padded_input.transpose(0, 1).unsqueeze(0),
                weight_fp32.transpose(0, 1).unsqueeze(1),
                bias=None, stride=1, padding=0, groups=dim,
            ).squeeze(0).transpose(0, 1)

            if x_fp32.ndim == 2:
                if residual_connection == 1:
                    out[start_idx:end_idx] = result + x_fp32[start_idx:end_idx]
                else:
                    out[start_idx:end_idx] = result
            else:
                if residual_connection == 1:
                    out[batch_idx] = result + x_fp32[batch_idx]
                else:
                    out[batch_idx] = result

            if (padded_input.size(0) - 1) == state_len:
                conv_state_fp32[state_idx][:state_len] = padded_input[1:]
            elif (padded_input.size(0) - 1) < state_len:
                conv_state_fp32[state_idx][:padded_input.size(0)] = padded_input

        return out.to(dtype), conv_state_fp32.to(dtype)

    @unittest.skip("Skip test_npu_fused_causal_conv1d now")
    @SupportedDevices(['Ascend950'])
    def test_npu_fused_causal_conv1d_update_3d(self):
        batch = 4
        dim = 128
        kernel_width = 3
        state_len = kernel_width - 1
        m_num = 2
        seq_len = m_num + 1
        dtype = torch.float16

        num_slots = batch + 1

        x = torch.randn(batch, seq_len, dim, dtype=dtype)
        weight = torch.randn(kernel_width, dim, dtype=dtype)
        conv_state = torch.randn(num_slots, state_len + m_num, dim, dtype=dtype)

        cache_indices = torch.tensor(list(range(batch)), dtype=torch.int32)
        num_accepted_tokens = torch.tensor([1, 2, 1, 3], dtype=torch.int32)
        residual_connection = 0

        golden_out, golden_states = self.get_golden_update(
            x, weight, conv_state.clone(), cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            residual_connection=residual_connection,
        )

        conv_state_npu = conv_state.clone().npu()
        out_npu = torch_npu.npu_fused_causal_conv1d(
            x.npu(), weight.npu(), conv_state_npu,
            cache_indices=cache_indices.npu(),
            num_accepted_tokens=num_accepted_tokens.npu(),
            activation_mode="None",
            run_mode=1,
            residual_connection=residual_connection,
            pad_slot_id=-1,
        )
        torch.npu.synchronize()

        pad_slot_id = -1
        valid_batch_mask = (cache_indices != pad_slot_id).cpu()
        out_valid_mask = torch.zeros(batch, seq_len, dim, dtype=torch.bool)
        for batch_idx in range(batch):
            if valid_batch_mask[batch_idx].item():
                out_valid_mask[batch_idx, :, :] = True

        self.assertRtolEqual(out_npu.cpu()[out_valid_mask], golden_out[out_valid_mask])
        self.assertRtolEqual(conv_state_npu.cpu(), golden_states)
    
    @unittest.skip("Skip test_npu_fused_causal_conv1d now")
    @SupportedDevices(['Ascend950'])
    def test_npu_fused_causal_conv1d_fn_float16(self):
        batch = 4
        dim = 128
        kernel_width = 3
        state_len = kernel_width - 1
        dtype = torch.float16

        seq_lens = [5, 3, 7, 4]
        cu_seq_len = sum(seq_lens)
        num_slots = batch * 2

        x = torch.randn(cu_seq_len, dim, dtype=dtype)
        weight = torch.randn(kernel_width, dim, dtype=dtype)
        conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype)

        starts = [0]
        for sl in seq_lens:
            starts.append(starts[-1] + sl)
        query_start_loc = torch.tensor(starts, dtype=torch.int32)
        cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32)
        initial_state_mode = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
        residual_connection = 0

        golden_out, golden_states = self.get_golden_fn(
            x, weight, conv_states.clone(), query_start_loc, cache_indices,
            initial_state_mode, residual_connection=residual_connection,
        )

        conv_states_npu = conv_states.clone().npu()
        out_npu = torch_npu.npu_fused_causal_conv1d(
            x.npu(), weight.npu(), conv_states_npu,
            query_start_loc=query_start_loc.npu(),
            cache_indices=cache_indices.npu(),
            initial_state_mode=initial_state_mode.npu(),
            activation_mode="None",
            run_mode=0,
            residual_connection=residual_connection,
            pad_slot_id=-1,
        )
        torch.npu.synchronize()

        self.assertRtolEqual(out_npu.cpu(), golden_out)
        self.assertRtolEqual(conv_states_npu.cpu(), golden_states)


if __name__ == "__main__":
    run_tests()
