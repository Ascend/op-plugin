import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuMoeInitRoutingQuant(TestCase):

    def cpu_op_exec(self, x, expert_idx, scale, offset, expert_range,
                    quant_mode, drop_mode):
        expert_start = expert_range[0]
        expert_end = expert_range[1]
        num_rows = x.shape[0]
        h = x.shape[1]
        k = expert_idx.shape[-1]
        expert_idx_in = expert_idx.copy().reshape(-1)
        actual_expert_total_num = np.sum((expert_idx >= expert_start)
                                         & (expert_idx < expert_end))

        # sorting
        expert_idx_in[(expert_idx_in < expert_start)] = np.int32(
            np.iinfo(np.int32).max)
        sorted_expert_indices = np.argsort(expert_idx_in,
                                           axis=-1,
                                           kind="stable")
        sorted_expert_idx = expert_idx_in[sorted_expert_indices]
        if drop_mode == 1:
            expanded_row_idx = sorted_expert_indices.astype(np.int32)
        else:
            expanded_row_idx = np.ones(num_rows * k).astype(np.int32) * -1
            tmp_indices = np.arange(actual_expert_total_num)
            expanded_row_idx[
                sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

        # static quant
        if quant_mode == 0:
            expanded_scale = None
            x_fp16 = x.astype(np.float16)
            scale_fp16 = scale.astype(np.float16)
            if scale_fp16.ndim == 1:
                scale_fp16 = scale_fp16[:, np.newaxis]
            expanded_x = x_fp16[
                sorted_expert_indices[:actual_expert_total_num] // k, :] * scale_fp16[0]
            if offset is not None:
                offset_fp16 = offset.astype(np.float16)
                if offset_fp16.ndim == 1:
                    offset_fp16 = offset_fp16[:, np.newaxis]
                expanded_x = expanded_x + offset_fp16[0]
            expanded_x = np.rint(expanded_x)
            expanded_x = np.clip(expanded_x, -128, 127)
            expanded_x = expanded_x.astype(np.int8)
            expanded_x = np.concatenate([
                expanded_x,
                np.zeros((num_rows * k - actual_expert_total_num, h)).astype(np.int8)
            ], axis=0)

        # dynamic quant
        if quant_mode == 1:
            expanded_x = x[sorted_expert_indices // k, :]
            expanded_x = expanded_x.astype(np.float32)
            if scale is None:
                expanded_x = expanded_x[:actual_expert_total_num, :]
                x_abs = np.abs(expanded_x)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = expanded_x / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)
            else:
                expanded_scale = scale[sorted_expert_idx[:actual_expert_total_num] - expert_start, :]
                expanded_x = expanded_x[:actual_expert_total_num, :]
                expanded_x = expanded_x * expanded_scale
                x_abs = np.abs(expanded_x)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = expanded_x / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)
            expanded_x = np.concatenate([
                expanded_x,
                np.zeros((num_rows * k - actual_expert_total_num, h)).astype(np.int8)
            ], axis=0)
            expanded_scale = np.concatenate([
                np.squeeze(expanded_scale),
                np.zeros((num_rows * k - actual_expert_total_num)).astype(np.float32)
            ])

        # histogram
        expert_tokens_count = np.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start)
        expert_tokens_count = np.concatenate([
            expert_tokens_count.astype(np.int32),
            np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int32)
        ])

        return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale

    def npu_op_exec(self, x, expert_idx, scale, offset, active_expert_range,
                    quant_mode, drop_mode):
        expert_num = active_expert_range[-1]
        expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expert_tokens_before_capacity, expanded_scale = torch_npu.npu_moe_init_routing_quant(
            x,
            expert_idx.to(torch.int32),
            scale=scale,
            offset=offset,
            active_num=0,
            expert_capacity=0,
            expert_num=expert_num,
            drop_pad_mode=drop_mode,
            expert_tokens_num_mode=2,
            expert_tokens_before_capacity_flag=False,
            quant_mode=quant_mode)
        return expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale

    def assertExpandedXRtolEqual(self, expanded_x, local_expanded_x_npu,
                                 dtype):
        if dtype == torch.bfloat16:
            self.assertRtolEqual(
                torch.tensor(expanded_x, dtype=torch.bfloat16),
                local_expanded_x_npu)
        elif dtype == np.int8:
            self.assertEqual(expanded_x.shape, local_expanded_x_npu.shape)
            self.assertEqual(np.int8, local_expanded_x_npu.numpy().dtype)
            max_diff = np.abs(expanded_x - local_expanded_x_npu.numpy()).max()
            self.assertLessEqual(max_diff, 1)
        else:
            self.assertRtolEqual(expanded_x, local_expanded_x_npu.numpy())

    def generate_inputs(self, bs, h, k, dtype, scale_shape, none_scale,
                        none_offset):
        if dtype == torch.bfloat16:
            x = np.random.uniform(-1, 1, size=(bs, h)).astype(np.float32)
            x_npu = torch.tensor(x, dtype=torch.bfloat16).npu()
        elif dtype == np.int8:
            x = np.random.uniform(-127, 128, size=(bs, h)).astype(dtype)
            x_npu = torch.from_numpy(x).npu()
        else:
            x = np.random.uniform(-1, 1, size=(bs, h)).astype(dtype)
            x_npu = torch.from_numpy(x).npu()
        expert_idx = np.random.randint(0, 32, size=(bs, k)).astype(np.int32)
        scale = None if none_scale else np.random.uniform(
            -1, 1, size=scale_shape).astype(np.float32)
        offset = None if none_offset or none_scale else np.random.uniform(
            -1, 1, size=scale_shape).astype(np.float32)
        expert_idx_npu = torch.from_numpy(expert_idx).npu()
        scale_npu = None if scale is None else torch.from_numpy(
            scale).contiguous().npu()[:1]
        offset_npu = None if offset is None else torch.from_numpy(
            offset).contiguous().npu()[:1]
        return x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu

    def calc_npu_vs_golden(self, x, expert_idx, scale, offset, x_npu,
                           expert_idx_npu, scale_npu, offset_npu, expert_range,
                           quant_mode, drop_mode):
        expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_npu, expanded_scale_npu = self.npu_op_exec(
            x_npu,
            expert_idx_npu,
            scale=scale_npu,
            offset=offset_npu,
            active_expert_range=expert_range,
            quant_mode=quant_mode,
            drop_mode=drop_mode)
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = self.cpu_op_exec(
            x,
            expert_idx,
            scale=scale,
            offset=offset,
            expert_range=expert_range,
            quant_mode=quant_mode,
            drop_mode=drop_mode)

        local_expanded_x_npu = expanded_x_npu.cpu()
        local_expanded_row_idx_npu = expanded_row_idx_npu.cpu()
        local_expert_tokens_count_npu = expert_tokens_count_npu.cpu()
        actual_expert_count = np.sum(local_expert_tokens_count_npu.numpy())
        if expanded_scale_npu is not None:
            local_expanded_scale_npu = expanded_scale_npu.cpu()
            local_expanded_scale_npu[actual_expert_count:] = 0
        else:
            local_expanded_scale_npu = None
        local_expanded_x_npu[actual_expert_count:] = 0

        return expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
            expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_static_quant(self):
        bs_list = [4]
        h_list = [1024]
        k_list = [8]
        expert_range_list = [[0, 32]]
        quant_mode_list = [0]  # static quant
        drop_mode_list = [0]
        dtype_list = [np.float16, np.float32, torch.bfloat16]
        none_scales = [False]
        none_offsets = [False]
        for bs, h, k, expert_range, quant_mode, drop_mode, dtype, none_scale, none_offset in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list,
                drop_mode_list, dtype_list, none_scales, none_offsets):
            scale_shape = (1, )
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu = self.generate_inputs(
                bs, h, k, dtype, scale_shape, none_scale, none_offset)

            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, _, _ \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                        x_npu, expert_idx_npu, scale_npu, offset_npu,
                                        expert_range, quant_mode, drop_mode)
            self.assertExpandedXRtolEqual(expanded_x, local_expanded_x_npu,
                                          np.int8)
            self.assertRtolEqual(expanded_row_idx,
                                 local_expanded_row_idx_npu.numpy())
            self.assertRtolEqual(expert_tokens_count,
                                 local_expert_tokens_count_npu.numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_dynamic_quant(self):
        bs_list = [4]
        h_list = [1024]
        k_list = [8]
        expert_range_list = [[0, 32]]
        quant_mode_list = [1]  # dynamic quant
        drop_mode_list = [0]
        dtype_list = [np.float16, np.float32, torch.bfloat16]
        none_scales = [True]
        none_offsets = [True]
        for bs, h, k, expert_range, quant_mode, drop_mode, dtype, none_scale, none_offset in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list,
                drop_mode_list, dtype_list, none_scales, none_offsets):
            expert_range_length = expert_range[1] - expert_range[0]
            scale_shape = (expert_range_length, h)
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu = self.generate_inputs(
                bs, h, k, dtype, scale_shape, none_scale, none_offset)

            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                          x_npu, expert_idx_npu, scale_npu, offset_npu,
                                          expert_range, quant_mode, drop_mode)
            self.assertExpandedXRtolEqual(expanded_x, local_expanded_x_npu,
                                          np.int8)
            self.assertRtolEqual(expanded_row_idx,
                                 local_expanded_row_idx_npu.numpy())
            self.assertRtolEqual(expanded_scale,
                                 local_expanded_scale_npu.numpy())
            self.assertRtolEqual(expert_tokens_count,
                                 local_expert_tokens_count_npu.numpy())


if __name__ == "__main__":
    run_tests()
