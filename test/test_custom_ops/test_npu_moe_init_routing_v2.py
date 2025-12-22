import unittest
import itertools
from os import scandir
import numpy as np
import random
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class MoeInitRoutingV2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, expert_idx, *, scale=None, offset=None,
                active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0,
                expert_tokens_num_type=0, expert_tokens_num_flag=False,
                quant_mode=0, active_expert_range=0, row_idx_type=0):
        ddd = dict(x=x, expert_idx=expert_idx, scale=scale, offset=offset,
                   active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
                   drop_pad_mode=drop_pad_mode,
                   expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                   quant_mode=quant_mode, active_expert_range=active_expert_range, row_idx_type=row_idx_type)
        for k, v in ddd.items():
            print(k, v)
        expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = \
            torch.ops.npu.npu_moe_init_routing_v2(
                x, expert_idx, scale=scale, offset=offset,
                active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
                drop_pad_mode=drop_pad_mode,
                expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type
            )
        return expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale


class TestNpuMoeInitRoutingV2(TestCase):

    def adapter_capacity(self, sorted_row_idx, sorted_expert_idx, capacity):
        count = 0
        last = sorted_expert_idx[0]
        for i, val in enumerate(sorted_expert_idx):
            if last != val:
                count = 1
                last = val
            else:
                count += 1
                if count > capacity:
                    sorted_expert_idx[i] = -1
                    sorted_row_idx[i] = -1

    def cpu_op_exec(self, x, expert_idx, scale, offset, expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity):
        expert_start = expert_range[0]
        expert_end = expert_range[1]
        expert_num = 32
        num_rows = x.shape[0]
        h = x.shape[1]
        k = expert_idx.shape[-1]
        expert_idx_in = expert_idx.copy().reshape(-1)
        actual_expert_total_num = np.sum((expert_idx >= expert_start) & (expert_idx < expert_end))

        expert_idx_in[(expert_idx_in < expert_start)] = np.int32(np.iinfo(np.int32).max)
        sorted_expert_indices = np.argsort(expert_idx_in, axis=-1, kind="stable")
        sorted_expert_idx = expert_idx_in[sorted_expert_indices]
        if row_idx_type == 1:
            expanded_row_idx = sorted_expert_indices.astype(np.int32)
        else:
            expanded_row_idx = np.ones(num_rows * k).astype(np.int32) * -1
            tmp_indices = np.arange(actual_expert_total_num)
            expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

        if not expert_tokens_num_flag:
            expert_tokens_count = None
        else:
            if drop_pad_mode == 0:
                if expert_tokens_num_type == 1:
                    expert_tokens_count = np.bincount(
                        sorted_expert_idx[:actual_expert_total_num] - expert_start)
                    expert_tokens_count = np.concatenate(
                        [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                elif expert_tokens_num_type == 0:
                    expert_tokens_count = np.bincount(
                        sorted_expert_idx[:actual_expert_total_num] - expert_start)
                    expert_tokens_count = np.concatenate(
                        [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                    expert_tokens_count = np.cumsum(expert_tokens_count)
                elif expert_tokens_num_type == 2:
                    expert_id, counts = np.unique(sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                    expert_tokens_count = np.column_stack((expert_id, counts))
                    if expert_tokens_count.shape[0] < expert_num:
                        expert_tokens_count = np.concatenate((expert_tokens_count, [[0,0],]), axis=0)
            else:
                expert_tokens_count = np.bincount(
                        sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate(
                    [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
            expert_tokens_count = expert_tokens_count.astype(np.int64)
        
        if drop_pad_mode == 0:
            if active_num == 0:
                active_num = actual_expert_total_num
            else:
                active_num = min(active_num, actual_expert_total_num)
            expanded_scale = None
            expanded_x = x[sorted_expert_indices[:active_num] // k, :]
            if scale is not None and quant_mode == -1:
                expanded_scale = scale[sorted_expert_indices[:active_num] // k]
        else:
            self.adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)

            sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
            offset_tmp = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_expert_indices):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset_tmp = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset_tmp] = sorted_expert_indices[i]
                    offset_tmp = offset_tmp + 1
            
            expanded_row_idx = np.full(sorted_expert_indices.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            expanded_x_mask = np.full((expert_num * expert_capacity, h), 1, dtype=int)
            expanded_x = np.full((expert_num * expert_capacity, h), 0, dtype=x.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = x[val // k]
                    expanded_x_mask[i] = np.full((h,), 0, dtype=int)

        if quant_mode == -1:
            expanded_x = expanded_x
            expanded_row_idx = expanded_row_idx
            if scale is not None and drop_pad_mode == 1:
                expanded_scale = np.full((expert_num * expert_capacity,), 0, dtype=scale.dtype)
                for i, val in enumerate(sort_row_tmp):
                    if val != -1:
                        expanded_scale[i] = scale[val // k]
            if scale is None:
                expanded_scale = None

        if quant_mode == 0:
            expanded_scale = None
            expanded_x_fp16 = expanded_x.astype(np.float16)
            scale_val = scale.astype(np.float16)
            offset_val = offset.astype(np.float16)
            scale_rst = expanded_x_fp16 * scale_val[0]
            add_offset = scale_rst + offset_val[0]
            round_data = np.rint(add_offset)
            round_data = np.clip(round_data, -128, 127)
            expanded_x = round_data.astype(np.int8)

        if quant_mode == 1:
            x_final = expanded_x.astype(np.float32)
            if scale is None:
                x_abs = np.abs(x_final)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)
            else:
                if scale.shape[0] == 1:
                    x_final = x_final * scale
                else:
                    if drop_pad_mode == 0:
                        x_final = x_final * scale[sorted_expert_idx[:active_num] - expert_start]
                    else:
                        for i, val in enumerate(sort_row_tmp):
                            if val != -1:
                                x_final[i] = x_final[i] * scale[i // expert_capacity]
                x_abs = np.abs(x_final)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)
            if x.dtype == np.int8:
                expanded_scale == None
        if drop_pad_mode == 1:
            expanded_x = np.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

        return expanded_x, expanded_row_idx.astype(np.int32), expert_tokens_count, expanded_scale

    def npu_op_exec(self, x, expert_idx, scale, offset, active_expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity):
        bs = x.shape[0]
        k = expert_idx.shape[1]
        expert_num = 32
        expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = torch_npu.npu_moe_init_routing_v2(
            x, expert_idx, scale=scale, offset=offset,
            active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
            drop_pad_mode=drop_pad_mode,
            expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
            active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type
        )
        return expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale

    def assertExpandedXRtolEqual(self, expanded_x, local_expanded_x_npu, dtype):
        if dtype == torch.bfloat16:
            self.assertRtolEqual(torch.tensor(expanded_x, dtype=torch.bfloat16), local_expanded_x_npu)
        elif dtype == np.int8:
            self.assertEqual(expanded_x.shape, local_expanded_x_npu.shape)
            self.assertEqual(np.int8, local_expanded_x_npu.numpy().dtype)
            max_diff = np.abs(expanded_x - local_expanded_x_npu.numpy()).max()
            self.assertLessEqual(max_diff, 1)
        else:
            self.assertRtolEqual(expanded_x, local_expanded_x_npu.numpy())

    def generate_inputs(self, bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode):
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
        scale = None if none_scale else np.random.uniform(-1, 1, size=scale_shape).astype(np.float32)
        offset = None if none_offset or none_scale else np.random.uniform(-1, 1, size=scale_shape).astype(np.float32)

        expert_idx_npu = torch.from_numpy(expert_idx).npu()
        scale_npu = None if scale is None else torch.from_numpy(scale).contiguous().npu()
        offset_npu = None if offset is None else torch.from_numpy(offset).contiguous().npu()

        expert_tokens_num_type = 1 if drop_pad_mode == 1 else random.choice([0, 1, 2])
        row_idx_type = 0 if drop_pad_mode == 1 else random.choice([0, 1])
        # active_num = 0 if drop_pad_mode == 1 else random.randint(0, bs * k)
        active_num = bs * k
        expert_capacity = -1 if drop_pad_mode == 0 else random.randint(1, bs)

        return x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, expert_tokens_num_type, row_idx_type, active_num, expert_capacity

    def calc_npu_vs_golden(self, x, expert_idx, scale, offset,
                           x_npu, expert_idx_npu, scale_npu, offset_npu,
                           expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity):
        expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_npu, expanded_scale_npu = self.npu_op_exec(
            x_npu, expert_idx_npu, scale=scale_npu, offset=offset_npu,
            active_expert_range=expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type, expert_tokens_num_flag=expert_tokens_num_flag, expert_tokens_num_type=expert_tokens_num_type,
            drop_pad_mode=drop_pad_mode, active_num=active_num, expert_capacity=expert_capacity)
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = self.cpu_op_exec(
            x, expert_idx, scale=scale, offset=offset,
            expert_range=expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type, expert_tokens_num_flag=expert_tokens_num_flag, expert_tokens_num_type=expert_tokens_num_type,
            drop_pad_mode=drop_pad_mode, active_num=active_num, expert_capacity=expert_capacity)

        local_expanded_x_npu = expanded_x_npu.cpu()
        local_expanded_row_idx_npu = expanded_row_idx_npu.cpu()
        local_expert_tokens_count_npu = expert_tokens_count_npu.cpu()
        local_expanded_scale_npu = expanded_scale_npu.cpu()

        actual_expert_total_num = np.sum((expert_idx >= expert_range[0]) & (expert_idx < expert_range[1]))
        if drop_pad_mode == 0:
            if active_num == 0:
                actual_expert_count = actual_expert_total_num
            else:
                actual_expert_count = min(active_num, actual_expert_total_num)
            local_expanded_x_npu = local_expanded_x_npu[:actual_expert_count]
            local_expanded_scale_npu = local_expanded_scale_npu[:actual_expert_count]

        if expert_tokens_num_flag == True and expert_tokens_num_type == 2:
            length = expert_tokens_count.shape[0]
            local_expert_tokens_count_npu = local_expert_tokens_count_npu[:length]

        return expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
            expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_no_quant(self):
        bs_list = [32]
        h_list = [14, 200]
        k_list = [5]
        expert_range_list = [[0, 16]]
        quant_mode_list = [-1]
        drop_pad_mode_list = [0, 1]
        row_idx_type_list = [0, 1]
        expert_tokens_num_flags = [True, False]
        dtype_list = [np.int8, np.float16, np.float32, torch.bfloat16]
        none_scales = [True, False]
        none_offsets = [True]
        for bs, h, k, expert_range, quant_mode, row_idx_type, dtype, none_scale, none_offset, expert_tokens_num_flag, drop_pad_mode in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list, row_idx_type_list,
                dtype_list, none_scales, none_offsets, expert_tokens_num_flags, drop_pad_mode_list):
            scale_shape = (bs,)
            expert_range = [0, 32] if drop_pad_mode == 1 else [0, 16]
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = self.generate_inputs(
                bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode)
            if  drop_pad_mode == 1 or expert_tokens_num_flag == False or expert_tokens_num_type == 0:
                continue
            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                          x_npu, expert_idx_npu, scale_npu, offset_npu,
                                          expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity)

            self.assertExpandedXRtolEqual(expanded_x, local_expanded_x_npu, dtype)
            self.assertRtolEqual(expanded_row_idx, local_expanded_row_idx_npu.numpy())
            if expert_tokens_num_flag:
                self.assertRtolEqual(expert_tokens_count, local_expert_tokens_count_npu.numpy())
            if none_scale:
                return
            self.assertRtolEqual(expanded_scale, local_expanded_scale_npu.numpy())

    @unittest.skip("Skipping test_npu_moe_init_routing_static_quant for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_static_quant(self):
        bs_list = [32]
        h_list = [14, 200]
        k_list = [5, 128]
        expert_range_list = [[0, 16]]
        quant_mode_list = [0]
        drop_pad_mode_list = [0, 1]
        row_idx_type_list = [0, 1]
        expert_tokens_num_flags = [True, False]
        dtype_list = [np.float16, np.float32, torch.bfloat16]
        none_scales = [False]
        none_offsets = [False]
        for bs, h, k, expert_range, quant_mode, row_idx_type, dtype, none_scale, none_offset, expert_tokens_num_flag, drop_pad_mode in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list, row_idx_type_list,
                dtype_list, none_scales, none_offsets, expert_tokens_num_flags, drop_pad_mode_list):
            scale_shape = (1,)
            expert_range = [0, 32] if drop_pad_mode == 1 else [0, 16]
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = self.generate_inputs(
                bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode)

            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                          x_npu, expert_idx_npu, scale_npu, offset_npu,
                                          expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity)

            self.assertExpandedXRtolEqual(expanded_x, local_expanded_x_npu, np.int8)
            self.assertRtolEqual(expanded_row_idx, local_expanded_row_idx_npu.numpy())
            if expert_tokens_num_flag:
                self.assertRtolEqual(expert_tokens_count, local_expert_tokens_count_npu.numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_dynamic_quant(self):
        bs_list = [32]
        h_list = [14, 200]
        k_list = [8]
        expert_range_list = [[0, 16]]
        quant_mode_list = [1]
        drop_pad_mode_list = [0, 1]
        row_idx_type_list = [0, 1]
        expert_tokens_num_flags = [True, False]
        dtype_list = [np.float16, np.float32, torch.bfloat16]
        none_scales = [True, False]
        none_offsets = [True]
        for bs, h, k, expert_range, quant_mode, row_idx_type, dtype, none_scale, none_offset, expert_tokens_num_flag, drop_pad_mode in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list, row_idx_type_list,
                dtype_list, none_scales, none_offsets, expert_tokens_num_flags, drop_pad_mode_list):
            expert_range_length = expert_range[1] - expert_range[0]
            scale_shape = (expert_range_length, h)
            expert_range = [0, 32] if drop_pad_mode == 1 else [0, 16]
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = self.generate_inputs(
                bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode)
            if  drop_pad_mode == 1 or expert_tokens_num_flag == False or expert_tokens_num_type == 0:
                continue
            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                          x_npu, expert_idx_npu, scale_npu, offset_npu,
                                          expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity)

            self.assertExpandedXRtolEqual(expanded_x, local_expanded_x_npu, np.int8)
            self.assertRtolEqual(expanded_row_idx, local_expanded_row_idx_npu.numpy())
            if expert_tokens_num_flag:
                self.assertRtolEqual(expert_tokens_count, local_expert_tokens_count_npu.numpy())
            if none_scale:
                return
            self.assertRtolEqual(expanded_scale, local_expanded_scale_npu.numpy())


if __name__ == "__main__":
    run_tests()