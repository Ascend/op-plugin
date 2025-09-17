import os
import shutil
import unittest

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUSwigluQuantV2(TestCase):
    def golden_swiglu_quant_torch(
        self,
        x,
        smooth_scales,
        offsets,
        group_index,
        activate_left,
        quant_mode,
        group_list_type,
        dst_type
    ):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y = torch.nn.functional.silu(x1) * x2 if activate_left else x1 * torch.nn.functional.silu(x2)
        dst_type_scale = 127.0 if dst_type in (torch.int8, None) else 7.0

        if group_index is not None:
            begin_index = 0
            for i in range(group_index.shape[0]):
                end_index = group_index[i] if group_list_type == 0 else begin_index + group_index[i]
                y_slice = y[begin_index:end_index]
                scale_slice = smooth_scales[i]
                offset_slice = offsets[i] if offsets is not None else 0
                y[begin_index:end_index] = y_slice * scale_slice + (offset_slice if quant_mode == 0 else 0)
                begin_index = end_index
        else:
            y = y * smooth_scales + (offsets if quant_mode == 0 else 0)
        
        scale = None
        if quant_mode == 1:
            scale = dst_type_scale / torch.max(torch.abs(y), dim=1)[0]
            y = y * scale[:, None]
        y = torch.round(y)
        y = torch.clamp(y, -1 - dst_type_scale, dst_type_scale).to(torch.int8)

        return y, scale

    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_npu_swiglu_quant(self, device="npu"):
        batch_size = 4608
        hidden_size = 2048
        x_shape = (batch_size, hidden_size)
        input_data = np.random.randn(*x_shape).astype(np.float32)

        quant_mode = 1
        group_list_type = 0
        dst_type = torch.int8
        activate_left = False
        num_groups = 8
        offsets = np.random.randn(num_groups, hidden_size // 2).astype(np.float32)
        group_size = batch_size // num_groups
        group_index = [(i + 1) * group_size for i in range(num_groups)]
        smooth_scales = np.random.randn(num_groups, hidden_size // 2).astype(np.float32)

        device = "npu"
        npu_x = torch.tensor(input_data, dtype=torch.float32, device=device)
        npu_group_index = torch.tensor(group_index, dtype=torch.int32, device=device)
        npu_smooth_scales = torch.tensor(smooth_scales, dtype=torch.float32, device=device)
        npu_offsets = torch.tensor(offsets, dtype=torch.float32, device=device)
        result = torch_npu.npu_swiglu_quant(
            npu_x,
            smooth_scales=npu_smooth_scales,
            offsets=npu_offsets,
            group_index=npu_group_index,
            activate_left=activate_left,
            quant_mode=quant_mode,
            group_list_type=group_list_type,
            dst_type=dst_type
        )

        device = "cpu"
        cpu_x = torch.tensor(input_data, dtype=torch.float32, device=device)
        cpu_group_index = torch.tensor(group_index, dtype=torch.int32, device=device)
        cpu_smooth_scales = torch.tensor(smooth_scales, dtype=torch.float32, device=device)
        cpu_out = self.golden_swiglu_quant_torch(
            cpu_x,
            smooth_scales=cpu_smooth_scales,
            offsets=offsets,
            group_index=cpu_group_index,
            activate_left=activate_left,
            quant_mode=quant_mode,
            group_list_type=group_list_type,
            dst_type=dst_type
        )

        self.assertRtolEqual(cpu_out[0].numpy(), result[0].cpu().numpy())
        if quant_mode == 1:
            self.assertRtolEqual(cpu_out[1].numpy(), result[1].cpu().numpy())

if __name__ == "__main__":
    run_tests()
