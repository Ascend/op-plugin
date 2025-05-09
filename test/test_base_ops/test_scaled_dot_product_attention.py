import torch
import numpy as np
import torch.nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_utils import SupportedDevices


class TestScaledDotProductAttention(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_sdpa_fia(self):
        shape_format1 = [
            [[np.float16, 0, (1, 3, 10, 32)], [np.float16, 0, (1, 3, 10, 32)], [np.float16, 0, (1, 3, 10, 32)]],
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], -1, 1)
            cpu_output = torch.nn.functional.scaled_dot_product_attention(cpu_input1.to(torch.float32), cpu_input2.to(torch.float32), cpu_input3.to(torch.float32))
            npu_output = torch.nn.functional.scaled_dot_product_attention(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output.to(torch.float16), npu_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_sdpa_attn_mask_dim_3(self):
        shape_format1 = [
            [[np.float16, 0, (1, 3, 32, 32)], [np.float16, 0, (1, 3, 32, 32)], [np.float16, 0, (1, 3, 32, 32)], [np.float16, 0, (1, 32, 32)]],
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -1, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], -1, 1)
            cpu_input4, npu_input4 = create_common_tensor(item[3], -1, 1)

            cpu_output = torch.nn.functional.scaled_dot_product_attention(cpu_input1.to(torch.float32), cpu_input2.to(torch.float32), cpu_input3.to(torch.float32),  attn_mask=cpu_input4.bool())
            npu_output = torch.nn.functional.scaled_dot_product_attention(npu_input1, npu_input2, npu_input3,  attn_mask=npu_input4.bool())
            self.assertRtolEqual(cpu_output.to(torch.float16), npu_output, 0.001)

if __name__ == "__main__":
    run_tests()
