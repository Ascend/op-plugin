import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantBatchdMatmulInplaceAdd(TestCase):

    def single_matmul(self, y, x1, x2, x1_scale, x2_scale):
        x1_scale = np.repeat(x1_scale, 32, axis=1)
        x2_scale = np.repeat(x2_scale, 32, axis=1)
        x1 = x1 * x1_scale
        x2 = x2 * x2_scale
        output = np.matmul(x1, x2)
        output = output + y
        return output

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_inplace_add(self):
        torch.manual_seed(0)
        M = 16
        K = 256
        N = 32
        x1 = torch.randint(2, 3, size=(M, K), dtype=torch.float8_e4m3fn)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.float8_e4m3fn)

        x1_clone = x1.clone().npu()
        x2_clone = x2.clone().npu()

        y = torch.randint(2, 3, size=(M, N), dtype=torch.float32)
        y1_clone = y.clone().npu()
        y2_clone = y.clone().npu()
        x2_scale = np.random.uniform(low=-5, high=5, size=(M, math.ceil(k/64), 2)).astype(np.int8)
        x1_scale = np.random.uniform(low=-5, high=5, size=(math.ceil(k/64), N, 2)).astype(np.int8)
        x2_scale_clone = x2_scale.clone.npu()
        x1_scale_clone = x1_scale.clone.npu()
        x1_dtype = torch_npu.float8_e8m0
        x2_dtype = torch_npu.float8_e8m0
        supported_output = self.single_matmul(y, x1, x2, x1_scale, x2_scale)
        custom_output1 = torch_npu.npu_add_quant_matmul_(y1, x1, x2, x2_scale, x1_scale=x1_scale,
                                                      group_sizes=None, x1_dtype=x1_dtype,
                                                      x2_dtype=x2_dtype)

        custom_output2 = torch_npu.npu_add_quant_matmul(y2, x1, x2, x2_scale, x1_scale=x1_scale,
                                                     group_sizes=None, x1_dtype=x1_dtype,
                                                     x2_dtype=x2_dtype)
        self.assertRtolEqual(supported_output, custom_output1.cpu().numpy(), 0.001)
        self.assertRtolEqual(supported_output, custom_output2.cpu().numpy(), 0.001)

if __name__ == "__main__":
    run_tests()