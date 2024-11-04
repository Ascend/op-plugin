import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestSwiGlu(TestCase):
    def get_golden(self, input_self_tensor, dim):
        def swiglu_v1(x):
            """0.1版本，FP32格式运算，最后输出转成BF16"""
            x = torch.chunk(x, 2, dim=dim)
            self_tensor = x[0].type(torch.float32)
            other = x[1].type(torch.float32)
            output = F.silu(self_tensor.npu()) * other.npu()
            return output.type(torch.bfloat16)

        output = swiglu_v1(input_self_tensor)
        return output

    @SupportedDevices(['Ascend910B'])
    def test_swiglu(self):
        shape = [8192, 1, 3904 * 2]
        dim = -1

        input_self_tensor = torch.rand(shape, device='cpu', dtype=torch.bfloat16).npu()
        torch.npu.synchronize()
        output = torch_npu.npu_swiglu(input_self_tensor, dim)
        torch.npu.synchronize()

        golden = self.get_golden(input_self_tensor, dim)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32))

if __name__ == "__main__":
    run_tests()