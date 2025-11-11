import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestNPUGelu(TestCase):

    def get_golden(self, input_tensor, approximate = "none"):
        last_dim = input_tensor.shape[-1]
        if last_dim % 2 == 1:
            return "shape error"
        d = last_dim // 2
        x1 = input_tensor[..., :d]
        x2 = input_tensor[..., d:]
        m = torch.nn.GELU(approximate)
        x1 = m(x1)
        output = x1 * x2
        return output

    @SupportedDevices(['Ascend910B'])
    def test_npu_gelu_all_modes(self):
        shape = [100, 400]
        test_combinations = [
            ("none", torch.float16),
            ("none", torch.float32),
            ("tanh", torch.float16),
            ("tanh", torch.float32)
        ]

        for mode, dtype in test_combinations:
            input_tensor = torch.rand(shape, dtype=dtype).npu()
            output = torch_npu.npu_gelu_mul(input_tensor, approximate=mode)
            golden = self.get_golden(input_tensor.cpu(), mode)
            self.assertRtolEqual(golden, output.cpu())

if __name__ == "__main__":
    run_tests()
