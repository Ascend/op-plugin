import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestNPUGeluBackward(TestCase):

    def get_golden(self, grad_tensor, input_self_tensor, approximate="none"):
        output = torch.ops.aten.gelu_backward(grad_tensor, input_self_tensor, approximate=approximate)
        return output
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_gelu_backward(self):
        shape = [100, 400]
        mode = "tanh"

        input_self_tensor = torch.rand(shape, dtype=torch.float16).npu()
        grad_tensor = torch.rand(shape, dtype=torch.float16).npu()
        torch.npu.synchronize()
        output = torch.ops.npu.npu_gelu_backward(grad_tensor, input_self_tensor, approximate=mode)
        torch.npu.synchronize()

        golden = self.get_golden(grad_tensor.cpu(), input_self_tensor.cpu(), mode)
        self.assertRtolEqual(output.type(torch.float16), golden.type(torch.float16))


if __name__ == "__main__":
    run_tests()
