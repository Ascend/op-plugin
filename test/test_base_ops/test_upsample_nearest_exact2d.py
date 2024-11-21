import torch
import torch_npu

from torch.nn.functional import interpolate
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestUpsampleNearestExact2d(TestCase):
    torch_dtypes = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "bfloat16" : torch.bfloat16,
        "float64" : torch.float64,
        "uint8" : torch.uint8
    }

    def create_tensor(self, dtype, shape):
        cpu_tensor = torch.randint(0, 100, size=shape, dtype=self.torch_dtypes.get(dtype))
        npu_tensor = cpu_tensor.to("npu")
        return cpu_tensor, npu_tensor

    def assert_equal(self, cpu_out, npu_out):
        if (cpu_out.shape != npu_out.shape):
            self.fail("shape error")
        if (cpu_out.dtype != npu_out.dtype):
            self.fail("dtype error!")
        result = torch.allclose(cpu_out, npu_out.cpu(), rtol=0.001, atol=0.001)
        if not result:
            self.fail("result error!")
        return True

    def cpu_op_exec(self, inputs, shapes):
        output = interpolate(inputs, size=shapes, mode="nearest-exact")
        return output

    def npu_op_exec(self, inputs, shapes):
        output = interpolate(inputs, size=shapes, mode="nearest-exact")
        return output
    
    @SupportedDevices(['Ascend910B'])
    def test_UpsampleNearestExact2d_common_shape_format(self):
        shape_format = [
            ["float32", (1, 1, 64, 64), (44, 44)],
            ["float16", (1, 16, 32, 32), (55, 55)],
            ["bfloat16", (1, 32, 16, 16), (66, 66)],
            ["float32", (8, 8, 16, 16), (48, 48)],
            ["float16", (16, 8, 16, 16), (48, 48)],
            ["bfloat16", (8, 8, 16, 16), (48, 48)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = self.create_tensor(item[0], item[1])
            if item[0] == "float16" or item[0] == "bfloat16":
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_inputs, item[2])
            npu_output = self.npu_op_exec(npu_inputs, item[2])
            cpu_output = cpu_output.to(npu_output.dtype)

            self.assert_equal(cpu_output, npu_output)

if __name__ == "__main__":
    run_tests()
