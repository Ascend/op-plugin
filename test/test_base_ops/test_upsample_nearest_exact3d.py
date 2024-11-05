import torch
import torch_npu

from torch.nn.functional import interpolate
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestUpsampleNearestExact3d(TestCase):
    torch_dtypes = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "bfloat16" : torch.bfloat16
    }

    def create_tensor(self, dtype, shape):
        cpu_tensor = torch.randn(size=shape, dtype=self.torch_dtypes.get(dtype))
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
    def test_UpsampleNearestExact3d_common_shape_format(self):
        shape_format = [
            ["float32", (1, 1, 38, 86, 73), (10, 63, 32)],
            ["float16", (8, 3, 29, 45, 51), (65, 54, 86)],
            ["bfloat16", (9, 2, 23, 50, 7), (63, 90, 60)],
            ["float32", (9, 7, 32, 55, 38), (13, 7, 86)],
            ["float16", (3, 3, 36, 64, 30), (24, 80, 22)],
            ["bfloat16", (9, 2, 46, 5, 75), (27, 32, 16)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = self.create_tensor(item[0], item[1])
            if item[0] == "float16" or item[0] == "bfloat16":
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_inputs, item[2])
            npu_output = self.npu_op_exec(npu_inputs, item[2])
            cpu_output = cpu_output.to(npu_output.dtype)

            if item[0] == "bfloat16":
                self.assert_equal(cpu_output, npu_output)
            else:
                self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_UpsampleNearestExact3d_large_scale_format(self):
        shape_format = [
            ["float32", (4, 1, 1, 2, 1), (1, 268, 12)],
            ["float16", (5, 2, 16, 3, 2), (1, 578, 3)],
            ["bfloat16", (6, 6, 3, 4, 2), (7, 6, 367)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = self.create_tensor(item[0], item[1])
            if item[0] == "float16" or item[0] == "bfloat16":
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_inputs, item[2])
            npu_output = self.npu_op_exec(npu_inputs, item[2])
            cpu_output = cpu_output.to(npu_output.dtype)

            if item[0] == "bfloat16":
                self.assert_equal(cpu_output, npu_output)
            else:
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
