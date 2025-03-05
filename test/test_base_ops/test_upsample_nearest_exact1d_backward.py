import torch
import torch_npu

from torch.nn.functional import interpolate
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestUpsampleNearestExact1dBackward(TestCase):
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
        result = torch.allclose(cpu_out, npu_out.cpu(), rtol=0.01, atol=0.01)
        if not result:
            self.fail("result error!")
        return True

    def cpu_op_exec(self, inputs, shapes):
        inputs.requires_grad_(True)
        output = interpolate(inputs, size=shapes, mode="nearest-exact")
        output.backward(torch.ones_like(output))
        gradcpu = inputs.grad
        return gradcpu

    def npu_op_exec(self, inputs, shapes):
        inputs.requires_grad_(True)
        output = interpolate(inputs, size=shapes, mode="nearest-exact")
        output.backward(torch.ones_like(output))
        grad = inputs.grad
        return grad
    
    @SupportedDevices(['Ascend910B'])
    def test_UpsampleNearestExact1dBackward_common_shape_format(self):
        shape_format = [
            ["float32", (8, 2, 2), (100)],
            ["float32", (4, 3, 1), (2)],
            ["float16", (1, 4, 2), (8)],
            ["bfloat16", (8, 8, 8), (12)],
            ["float32", (2, 3, 2), (3)],
            ["float16", (4, 10, 16), (15)],
            ["bfloat16", (10, 4, 3), (2)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = self.create_tensor(item[0], item[1])
            if item[0] == "float16" or item[0] == "bfloat16":
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_grad = self.cpu_op_exec(cpu_inputs, item[2])
            npu_grad = self.npu_op_exec(npu_inputs, item[2])
            cpu_grad = cpu_grad.to(npu_grad.dtype)

            if item[0] == "bfloat16":
                self.assert_equal(cpu_grad, npu_grad)
            else:
                self.assertRtolEqual(cpu_grad, npu_grad)

    @SupportedDevices(['Ascend910B'])
    def test_UpsampleNearestExact1dBackward_large_scale_format(self):
        shape_format = [
            ["float32", (4, 1, 20), (368)],
            ["float16", (5, 2, 16), (512)],
            ["bfloat16", (6, 6, 30), (208)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = self.create_tensor(item[0], item[1])
            if item[0] == "float16" or item[0] == "bfloat16":
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_grad = self.cpu_op_exec(cpu_inputs, item[2])
            npu_grad = self.npu_op_exec(npu_inputs, item[2])
            cpu_grad = cpu_grad.to(npu_grad.dtype)

            if item[0] == "bfloat16":
                self.assert_equal(cpu_grad, npu_grad)
            else:
                self.assertRtolEqual(cpu_grad, npu_grad)

    @SupportedDevices(['Ascend910B'])
    def test_UpsampleNearestExact1dBackward_large_shape_format(self):
        shape_format = [
            ["float32", (2, 2, 3441), (9234)],
            ["float16", (7, 6, 14777), (12768)],
            ["bfloat16", (1, 5, 5663), (8779)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = self.create_tensor(item[0], item[1])
            if item[0] == "float16" or item[0] == "bfloat16":
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_grad = self.cpu_op_exec(cpu_inputs, item[2])
            npu_grad = self.npu_op_exec(npu_inputs, item[2])
            cpu_grad = cpu_grad.to(npu_grad.dtype)

            if item[0] == "bfloat16":
                self.assert_equal(cpu_grad, npu_grad)
            else:
                self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
