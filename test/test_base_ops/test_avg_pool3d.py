import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_utils import SupportedDevices


class TestAvgPool3D(TestCase):
    def create_tensor(self, dtype, shape):
        cpu_tensor = torch.randn(size=shape, dtype=dtype)
        npu_tensor = cpu_tensor.to("npu")
        return cpu_tensor, npu_tensor

    def assert_equal_bfloat16(self, cpu_outs, npu_outs):
        for cpu_out, npu_out in zip(cpu_outs, npu_outs):
            if (cpu_out.shape != npu_out.shape):
                self.fail("shape error")
            if (cpu_out.dtype != npu_out.dtype):
                self.fail("dtype error!")
            result = torch.allclose(cpu_out, npu_out.cpu(), rtol=0.01, atol=0.001)
            if not result:
                self.fail("result error!")
        return True

    def cpu_op_exec(self, kernel_size, stride, input1):
        m = torch.nn.AvgPool3d(kernel_size, stride)
        output_data = m(input1)
        return output_data

    def cpu_op_exec_fp16(self, kernel_size, stride, input1):
        m = torch.nn.AvgPool3d(kernel_size, stride)
        output_data = m(input1.float())
        return output_data.half()

    def npu_op_exec(self, kernel_size, stride, input1):
        m = torch.nn.AvgPool3d(kernel_size, stride).npu()
        output_data = m(input1)
        return output_data

    def test_avg_pool_3d_fp32(self):
        shape_format = [
            [[np.float32, -1, (20, 16, 50, 44, 31)], (3, 2, 2), (2, 1, 2)],
            [[np.float32, -1, (2, 1, 4, 4, 4)], 3, 2],
            [[np.float32, -1, (2, 1, 4, 4, 4)], 2, 2],
            [[np.float32, -1, (2, 4, 4, 4)], 2, 2],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            npu_output = self.npu_op_exec(item[1], item[2], npu_input1)
            cpu_output = self.cpu_op_exec(item[1], item[2], cpu_input1)
            self.assertRtolEqual(cpu_output, npu_output.cpu(), 1.0e-3)

    def test_avg_pool_3d_fp16(self):
        shape_format = [
            [[np.float16, -1, (20, 16, 50, 44, 31)], (3, 2, 2), (2, 1, 2)],
            [[np.float16, -1, (2, 1, 4, 4, 4)], 3, 2],
            [[np.float16, -1, (2, 1, 4, 4, 4)], 2, 2],
            [[np.float16, -1, (2, 4, 4, 4)], 2, 2],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            npu_output = self.npu_op_exec(item[1], item[2], npu_input1)
            cpu_output = self.cpu_op_exec_fp16(item[1], item[2], cpu_input1)
            self.assertRtolEqual(cpu_output, npu_output.cpu())

    @SupportedDevices(['Ascend910B'])
    def test_avg_pool_3d_bf16(self):
        shape_format = [
            [torch.bfloat16, (2, 256, 67, 64, 64), (3, 1, 1), (2, 1, 1)],
            [torch.bfloat16, (1, 256, 11, 64, 64), (2, 1, 1), (2, 1, 1)],
            [torch.bfloat16, (1, 128, 19, 128, 128), (2, 1, 1), (2, 1, 1)],
            [torch.bfloat16, (1, 256, 19, 60, 106), (2, 1, 1), (2, 1, 1)],
            [torch.bfloat16, (1, 512, 11, 30, 53), (2, 1, 1), (2, 1, 1)],
        ]
        for item in shape_format:
            cpu_input, npu_input = self.create_tensor(item[0], item[1])
            if item[0] == torch.bfloat16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(item[2], item[3], cpu_input)
            npu_output = self.npu_op_exec(item[2], item[3], npu_input)
            cpu_output = cpu_output.to(npu_output.dtype)

            if item[0] == torch.bfloat16:
                self.assert_equal_bfloat16(cpu_output, npu_output)
            else:
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
