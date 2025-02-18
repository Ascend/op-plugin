import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestAddmv(TestCase):
    def cpu_op_exec(self, a, b, c, alpha, beta):
        output = torch.addmv(c, a, b, alpha=alpha, beta=beta)
        output = output.numpy()
        return output

    def npu_op_exec(self, a, b, c, alpha, beta):
        output = torch.addmv(c, a, b, alpha=alpha, beta=beta)
        output = output.to("cpu")
        output = output.numpy()
        return output

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec_out(self, a, b, c, beta, alpha, input1):
        torch.addmv(c, a, b, alpha=alpha, beta=beta, out=input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_addmv_fp16(self):
        shape_format = [
            [[np.float16, 3, (2, 3)], [np.float16, 3, (3,)], [np.float16, 3, (2,)]]
        ]
        for item in shape_format:

            input_a, npu_input_a = create_common_tensor(item[0], -2, 2)
            input_b, npu_input_b = create_common_tensor(item[1], -2, 2)
            input_c, npu_input_c = create_common_tensor(item[2], -2, 2)

            input_a = input_a.to(torch.float32)
            input_b = input_b.to(torch.float32)
            input_c = input_c.to(torch.float32)

            cpu_output = self.cpu_op_exec(input_a, input_b, input_c, 1, 1)
            npu_output = self.npu_op_exec(npu_input_a, npu_input_b, npu_input_c, 1, 1)

            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_addmv_out_fp16(self):
        shape_format = [
            [
                [np.float16, 3, (2, 3)],
                [np.float16, 3, (3,)],
                [np.float16, 3, (2,)],
                [np.float16, 3, (10,)],
            ]
        ]
        for item in shape_format:

            input_a, npu_input_a = create_common_tensor(item[0], -2, 2)
            input_b, npu_input_b = create_common_tensor(item[1], -2, 2)
            input_c, npu_input_c = create_common_tensor(item[2], -2, 2)
            _, npu_input = create_common_tensor(item[3], -2, 2)

            input_a = input_a.to(torch.float32)
            input_b = input_b.to(torch.float32)
            input_c = input_c.to(torch.float32)

            cpu_output = self.cpu_op_exec(input_a, input_b, input_c, 1, 1)
            npu_output = self.npu_op_exec_out(
                npu_input_a, npu_input_b, npu_input_c, 1, 1, npu_input
            )
            cpu_output = cpu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_addmv_fp32(self):
        shape_format = [
            [[np.float16, 0, (2, 3)], [np.float16, 0, (3,)], [np.float16, 0, (2,)]],
            [
                [np.float16, 0, (3168, 320)],
                [np.float16, 0, (320,)],
                [np.float16, 0, (3168,)],
            ],
        ]
        for item in shape_format:

            input_a, npu_input_a = create_common_tensor(item[0], -2, 2)
            input_b, npu_input_b = create_common_tensor(item[1], -2, 2)
            input_c, npu_input_c = create_common_tensor(item[2], -2, 2)

            cpu_output = self.cpu_op_exec(
                input_a.float(), input_b.float(), input_c.float(), 1, 1
            )
            npu_output = self.npu_op_exec(
                npu_input_a.float(), npu_input_b.float(), npu_input_c.float(), 1, 1
            )

            self.assertRtolEqual(cpu_output, npu_output, prec=1.0e-3, prec16=1.0e-3)

    @SupportedDevices(['Ascend910B'])
    def test_addmv_fp16_to_fp32(self):
        shape_format = [
            [[np.float16, 0, (2, 3)], [np.float16, 0, (3,)], [np.float16, 0, (2,)]],
            [
                [np.float16, 0, (3168, 320)],
                [np.float16, 0, (320,)],
                [np.float16, 0, (3168,)],
            ],
        ]
        for item in shape_format:

            input_a, npu_input_a = create_common_tensor(item[0], -2, 2)
            input_b, npu_input_b = create_common_tensor(item[1], -2, 2)
            input_c, npu_input_c = create_common_tensor(item[2], -2, 2)

            cpu_output = self.cpu_op_exec(
                input_a.float(), input_b.float(), input_c, 1, 1
            )
            npu_output = self.npu_op_exec(
                npu_input_a.float(), npu_input_b.float(), npu_input_c, 1, 1
            )

            self.assertRtolEqual(cpu_output, npu_output, prec=1.0e-3, prec16=1.0e-3)

    @skipIfUnsupportMultiNPU(2)
    def test_addmv_device_check(self):
        npu_input_a = torch.randn(2, 3).npu()
        npu_input_b = torch.randn(3,).npu()
        npu_input_c = torch.randn(2,).to("npu:1")
        msg = "Expected all tensors to be on the same device, but found at least two devices,"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.addmv(npu_input_a, npu_input_b, npu_input_c)


if __name__ == "__main__":
    run_tests()
