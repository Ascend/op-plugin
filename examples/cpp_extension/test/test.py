import torch
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import op_extension


class TestCustomAdd(TestCase):

    def test_add_custom_ops(self):
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)

        x_npu = x.npu()
        y_npu = y.npu()
        output = torch.ops.npu.my_add(x_npu, y_npu)
        cpuout = torch.add(x, y)

        self.assertRtolEqual(output, cpuout)

    def test_matmul_leakyrelu_custom_ops(self):
        a = torch.rand([1024, 256], dtype=torch.float16).npu()
        b = torch.rand([256, 640], dtype=torch.float16).npu()
        bias = torch.randn([640], dtype=torch.float32).npu()

        output = torch.ops.npu.my_matmul_leakyrelu(a, b, bias)
        m = nn.LeakyReLU(0.001)
        cpuout = m(torch.matmul(a.cpu().type(output.dtype), b.cpu().type(output.dtype)) + bias.cpu())

        self.assertRtolEqual(output, cpuout)

if __name__ == "__main__":
    run_tests()
