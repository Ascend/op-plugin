import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestNpuBroadcast(TestCase):
    def custom_op_exec(self, input1, shape):
        output = torch.broadcast_to(input1, shape)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, size):
        output = torch_npu.npu_broadcast(input1, size)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_npu_broadcast(self):
        input1 = [
            torch.tensor([1, 2, 3]).npu(),
            torch.tensor([[1], [2], [3]]).npu()
        ]
        for item in input1:
            custom_output = self.custom_op_exec(item, (3, 3))
            npu_output = self.npu_op_exec(item, (3, 3))
            self.assertRtolEqual(custom_output, npu_output)

    @skipIfUnsupportMultiNPU(2)
    def test_npu_broadcast_multinpu(self):
        dev0 = torch.device("npu:0")
        dev1 = torch.device("npu:1")

        size = 2**26

        a = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)
        b = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)

        to_backward_recipient = a * b
        s = to_backward_recipient.to(device="npu:0").sum()
        torch_npu.npu.synchronize(device=dev0)
        torch_npu.npu.synchronize(device=dev1)
        s.backward()
        self.assertTrue(a.grad.sum().item() == size)
        self.assertTrue(b.grad.sum().item() == size)


if __name__ == "__main__":
    run_tests()
