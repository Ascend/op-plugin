import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestMaxunpool2d(TestCase):
    def test_max_unpool2d(self):
        input1 = torch.tensor([[[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        pool2d = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        out, ind = pool2d(input1)
        unpool2d = torch.nn.MaxUnpool2d(2, stride=2)
        npu_out = unpool2d(out.npu(), ind.npu())
        cpu_out = unpool2d(out, ind)
        self.assertRtolEqual(cpu_out, npu_out.cpu())

    @skipIfUnsupportMultiNPU(2)
    def test_max_unpool2d_device_check(self):
        input1 = torch.tensor([[[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        pool2d = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        out, ind = pool2d(input1)
        unpool2d = torch.nn.MaxUnpool2d(2, stride=2)
        msg = "Expected all tensors to be on the same device, but found at least two devices,"
        with self.assertRaisesRegex(RuntimeError, msg):
            unpool2d(out.npu(), ind.to("npu:1"))


if __name__ == "__main__":
    run_tests()
