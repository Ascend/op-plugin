import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestBatchGatherMatmul(TestCase):
    def npu_batch_gather_matmul(self, y, x, weightB, indices, weightA, y_offset=0, y_slice_size=128, layer_idx=0,
                                scale=2):
        zeros_wa_4d = torch.zeros((1, weightA.shape[1], weightA.shape[2], weightA.shape[3]), dtype=torch.float16).npu()
        zeros_wb_4d = torch.zeros((1, weightB.shape[1], weightB.shape[2], weightB.shape[3]), dtype=torch.float16).npu()
        WA = torch.cat((weightA, zeros_wa_4d), dim=0)[indices, layer_idx, :, :].transpose(-1, -2)
        WB = torch.cat((weightB, zeros_wb_4d), dim=0)[indices, layer_idx, :, :].transpose(-1, -2)
        Z1 = torch.bmm(x.unsqueeze(1), WA)
        Z2 = torch.bmm(Z1, WB).squeeze() * scale
        return y + Z2

    @SupportedDevices(["Ascend310P"])
    def test_batch_gather_matmul(self):
        torch.manual_seed(12)
        y = torch.randn(1, 128).half().npu()
        x = torch.randn(1, 16).half().npu()
        weightA = torch.randn(2, 1, 16, 16).half().npu()
        indices = torch.randint(0, 1, (1,)).to(torch.int32).npu()
        weightB = torch.randn(2, 1, 128, 16).half().npu()

        output_npu = self.npu_batch_gather_matmul(y, x, weightB, indices, weightA, y_offset=0, y_slice_size=128,
                                                  layer_idx=0, scale=2)
        torch_npu.npu_batch_gather_matmul(y,
                                          x,
                                          weightB,
                                          indices,
                                          weightA,
                                          y_offset=0, y_slice_size=128, layer_idx=0, scale=2)

        self.assertEqual(y.cpu(), output_npu)

    @SupportedDevices(["Ascend310P"])
    def test_batch_gather_matmul_(self):
        torch.manual_seed(12)
        y = torch.randn(10, 128).half().npu()
        x = torch.randn(10, 128).half().npu()
        weightA = torch.randn(2, 1, 16, 128).half().npu()
        indices = torch.randint(0, 2, (10,)).to(torch.int32).npu()
        weightB = torch.randn(2, 1, 128, 16).half().npu()

        output_npu = self.npu_batch_gather_matmul(y, x, weightB, indices, weightA, y_offset=0, y_slice_size=128,
                                                  layer_idx=0, scale=2)
        y_out = torch_npu.npu_batch_gather_matmul_(y, x, weightB, indices, weightA,
                                                   y_offset=0, y_slice_size=128, layer_idx=0, scale=2)

        self.assertEqual(y_out.cpu(), output_npu)


if __name__ == "__main__":
    run_tests()
