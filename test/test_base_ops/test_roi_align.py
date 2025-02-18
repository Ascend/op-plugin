import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestRoiAlign(TestCase):
    def test_roi_align_fp32(self):
        _input = torch.FloatTensor([[[[1, 2, 3, 4, 5, 6],
                                      [7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18],
                                      [19, 20, 21, 22, 23, 24],
                                      [25, 26, 27, 28, 29, 30],
                                      [31, 32, 33, 34, 35, 36]]]]).npu()
        rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
        expect_out = torch.tensor([[[[4.5000, 6.5000, 8.5000],
                                     [16.5000, 18.5000, 20.5000],
                                     [28.5000, 30.5000, 32.5000]]]], dtype=torch.float32)
        out = torch_npu.npu_roi_align(_input, rois, 0.25, 3, 3, 2, 0)
        self.assertRtolEqual(expect_out, out.cpu())
    
    @skipIfUnsupportMultiNPU(2)
    def test_roi_align_device_check(self):
        _input = torch.FloatTensor([[[[1, 2, 3, 4, 5, 6],
                                      [7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18],
                                      [19, 20, 21, 22, 23, 24],
                                      [25, 26, 27, 28, 29, 30],
                                      [31, 32, 33, 34, 35, 36]]]]).npu()
        rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).to("npu:1")
        msg = "Expected all tensors to be on the same device, but found at least two devices,"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch_npu.npu_roi_align(_input, rois, 0.25, 3, 3, 2, 0)


if __name__ == "__main__":
    run_tests()
