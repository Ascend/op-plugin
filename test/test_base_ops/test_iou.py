import torch
import torch_npu
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

class TestIou(TestCase):
    @SupportedDevices(['Ascend910A', 'Ascend910B'])
    def test_iou_fp16(self):
        bboxes = torch.tensor([[0, 0, 10, 10],
                               [10, 10, 20, 20],
                               [32, 32, 38, 42]], dtype=torch.float16).to("npu")
        gtboxes = torch.tensor([[0, 0, 10, 20],
                               [0, 10, 10, 10],
                               [10, 10, 20, 20]], dtype=torch.float16).to("npu")
        expect_iof = torch.tensor([[0.4990, 0.0000, 0.0000],
                                   [0.0000, 0.0000, 0.0000],
                                   [0.0000, 0.9980, 0.0000]], dtype=torch.float16)
        output_iof = torch_npu.npu_iou(bboxes, gtboxes, 1)
        self.assertRtolEqual(expect_iof, output_iof.cpu())

        expect_iou = torch.tensor([[0.4985, 0.0000, 0.0000],
                                   [0.0000, 0.0000, 0.0000],
                                   [0.0000, 0.9961, 0.0000]], dtype=torch.float16)
        output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
        self.assertRtolEqual(expect_iou, output_iou.cpu())

    @SupportedDevices(['Ascend910A', 'Ascend910B'])
    def test_iou_fp16_pt(self):
        bboxs = torch.tensor([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]], dtype=torch.float16).npu()
        gtboxes = torch.tensor([[1, 2, 3, 4],
                                [5, 6, 7, 8]], dtype=torch.float16).npu()
        expect_iof = torch.tensor([[0.9902, 0.0000, 0.0000, 0.0000],
                                   [0.0000, 0.9902, 0.0000, 0.0000]], dtype=torch.float16)
        output_iof = torch_npu.npu_ptiou(bboxs, gtboxes, 1)
        self.assertRtolEqual(expect_iof, output_iof.cpu(), 1.e-3)

        expect_iou = torch.tensor([[0.9805, 0.0000, 0.0000, 0.0000],
                                   [0.0000, 0.9805, 0.0000, 0.0000]], dtype=torch.float16)
        output_iou = torch_npu.npu_ptiou(bboxs, gtboxes, 0)
        self.assertRtolEqual(expect_iou, output_iou.cpu(), 1.e-3)


    @SupportedDevices(['Ascend950'])
    def test_iou_fp16_950(self):
        bboxes = torch.tensor([[0, 0, 10, 10],
                               [10, 10, 20, 20],
                               [32, 32, 38, 42]], dtype=torch.float16).to("npu")
        gtboxes = torch.tensor([[0, 0, 10, 20],
                               [0, 0, 10, 10],
                               [10, 10, 20, 20]], dtype=torch.float16).to("npu")
        expect_iof = torch.tensor([[0.5005, 0.0005, 0.0000],
                                   [1.0000, 0.0000, 0.0000],
                                   [0.0000, 1.0000, 0.0000]], dtype=torch.float16)
        output_iof = torch_npu.npu_iou(bboxes, gtboxes, 1)
        self.assertRtolEqual(expect_iof, output_iof.cpu(), 0.001)

        expect_iou = torch.tensor([[0.5005, 0.0003, 0.0000],
                                   [1.0000, 0.0000, 0.0000],
                                   [0.0000, 1.0000, 0.0000]], dtype=torch.float16)
        output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
        self.assertRtolEqual(expect_iou, output_iou.cpu(), 0.001)


    @SupportedDevices(['Ascend950'])
    def test_iou_fp32_950(self):
        bboxes = torch.tensor([[0, 0, 10, 10],
                               [10, 10, 20, 20],
                               [32, 32, 38, 42]], dtype=torch.float32).to("npu")
        gtboxes = torch.tensor([[0, 0, 10, 20],
                               [0, 0, 10, 10],
                               [10, 10, 20, 20]], dtype=torch.float32).to("npu")
        output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
        self.assertEqual(output_iou.shape, [3, 3])


    @SupportedDevices(['Ascend950'])
    def test_iou_mode_iof(self):
        bboxes = torch.tensor([[0, 0, 10, 10],
                               [10, 10, 20, 20]], dtype=torch.float16).to("npu")
        gtboxes = torch.tensor([[0, 0, 10, 20]], dtype=torch.float16).to("npu")
        output = torch_npu.npu_iou(bboxes, gtboxes, 1)
        self.assertEqual(output.shape, [1, 2])


    @SupportedDevices(['Ascend950'])
    def test_iou_shape(self):
        n = 6
        m = 8
        bboxes = torch.randn(n, 4).npu()
        gtboxes = torch.randn(m, 4).npu()
        output = torch_npu.npu_iou(bboxes, gtboxes, 0)
        self.assertEqual(output.shape, [m, n])


    @SupportedDevices(['Ascend950'])
    def test_iou_large_shape(self):
        n = 200
        m = 100
        bboxes = torch.randn(n, 4, dtype=torch.float16).npu()
        gtboxes = torch.randn(m, 4, dtype=torch.float16).npu()
        output = torch_npu.npu_iou(bboxes, gtboxes, 0)
        self.assertEqual(output.shape, [m, n])


if __name__ == "__main__":
    run_tests()
