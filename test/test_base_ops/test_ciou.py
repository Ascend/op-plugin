import math
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

class TestNpuCiou(TestCase):
    def generate_giou_data(self, n, m, dtype):
        data_bboxes = 20 * np.random.rand(4, n).astype(dtype)
        data_gtboxes = 20 * np.random.rand(4, n).astype(dtype)

        cpu_input1 = torch.from_numpy(data_bboxes)
        cpu_input2 = torch.from_numpy(data_gtboxes)

        npu_input1 = cpu_input1.npu()
        npu_input2 = cpu_input2.npu()

        list1 = [cpu_input1, cpu_input2, npu_input1, npu_input2]
        return list1

    def cpu_op_exec(self, bboxes, gtboxes, trans=True, is_cross=False, mode="iou"):
        b1_x1, b1_x2 = bboxes[0] - bboxes[2] / 2, bboxes[0] + bboxes[2] / 2
        b1_y1, b1_y2 = bboxes[1] - bboxes[3] / 2, bboxes[1] + bboxes[3] / 2
        b2_x1, b2_x2 = gtboxes[0] - gtboxes[2] / 2, gtboxes[0] + gtboxes[2] / 2
        b2_y1, b2_y2 = gtboxes[1] - gtboxes[3] / 2, gtboxes[1] + gtboxes[3] / 2

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        eps = 1e-9
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        try:
            iou = inter / union
        except ZeroDivisionError:
            print("union is 0, raise ZeroDivisionError.")

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        try:
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            alpha = v / ((1 + eps) - iou + v)
            res_numpy = iou - rho2 / c2 - v * alpha
        except ZeroDivisionError:
            print("union is 0, raise ZeroDivisionError.")
        res_numpy = np.expand_dims(res_numpy, 0)
        v = np.expand_dims(v, 0)
        return res_numpy

    def npu_op_exec(self, box1, box2, trans=True, is_cross=False, mode=0):
        overlap = torch_npu.npu_ciou(box1, box2, trans, is_cross, mode, True)
        overlap = overlap.to("cpu")
        overlap = overlap.numpy()
        return overlap

    @SupportedDevices(['Ascend910A', 'Ascend910B'])
    def test_npu_ciou_shape_format(self):
        shape_list = [
            [6, 6],
            [12, 12],
            [100, 100]
        ]
        is_trans_list = [True]
        mode_list = ["iou"]
        dtype = np.float32
        # pylint:disable=complicate-comprehension
        shape_format = [[j, k, m]
                        for j in shape_list
                        for k in is_trans_list
                        for m in mode_list]
        for item in shape_format:
            mode_digit = 0 if item[-1] == "iou" else 1
            is_cross = False
            list1 = self.generate_giou_data(*item[0], dtype)
            cpu_overlap = self.cpu_op_exec(list1[0], list1[1], item[1], is_cross, item[-1])
            overlap = self.npu_op_exec(list1[2], list1[3], item[1], is_cross, mode_digit)
            self.assertRtolEqual(cpu_overlap, overlap)


    @SupportedDevices(['Ascend950'])
    def test_npu_ciou_shape_format_1024(self):
        shape_list = [
            [1024, 1024],
            [2048, 2048],
            [3072, 3072]
        ]
        is_trans_list = [True]
        mode_list = ["iou"]
        dtype = np.float32
        # pylint:disable=complicate-comprehension
        shape_format = [[j, k, m]
                        for j in shape_list
                        for k in is_trans_list
                        for m in mode_list]
        for item in shape_format:
            mode_digit = 0 if item[-1] == "iou" else 1
            is_cross = False
            list1 = self.generate_giou_data(*item[0], dtype)
            cpu_overlap = self.cpu_op_exec(list1[0], list1[1], item[1], is_cross, item[-1])
            overlap = self.npu_op_exec(list1[2], list1[3], item[1], is_cross, mode_digit)
            self.assertRtolEqual(cpu_overlap, overlap)

    @SupportedDevices(['Ascend950'])
    def test_npu_ciou_fp16(self):
        list1 = self.generate_giou_data(1024, 1024, np.float16)
        overlap = self.npu_op_exec(list1[2], list1[3], trans=True, is_cross=False, mode=0)
        self.assertEqual(overlap.shape[0], 1)
        self.assertEqual(overlap.shape[1], 1024)

    @SupportedDevices(['Ascend950'])
    def test_npu_ciou_mode_iof(self):
        n = 1024
        list1 = self.generate_giou_data(n, n, np.float32)
        overlap = self.npu_op_exec(list1[2], list1[3], trans=True, is_cross=False, mode=1)
        self.assertEqual(overlap.shape[0], 1)
        self.assertEqual(overlap.shape[1], n)

    @SupportedDevices(['Ascend950'])
    def test_npu_ciou_atan_sub_flag_false(self):
        n = 1024
        list1 = self.generate_giou_data(n, n, np.float32)
        overlap = torch_npu.npu_ciou(list1[2], list1[3], trans=True, is_cross=False, mode=0, atan_sub_flag=False)
        self.assertEqual(overlap.shape, [1, n])

    @SupportedDevices(['Ascend950'])
    def test_npu_ciou_mixed_precision(self):
        n = 1024
        data_bboxes = 20 * np.random.rand(4, n).astype(np.float16)
        data_gtboxes = 20 * np.random.rand(4, n).astype(np.float32)
        npu_bboxes = torch.from_numpy(data_bboxes).npu()
        npu_gtboxes = torch.from_numpy(data_gtboxes).npu()
        overlap = torch_npu.npu_ciou(npu_bboxes, npu_gtboxes, trans=True, is_cross=False, mode=0, atan_sub_flag=True)
        self.assertEqual(overlap.shape, [1, n])

    @SupportedDevices(['Ascend950'])
    def test_npu_ciou_invalid_shape(self):
        n = 100
        list1 = self.generate_giou_data(n, n, np.float32)
        self.assertRaises(RuntimeError, torch_npu.npu_ciou, list1[2], list1[3], True, False, 0, True)


if __name__ == "__main__":
    run_tests()
