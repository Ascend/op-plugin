import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class DataInfo(object):
    def __init__(self, min_d, max_d, shape_x, shape_scale, shape_zp, dtype_x, dtype_scale, dtype_zp):
        self.min_d = min_d
        self.max_d = max_d
        self.shape_x = shape_x
        self.shape_scale = shape_scale
        self.shape_zp = shape_zp
        self.dtype_x = dtype_x
        self.dtype_scale = dtype_scale
        self.dtype_zp = dtype_zp


class TestNPUQuantize(TestCase):

    def generate_data_npu_quantize(self, datainfo):
        input_x = np.random.uniform(datainfo.min_d, datainfo.max_d, datainfo.shape_x).astype(datainfo.dtype_x)
        scales = np.random.uniform(datainfo.min_d, datainfo.max_d, datainfo.shape_scale).astype(datainfo.dtype_scale)
        zero_points = np.random.uniform(datainfo.min_d, datainfo.max_d, datainfo.shape_zp).astype(datainfo.dtype_zp)
        npu_input_x = torch.from_numpy(input_x)
        npu_input_scales = torch.from_numpy(scales)
        npu_input_zero_points = torch.from_numpy(zero_points)
        return npu_input_x, npu_input_scales, npu_input_zero_points

    def cpu_op_exec_per_channel(self, input_x, input_scales, input_zero_points, axis, dtype):
        output = torch.quantize_per_channel(input_x, input_scales, input_zero_points, axis, dtype).int_repr()
        output = output.numpy()
        return output
    
    def cpu_op_exec_ascend_quant_v2(self, input_x, input_scales, input_zero_points, axis, dtype):
        input_x = input_x.astype("float32")
        input_scales = input_scales.astype("float32")
        input_zero_points = input_zero_points.astype("float32")

        add_offset = input_x * input_scales + input_zero_points
        round_data = np.round(add_offset, 0)
        output = np.clip(round_data, -128, 127).astype("int8")
        return output
    
    def npu_op_exec_ascend_quant_v2(self, input_x, input_scales, input_zero_points, axis, dtype):
        input_x = input_x.to("npu")
        input_scales = input_scales.to("npu")
        input_zero_points = input_zero_points.to("npu")
        output = torch_npu.npu_quantize(input_x, input_scales, input_zero_points, dtype, axis, div_mode=False)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_per_channel(self, input_x, input_scales, input_zero_points, axis, dtype):
        input_x = input_x.to("npu")
        input_scales = input_scales.to("npu")
        input_zero_points = input_zero_points.to("npu")
        output = torch_npu.npu_quantize(input_x, input_scales, input_zero_points, dtype, axis)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_npu_quantize_3_3_0_int32(self, device="npu"):
        datainfo = DataInfo(-1, 1, (3, 3), (3,), (3,), np.float32, np.float32, np.int32)
        input_x1, scales, zero_points = self.generate_data_npu_quantize(datainfo)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 0, torch.qint32)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 0, torch.qint32)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_npu_quantize_3_3_3_3_1_int8(self, device="npu"):
        datainfo = DataInfo(-1, 1, (3, 3), (3,), (3,), np.float32, np.float32, np.int8)
        input_x1, scales, zero_points = self.generate_data_npu_quantize(datainfo)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 1, torch.qint8).astype(np.int32)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 1, torch.qint8).astype(np.int32)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_npu_quantize_3_3_3_3_3_3_3_3_4_uint8(self, device="npu"):
        datainfo = DataInfo(-1, 1, (3, 3, 3, 3, 3, 3, 3, 3), (3,), (3,), np.float32, np.float32, np.int32)
        input_x1, scales, zero_points = self.generate_data_npu_quantize(datainfo)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 4, torch.quint8)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 4, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_npu_quantize_30_30_30_30_30_2_uint8(self, device="npu"):
        datainfo = DataInfo(-1, 1, (30, 30, 30, 30), (30,), (30,), np.float16, np.float32, np.uint8)
        input_x1, scales, zero_points = self.generate_data_npu_quantize(datainfo)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1_cpu, scales, zero_points, 2, torch.quint8)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 2, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quantize_ascend_quant_v2_perchannel(self):
        datainfo = DataInfo(-1, 1, (16, 128), (128,), (128,), np.float16, np.float16, np.float16)
        input_x1, scales, zero_points = self.generate_data_npu_quantize(datainfo)
        cpu_output1 = self.cpu_op_exec_ascend_quant_v2(input_x1.numpy(), scales.numpy(), zero_points.numpy(), 1, torch.qint8)
        npu_output1 = self.npu_op_exec_ascend_quant_v2(input_x1, scales, zero_points, 1, torch.qint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quantize_ascend_quant_v2_perhead(self):
        datainfo = DataInfo(-1, 1, (16, 128), (16, 1), (16, 1), np.float16, np.float16, np.float16)
        input_x1, scales, zero_points = self.generate_data_npu_quantize(datainfo)
        cpu_output1 = self.cpu_op_exec_ascend_quant_v2(input_x1.numpy(), scales.numpy(), zero_points.numpy(), -2, torch.qint8)
        npu_output1 = self.npu_op_exec_ascend_quant_v2(input_x1, scales, zero_points, -2, torch.qint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quantize_ascend_quant_v2_pertensor(self):
        datainfo = DataInfo(-1, 1, (16, 128), (1,), (1,), np.float16, np.float16, np.float16)
        input_x1, scales, zero_points = self.generate_data_npu_quantize(datainfo)
        cpu_output1 = self.cpu_op_exec_ascend_quant_v2(input_x1.numpy(), scales.numpy(), zero_points.numpy(), -1, torch.qint8)
        npu_output1 = self.npu_op_exec_ascend_quant_v2(input_x1, scales, zero_points, -1, torch.qint8)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
