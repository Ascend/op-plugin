import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class DataInfo(object):
    def __init__(self, min_d, max_d, shape_x, shape_scale, shape_group_index, dtype_x, dtype_scale):
        self.min_d = min_d
        self.max_d = max_d
        self.shape_x = shape_x
        self.shape_scale = shape_scale
        self.shape_group_index = shape_group_index
        self.dtype_x = dtype_x
        self.dtype_scale = dtype_scale


class TestNPUGroupQuant(TestCase):

    def generate_data_npu_quantize(self, datainfo):
        input_x = np.random.uniform(datainfo.min_d, datainfo.max_d, datainfo.shape_x).astype(datainfo.dtype_x)
        scale = np.random.uniform(datainfo.min_d, datainfo.max_d, datainfo.shape_scale).astype(datainfo.dtype_scale)
        offset = np.random.uniform(datainfo.min_d, datainfo.max_d, (1,)).astype(datainfo.dtype_scale)

        S = datainfo.shape_x[0]
        E = datainfo.shape_group_index[0]
        group_index = np.random.uniform(0, S, E - 1).astype('int32')
        group_index = np.sort(group_index)
        group_index = np.append(group_index, S)

        npu_input_x = torch.from_numpy(input_x)
        npu_input_scale = torch.from_numpy(scale)
        npu_input_group_index = torch.from_numpy(group_index)
        npu_input_offset = torch.from_numpy(offset)

        return [npu_input_x, npu_input_scale, npu_input_group_index, npu_input_offset]

    def cpu_op_exec_group_quant(self, input_x, input_scale, input_group_index, input_offset, dtype):
        S = input_x.shape[0]
        H = input_x.shape[1]
        E = input_scale.shape[0]

        input_x = input_x.astype("float32")
        input_scale = input_scale.astype("float32")
        input_offset = input_offset.astype("float32")
        y = np.empty(shape=(0, H), dtype='float32')

        for row_scale in range(E):
            x_start_row = 0 if row_scale == 0 else input_group_index[row_scale - 1]
            x_end_row = input_group_index[row_scale]
            if x_start_row < x_end_row:
                y_rows = input_x[x_start_row:x_end_row] * input_scale[row_scale] + input_offset
                y = np.concatenate([y, y_rows], axis=0)
        
        y = np.round(y, 0)
        y = np.clip(y, -128, 127).astype("int8")
        return y

    def npu_op_exec_group_quant(self, input_x, input_scale, input_group_index, input_offset, dtype):
        input_x = input_x.to("npu")
        input_scale = input_scale.to("npu")
        input_group_index = input_group_index.to("npu")
        input_offset = input_offset.to("npu")
        output = torch_npu.npu_group_quant(input_x, input_scale, input_group_index, offset=input_offset, dst_dtype=dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @SupportedDevices(['Ascend910B'])
    def test_npu_group_quant(self):
        datainfo = DataInfo(-1, 1, (16, 128), (5, 128), (5,), np.float32, np.float32)
        x, scale, group_index, offset = self.generate_data_npu_quantize(datainfo)
        cpu_output1 = self.cpu_op_exec_group_quant(x.numpy(), scale.numpy(), group_index.numpy(), offset.numpy(), torch.qint8)
        npu_output1 = self.npu_op_exec_group_quant(x, scale, group_index, offset, torch.qint8)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
