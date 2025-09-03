import unittest
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestDynamicBlockQuant(TestCase):
    def calc_base(self, cpu_input):
        if len(cpu_input.shape) == 3:
            b0, m0, n0 = cpu_input.shape
            m = b0 * m0
            n = n0
        elif len(cpu_input.shape) == 2:
            m, n = cpu_input.shape
        pad_size = (128 - (n % 128)) % 128
        x = cpu_input.view((m, n))
        x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1)
        y_data = x_view * (127.0 / x_amax.unsqueeze(2))
        y_data = y_data.round()
        y_data = torch.clamp(y_data, -127, 127).to(torch.int8)
        y_data = y_data.view(m, n + pad_size)[:, :n].view(cpu_input.shape)
        scale = x_amax / 127.0
        if len(cpu_input.shape) == 3:
            scale = scale.view((b0, m0 - 1))
        else:
            scale = scale.view((m, -1))
        return y_data, scale

    def generate_data(self, lo, hi, shape, dtype):
        predict = np.random.uniform(lo, hi, shape).astype(dtype)
        # modify from numpy.ndarray to torch.tensor
        npu_predict = torch.from_numpy(predict)
        return npu_predict

    def cpu_op_exec(self, x):
        return self.calc_base(x)

    def npu_op_exec(self, x):
        x = x.to("npu")
        min_scale = 0
        dst_type = 1 # ACL_INT8
        row_block_size = 1
        col_block_size = 128
        output = torch_npu.npu_dynamic_block_quant(x, dst_type=dst_type)
        return output

    @unittest.skip("skip until CANN is updated to support aclnnScatterPaKvCache")
    def test_npu_dynamic_block_quant(self):
        x = self.generate_data(-2, 2, (4, 3), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
