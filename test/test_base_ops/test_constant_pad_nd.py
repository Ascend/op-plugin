import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


torch.npu.set_compile_mode(jit_compile=True)


class TestConstantPadNd(TestCase):

    def test_constant_pad_nd_with_negative(self):
        x = torch.randint(0, 100, (1, 10), dtype=torch.int8)
        x_npu = x.npu()
        pad = (1, -1, 1, -1)
        value = -74
        res = torch.constant_pad_nd(x, pad, value)
        res_npu = torch.constant_pad_nd(x_npu, pad, value)
        self.assertEqual(res, res_npu)


if __name__ == "__main__":
    run_tests()
