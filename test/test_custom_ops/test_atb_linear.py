import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAtbLinear(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_atb_linear(self):
        atb_res = torch.zeros((8192, 8192), dtype=torch.float32).npu()
        x = torch.rand((4096, 8192), dtype=torch.float16).npu()
        weight = torch.rand((4096, 8192), dtype=torch.float16).npu()
        c = torch.rand((8192, 8192), dtype=torch.float32).npu()

        # 分开算子计算结果
        product = torch.mm(x.T, weight)
        npu_res = product + c

        torch_npu._npu_matmul_add_fp32(x, weight, c)
        atb_res = atb_res + c
        self.assertRtolEqual(npu_res, atb_res, 0.001, 0.001)


if __name__ == "__main__":
    run_tests()
