import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestTransposeBatchMatmul(TestCase):

    def supported_op_exec(self, x1, x2, scale):
        x1 = x1.permute([1, 0, 2])
        out = torch.matmul(x1.float(), x2.float())
        out = out.permute([1, 0, 2])
        out = out.reshape(out.shape[0], 1, out.shape[1] * out.shape[2])
        output = torch_npu.npu_quantize(out, scale, None, torch.qint8, -1, True)
        return output

    @unittest.skip("Skipping test_npu_transpose_batchmatmul temporarily")
    @SupportedDevices(["Ascend910B"])
    def test_npu_transpose_batchmatmul(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randn((M, Batch, K), dtype=torch.float16)
        x2 = torch.randn((Batch, K, N), dtype=torch.float16)
        scale = torch.rand((Batch * N, ), dtype=torch.float32)
        scale = torch_npu.npu_trans_quant_param(scale, round_mode=1)
        supported_output = self.supported_op_exec(x1, x2, scale)
        custom_output = torch_npu.npu_transpose_batchmatmul(x1.npu(), x2.npu(), scale=scale.npu(),
                                                            perm_x1=[1, 0, 2], perm_y=[1, 0, 2]).to("cpu")
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()
