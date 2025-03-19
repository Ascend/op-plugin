import unittest
import torch
import torch_npu
import hypothesis

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantMatmulDequant(TestCase):

    @unittest.skip("skip test_quant_matmul_dequant_dynamic_pertoken")
    @SupportedDevices(['Ascend310P'])
    def test_quant_matmul_dequant_dynamic_pertoken(self):
        M = 64
        K = 256
        N = 512

        torch_npu.npu.set_compile_mode(jit_compile=False)

        x = torch.randn(M, K).half().npu()
        quantized_weight = torch.randn(K, N).char().npu()
        perchannel_scale = torch.randn(N).float().npu()

        pertoken_scale = (torch.max(torch.abs(x), 1)[0].float() / 127.0).npu()
        x_float = x.transpose(0, 1).float()
        x_quantized = torch_npu.npu_quantize(x_float, (1.0 / pertoken_scale), None,
                                             torch.qint8, axis=-1, div_mode=False)
        x_quantized = x_quantized.transpose(0, 1)
        y_golden = torch.matmul(x_quantized.float(), quantized_weight.float()) \
                   * perchannel_scale * (pertoken_scale.reshape(-1, 1))
        y_golden = y_golden.half()

        quantized_weight_trans = torch.transpose(quantized_weight, 0, 1).contiguous()
        y = torch_npu.npu_quant_matmul_dequant(x, quantized_weight_trans, perchannel_scale,
                                               quant_mode='pertoken')

        self.assertRtolEqual(y_golden, y)

    @unittest.skip("skip test_quant_matmul_dequant_static_pertensor")
    @SupportedDevices(['Ascend310P'])
    def test_quant_matmul_dequant_static_pertensor(self):
        M = 64
        K = 256
        N = 512

        torch_npu.npu.set_compile_mode(jit_compile=False)

        x = torch.randn(M, K).half().npu()
        quantized_weight = torch.randn(K, N).char().npu()
        perchannel_scale = torch.randn(N).float().npu()
        smooth_scale = torch.randn(K).half().npu()
        pertensor_scale = torch.randn(1).float().npu()

        x_quantized = torch_npu.npu_quantize(x.float() * smooth_scale.float(),
                                             (1.0 / pertensor_scale).repeat(x.size()[1]),
                                             None, torch.qint8, axis=-1, div_mode=False)
        y_golden = torch.matmul(x_quantized.float(), quantized_weight.float()) \
                   * perchannel_scale * pertensor_scale
        y_golden = y_golden.half()

        quantized_weight_trans = torch.transpose(quantized_weight, 0, 1).contiguous()
        y = torch_npu.npu_quant_matmul_dequant(x, quantized_weight_trans, perchannel_scale,
                                               x_scale=pertensor_scale, smooth_scale=smooth_scale,
                                               quant_mode='pertensor')

        self.assertRtolEqual(y_golden, y)


if __name__ == "__main__":
    run_tests()
