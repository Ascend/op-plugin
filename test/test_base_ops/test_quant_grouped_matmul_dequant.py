import unittest
import torch
import torch_npu
import hypothesis

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantGroupedMatmulDequant(TestCase):

    @unittest.skip("skip test_quant_grouped_matmul_dequant_dynamic_pertoken")
    @SupportedDevices(['Ascend310P'])
    def test_quant_grouped_matmul_dequant_dynamic_pertoken(self):
        G = 4
        M = 64
        K = 256
        N = 512

        torch_npu.npu.set_compile_mode(jit_compile=False)

        x = torch.randn(M, K).half().npu()
        quantized_weight = torch.randn(G, K, N).char().npu()
        perchannel_scale = torch.randn(G, N).float().npu()
        group_list = torch.tensor([7, 29, 31, 64]).npu()

        pertoken_scale = (torch.max(torch.abs(x), 1)[0].float() / 127.0).npu()
        x_float = x.transpose(0, 1).float()
        x_quantized = torch_npu.npu_quantize(x_float, (1.0 / pertoken_scale), None,
                                             torch.qint8, axis=-1, div_mode=False)
        x_quantized = x_quantized.transpose(0, 1)
        y_golden = []
        start_m = 0
        for i in range(G):
            end_m = group_list[i].item()
            tmp = torch.matmul(x_quantized[start_m:end_m, :].float(), quantized_weight[i, :, :].float()) \
                  * perchannel_scale[i, :] * (pertoken_scale[start_m:end_m].reshape(-1, 1))
            y_golden.append(tmp.half())
            start_m = end_m
        y_golden = torch.cat(y_golden)

        quantized_weight_trans = torch.transpose(quantized_weight, 1, 2).contiguous()
        y = torch_npu.npu_quant_grouped_matmul_dequant(x, quantized_weight_trans, perchannel_scale,
                                                       group_list, quant_mode='pertoken')

        self.assertRtolEqual(y_golden, y)

    @unittest.skip("skip test_quant_grouped_matmul_dequant_static_pertensor")
    @SupportedDevices(['Ascend310P'])
    def test_quant_grouped_matmul_dequant_static_pertensor(self):
        G = 4
        M = 64
        K = 256
        N = 512

        torch_npu.npu.set_compile_mode(jit_compile=False)

        x = torch.randn(M, K).half().npu()
        quantized_weight = torch.randn(G, K, N).char().npu()
        perchannel_scale = torch.randn(G, N).float().npu()
        smooth_scale = torch.randn(K).half().npu()
        pertensor_scale = torch.randn(1).float().npu()
        group_list = torch.tensor([7, 29, 31, 64]).npu()

        x_quantized = torch_npu.npu_quantize(x.float() * smooth_scale.float(),
                                             (1.0 / pertensor_scale).repeat(x.size()[1]),
                                             None, torch.qint8, axis=-1, div_mode=False)
        y_golden = []
        start_m = 0
        for i in range(G):
            end_m = group_list[i].item()
            tmp = torch.matmul(x_quantized[start_m:end_m, :].float(), quantized_weight[i, :, :].float()) \
                  * perchannel_scale[i, :] * pertensor_scale
            y_golden.append(tmp.half())
            start_m = end_m
        y_golden = torch.cat(y_golden)

        quantized_weight_trans = torch.transpose(quantized_weight, 1, 2).contiguous()
        y = torch_npu.npu_quant_grouped_matmul_dequant(x, quantized_weight_trans, perchannel_scale, group_list,
                                                       x_scale=pertensor_scale, smooth_scale=smooth_scale,
                                                       quant_mode='pertensor')

        self.assertRtolEqual(y_golden, y)


if __name__ == "__main__":
    run_tests()
