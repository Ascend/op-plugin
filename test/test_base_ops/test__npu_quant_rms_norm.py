import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantRmsNorm(TestCase):
    def cpu_quant_rms_norm(self, x, gamma, beta, scale, offset, eps=1e-5):

        x_cpu = x.cpu().float()
        gamma_cpu = gamma.cpu().float().squeeze(0)
        beta_cpu = beta.cpu().float().squeeze(0)
        scale_cpu = scale.cpu().float().item()
        offset_cpu = offset.cpu().item()

        batch, seq, dim = x_cpu.shape
        output = torch.zeros(batch, seq, dim, dtype=torch.int8)

        for b in range(batch):
            for s in range(seq):

                x_sample = x_cpu[b, s, :]
                mean_square = torch.mean(x_sample ** 2)
                rms = torch.sqrt(mean_square + eps)

                x_normalized = x_sample / rms

                y = x_normalized * gamma_cpu + beta_cpu

                y_quantized = torch.round(y / scale_cpu + offset_cpu)
                y_quantized = torch.clamp(y_quantized, -128, 127).to(torch.int8)
                output[b, s, :] = y_quantized

        return output

    @SupportedDevices(["Ascend910B"])
    def test_quant_int8_rms_norm(self):
        torch.manual_seed(12)
        x = torch.randn(4, 1, 128).half()
        gamma = torch.randn(1, 128).half()
        beta = torch.randn(1, 128).half()
        scale = torch.tensor([0.3], dtype=torch.float16)
        offset = torch.tensor([2], dtype=torch.int8)
        output_npu = torch.zeros_like(x, dtype=torch.int8).npu()

        output_cpu = self.cpu_quant_rms_norm(x, gamma, beta, scale, offset)
        torch_npu._npu_quant_rms_norm(x.npu(), 
                                    gamma.npu(), 
                                    beta.npu(), 
                                    scale.npu(), 
                                    offset.npu(), 
                                    output_npu, 
                                    eps=1e-5)

        self.assertEqual(output_npu.cpu(), output_cpu)


if __name__ == "__main__":
    run_tests()
