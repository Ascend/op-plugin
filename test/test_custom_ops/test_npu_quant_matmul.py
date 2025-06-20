import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantMatmul(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_a8w8(self):
        torch.manual_seed(0)
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        scale_quant = torch_npu.npu_trans_quant_param(scale.npu(), None)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale_quant.npu(), output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_a4w4(self):
        torch.manual_seed(0)
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int32)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int32)
        x1_clone = x1.clone().float().npu()
        x2_clone = x2.clone().float().npu()
        scale = torch.randn(1, dtype=torch.float32)
        scale_clone = scale.clone().npu()
        scale_quant = torch_npu.npu_trans_quant_param(scale_clone, None)

        # convert int32 to int4*8
        scale_tmp = scale.clone().npu()
        scale_tmp[0] = 1
        x1_clone = torch_npu.npu_quantize(x1_clone, scale_tmp, None, torch.quint4x2, -1, False)
        x2_clone = torch_npu.npu_quantize(x2_clone, scale_tmp, None, torch.quint4x2, -1, False)

        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone, x2_clone, scale_quant, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_a4w4_pertoken(self):
        torch.manual_seed(0)
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int32)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int32)
        x1_clone = x1.clone().float().npu()
        x2_clone = x2.clone().float().npu()
        scale = torch.randn(1, dtype=torch.float32)
        pertoken_scale = torch.randn(8192, dtype=torch.float32)
        scale_clone = scale.clone().npu()
        pertoken_scale_clone = pertoken_scale.clone().npu()

        # convert int32 to int4*8
        scale_tmp = scale.clone().npu()
        scale_tmp[0] = 1
        x1_clone = torch_npu.npu_quantize(x1_clone, scale_tmp, None, torch.quint4x2, -1, False)
        x2_clone = torch_npu.npu_quantize(x2_clone, scale_tmp, None, torch.quint4x2, -1, False)

        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale * pertoken_scale.unsqueeze(-1)
        custom_output_bf16 = torch_npu.npu_quant_matmul(x1_clone, x2_clone, scale_clone, pertoken_scale=pertoken_scale_clone, output_dtype=torch.bfloat16)
        custom_output_fp16 = torch_npu.npu_quant_matmul(x1_clone, x2_clone, scale_clone, pertoken_scale=pertoken_scale_clone, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output_bf16.float().cpu().numpy(), 0.01)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output_fp16.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_continuous_x2_tensor(self):
        torch.manual_seed(0)
        x1 = torch.randint(-5, 5, (5, 160), dtype=torch.int32)
        x2 = torch.randint(-5, 5, (80, 160), dtype=torch.int32)
        x1_clone = x1.clone().float().npu()
        x2_clone = x2.clone().float().npu()
        scale = torch.randn(1, dtype=torch.float32)
        scale_clone = scale.clone().npu()
        scale_quant = torch_npu.npu_trans_quant_param(scale_clone, None)

        # convert int32 to int4*8
        scale_tmp = scale.clone().npu()
        scale_tmp[0] = 1
        x1_clone = torch_npu.npu_quantize(x1_clone, scale_tmp, None, torch.quint4x2, -1, False)
        x2_clone = torch_npu.npu_quantize(x2_clone, scale_tmp, None, torch.quint4x2, -1, False)

        supported_output = torch.matmul(x1.to(torch.int32), x2.t().to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone, x2_clone.t(), scale_quant, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_bf16(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_bf16_nz(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        x2_nz = torch_npu.npu_format_cast(x2_clone.npu().contiguous(), 29)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_nz.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_fp16_nz(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        x2_nz = torch_npu.npu_format_cast(x2_clone.npu().contiguous(), 29)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_nz.npu(), scale.npu(), output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @unittest.skip("skip test_npu_quant_matmul_int32")
    def test_npu_quant_matmul_int32(self):
        x1 = torch.randint(-5, 5, (16, 6656), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (6656, 4992), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32))
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale.npu(), output_dtype=torch.int32)
        self.assertRtolEqual(supported_output.cpu().numpy().astype(np.float32), custom_output.cpu().numpy().astype(np.float32), 0.001)

    # pylint:disable = huawei-too-many-arguments
    def a8w4_quant_golden(self, x, weight, scale, perTokenScale, groupList, bias, output_dtype):
        m, k = x.shape
        k, n = weight.shape
        x = x.numpy()
        weight = weight.cpu().numpy()
        scale = scale.numpy()
        perTokenScale = perTokenScale.numpy()
        bias = bias.numpy()
        bias = bias.reshape(1, -1)
        weight_int8 = weight.astype(np.int8)
        x = np.concatenate([x.reshape(m, 1, k) // 16, (x.reshape(m, 1, k) & 0x0F) - 8], axis=1).reshape(m * 2, k)

        groupNum = 1
        quantGroupNum = scale.shape[0]
        index = np.cumsum(groupList)
        xSplit = np.split(x, index * 2, axis=0)
        xSplit[0] = x
        perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
        weightGroup = weight_int8.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
        mmOuts = []
        atomic = np.float16
        mmi = []
        for i in range(groupNum):
            xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
            mmi = np.zeros([xi.shape[0], n], dtype=atomic)
            for j in range(quantGroupNum):
                mm = np.matmul(xi[:, j, :], weightGroup[i, j, ...])
                mm = mm.astype(np.float32) * scale[j, :].reshape(1, -1)
                mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)

            mmi = mmi.reshape(-1, 2, n).astype(np.float32)
            mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias[i].reshape(1, n)
            mmi = mmi * perTokenScale
            mmOuts.append(mmi)
        golden = np.concatenate(mmOuts, axis=0)
        golden_tensor = torch.from_numpy(mmi)

        return golden_tensor.to(output_dtype)

    def get_eb(self, golden: torch.Tensor, actual: torch.Tensor):
        golden = golden.to(torch.float32)
        golden_nmax = torch.clamp(torch.abs(golden), min=1)
        actual_error = actual.to(torch.float32) - golden
        EB = torch.mean(actual_error / golden_nmax)
        if golden.dtype == torch.float16:
            result = EB <= 2 ** (-10)
        else:
            result = EB <= 2 ** (-7)
        return result

    def ref_compare(self, golden: torch.Tensor, actual: torch.Tensor, thresh: float):
        golden = golden.to(torch.float32)
        golden_nmax = torch.clamp(torch.abs(golden), min=1)
        abs_error = torch.abs(actual.to(torch.float32) - golden)
        result = (abs_error <= thresh * golden_nmax).all()
        return result

    def golden_compare(self, out_tensor, golden_out_tensor, ksize):
        eb = self.get_eb(golden_out_tensor, out_tensor)
        if out_tensor.dtype == torch.float16:
            cmp = self.ref_compare(golden_out_tensor, out_tensor, 2 ** -8 if ksize < 2048 else 2**-7)
        else:
            cmp = self.ref_compare(golden_out_tensor, out_tensor, 2 ** -7 if ksize < 2048 else 2**-6)
        return eb and cmp

    @unittest.skip("skip test_npu_quant_matmul_A8W4")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_A8W4_float16(self):
        M = 1
        K = 8192
        N = 128 * 8
        group_size = 256

        x1 = torch.randint(-128, 128, [M, K], dtype=torch.int8)
        x2 = torch.randint(1, 8, (K, N), dtype=torch.int32)
        y_offset = torch.randn([N, ], dtype=torch.float32)
        x1_scale = torch.randn([M, 1], dtype=torch.float32) * 0.01
        x2_scale = torch.randn([K // group_size, N], dtype=torch.float32).uniform_(0, 1)
        group_size_list = [0, 0, group_size]

        x1_npu = x1.npu()
        x2_npu = torch_npu.npu_convert_weight_to_int4pack(x2.npu())
        y_offset_npu = y_offset.npu()
        x1_scale_npu = x1_scale.npu()
        x2_scale_tmp = torch_npu.npu_trans_quant_param(x2_scale.npu().reshape([K // group_size * N, ]))
        x2_scale_npu = x2_scale_tmp.reshape([K // group_size, N])

        npu_out = torch_npu.npu_quant_matmul(x1_npu, x2_npu, scale=x2_scale_npu,
                                             offset=y_offset_npu, pertoken_scale=x1_scale_npu,
                                             bias=None, output_dtype=torch.float16, group_sizes=group_size_list)
        cpu_out = self.a8w4_quant_golden(x=x1, weight=x2, scale=x2_scale, perTokenScale=x1_scale,
                                         groupList=group_size_list, bias=y_offset, output_dtype=torch.float16)

        self.assertTrue(self.golden_compare(npu_out.cpu(), cpu_out, K))
    
    @unittest.skip("skip test_npu_quant_matmul_A8W4")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_A8W4_bfloat16(self):
        M = 1
        K = 8192
        N = 128 * 8
        group_size = 256

        x1 = torch.randint(-128, 128, [M, K], dtype=torch.int8)
        x2 = torch.randint(1, 8, (K, N), dtype=torch.int32)
        y_offset = torch.randn([N, ], dtype=torch.float32)
        x1_scale = torch.randn([M, 1], dtype=torch.float32) * 0.01
        x2_scale = torch.randn([K // group_size, N], dtype=torch.float32).uniform_(0, 1)
        group_size_list = [0, 0, group_size]

        x1_npu = x1.npu()
        x2_npu = torch_npu.npu_convert_weight_to_int4pack(x2.npu())
        y_offset_npu = y_offset.npu()
        x1_scale_npu = x1_scale.npu()
        x2_scale_tmp = torch_npu.npu_trans_quant_param(x2_scale.npu().reshape([K // group_size * N, ]))
        x2_scale_npu = x2_scale_tmp.reshape([K // group_size, N])

        npu_out = torch_npu.npu_quant_matmul(x1_npu, x2_npu, scale=x2_scale_npu,
                                             offset=y_offset_npu, pertoken_scale=x1_scale_npu,
                                             bias=None, output_dtype=torch.bfloat16, group_sizes=group_size_list)
        cpu_out = self.a8w4_quant_golden(x=x1, weight=x2, scale=x2_scale, perTokenScale=x1_scale,
                                         groupList=group_size_list, bias=y_offset, output_dtype=torch.bfloat16)

        self.assertTrue(self.golden_compare(npu_out.cpu(), cpu_out, K))


if __name__ == "__main__":
    run_tests()
