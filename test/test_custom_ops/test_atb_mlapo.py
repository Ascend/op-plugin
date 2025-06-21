import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


# Since the benchmark function requires atb operator splicing, pta is not adapted yet. Currently only caretaker operator function
class TestAtbMLAPO(TestCase):

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_atb_mlapo_20(self):
        token_num = 8
        head_num = 128
        N_7168 = 7168
        block_num = 192
        block_size = 128
        dtype = torch.float16
        device = 'npu'

        input1 = torch.randn((token_num, N_7168), dtype=dtype, device=device)

        gamma0 = torch.randn((N_7168), dtype=dtype, device=device)
        beta0 = torch.randn((N_7168), dtype=dtype, device=device)
        quant_scale0 = torch.randn((1,), dtype=dtype, device=device)
        quant_offset0 = torch.randint(
            0, 7, (1,), dtype=torch.int8, device=device)

        wdqkv = torch.randint(0, 7, (1, 224, 2112, 32),
                              dtype=torch.int8, device=device)
        wdqkv = torch_npu.npu_format_cast(wdqkv, 29)
        de_scale0 = torch.randint(
            0, 7, (2112, ), dtype=torch.int64, device=device)
        bias0 = torch.randint(0, 7, (2112, ), dtype=torch.int32, device=device)

        gamma1 = torch.randn((1536), dtype=dtype, device=device)
        beta1 = torch.randn((1536), dtype=dtype, device=device)
        quant_scale1 = torch.randn((1,), dtype=dtype, device=device)
        quant_offset1 = torch.randint(
            0, 7, (1,), dtype=torch.int8, device=device)

        wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32),
                            dtype=torch.int8, device=device)
        wuq = torch_npu.npu_format_cast(wuq, 29)
        de_scale1 = torch.randint(
            0, 7, (head_num * 192, ), dtype=torch.int64, device=device)
        bias1 = torch.randint(0, 7, (head_num * 192, ),
                              dtype=torch.int32, device=device)

        gamma2 = torch.randn((512), dtype=dtype, device=device)

        cos = torch.randn((token_num, 64), dtype=dtype, device=device)
        sin = torch.randn((token_num, 64), dtype=dtype, device=device)

        wuk = torch.randn((head_num, 128, 512), dtype=dtype, device=device)
        wuk = torch_npu.npu_format_cast(wuk, 29)

        kv_cache = torch.randint(
            0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
        kv_cache = torch_npu.npu_format_cast(kv_cache, 29)
        kv_cache_rope = torch.randn(
            (block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)
        kv_cache_rope = torch_npu.npu_format_cast(kv_cache_rope, 29)

        slotmapping = torch.randint(
            0, 7, (token_num,), dtype=torch.int32, device=device)

        ctkv_scale = torch.randn((1,), dtype=dtype, device=device)
        qnope_scale = torch.randn((head_num), dtype=dtype, device=device)

        q_out0 = torch.randint(
            0, 7, (token_num, head_num, 512), dtype=torch.int8, device=device)
        kv_cache_out0 = torch.randint(
            0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
        kv_cache_out0 = torch_npu.npu_format_cast(kv_cache_out0, 29)
        q_out1 = torch.randn((token_num, head_num, 64),
                             dtype=dtype, device=device)
        kv_cache_out1 = torch.randn(
            (block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)
        kv_cache_out1 = torch_npu.npu_format_cast(kv_cache_out1, 29)

        torch_npu.atb.npu_mla_preprocess(
            input1, gamma0, beta0, wdqkv, de_scale0,
            gamma1, beta1, wuq, de_scale1,
            gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping,
            quant_scale0=quant_scale0,
            quant_offset0=quant_offset0,
            bias0=bias0,
            quant_scale1=quant_scale0,
            quant_offset1=quant_offset1,
            bias1=bias1,
            ctkv_scale=ctkv_scale,
            q_nope_scale=qnope_scale,
            cache_mode="int8_nzcache",
            quant_mode="per_tensor_quant_asymm",
            q_out0=q_out0,
            kv_cache_out0=kv_cache_out0,
            q_out1=q_out1,
            kv_cache_out1=kv_cache_out1,
        )
        torch.npu.synchronize()

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_atb_mlapo_20_model_case(self):
        token_num = 1
        head_num = 128
        N_7168 = 7168
        block_num = 192
        block_size = 128
        dtype = torch.bfloat16
        device = 'npu:0'

        input1 = torch.randn((token_num, N_7168), dtype=dtype, device=device)

        gamma0 = torch.randn((N_7168), dtype=dtype, device=device)
        beta0 = torch.randn((N_7168), dtype=dtype, device=device)
        quant_scale0 = torch.randn((1,), dtype=dtype, device=device)
        quant_offset0 = torch.randint(
            0, 7, (1,), dtype=torch.int8, device=device)

        wdqkv = torch.randint(0, 7, (1, 224, 2112, 32),
                              dtype=torch.int8, device=device)
        wdqkv = torch_npu.npu_format_cast(wdqkv, 29)
        de_scale0 = torch.rand((2112, ), dtype=torch.float, device=device)
        bias0 = torch.randint(0, 7, (2112, ), dtype=torch.int32, device=device)

        gamma1 = torch.randn((1536), dtype=dtype, device=device)
        beta1 = torch.randn((1536), dtype=dtype, device=device)
        quant_scale1 = torch.randn((1,), dtype=dtype, device=device)
        quant_offset1 = torch.randint(
            0, 7, (1,), dtype=torch.int8, device=device)

        wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32),
                            dtype=torch.int8, device=device)
        wuq = torch_npu.npu_format_cast(wuq, 29)
        de_scale1 = torch.rand(
            (head_num * 192, ), dtype=torch.float, device=device)
        bias1 = torch.randint(0, 7, (head_num * 192, ),
                              dtype=torch.int32, device=device)

        gamma2 = torch.randn((512), dtype=dtype, device=device)

        cos = torch.randn((token_num, 64), dtype=dtype, device=device)
        sin = torch.randn((token_num, 64), dtype=dtype, device=device)

        wuk = torch.randn((head_num, 128, 512), dtype=dtype, device=device)
        wuk = torch_npu.npu_format_cast(wuk, 29)

        kv_cache = torch.randint(
            0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
        kv_cache = torch_npu.npu_format_cast(kv_cache, 29)
        kv_cache_rope = torch.randn(
            (block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)
        kv_cache_rope = torch_npu.npu_format_cast(kv_cache_rope, 29)

        slotmapping = torch.randint(
            0, 7, (token_num,), dtype=torch.int32, device=device)

        ctkv_scale = torch.randn((1,), dtype=dtype, device=device)
        qnope_scale = torch.randn((head_num), dtype=dtype, device=device)

        q_out0 = torch.randint(
            0, 7, (token_num, head_num, 512), dtype=torch.int8, device=device)
        kv_cache_out0 = torch.randint(
            0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
        kv_cache_out0 = torch_npu.npu_format_cast(kv_cache_out0, 29)
        q_out1 = torch.randn((token_num, head_num, 64),
                             dtype=dtype, device=device)
        kv_cache_out1 = torch.randn(
            (block_num, head_num*64//16, block_size, 16), dtype=dtype, device=device)
        kv_cache_out1 = torch_npu.npu_format_cast(kv_cache_out1, 29)

        torch_npu.atb.npu_mla_preprocess(
            input1, gamma0, beta0, wdqkv, de_scale0,
            gamma1, beta1, wuq, de_scale1,
            gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping,
            quant_scale0=quant_scale0,
            quant_offset0=quant_offset0,
            bias0=bias0,
            quant_scale1=quant_scale0,
            quant_offset1=quant_offset1,
            bias1=bias1,
            ctkv_scale=ctkv_scale,
            q_nope_scale=qnope_scale,
            cache_mode="int8_nzcache",
            quant_mode="per_tensor_quant_asymm",
            q_out0=q_out0,
            kv_cache_out0=kv_cache_out0,
            q_out1=q_out1,
            kv_cache_out1=kv_cache_out1,
        )
        out = torch_npu.atb.npu_mla_preprocess(
            input1, gamma0, beta0, wdqkv, de_scale0,
            gamma1, beta1, wuq, de_scale1,
            gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping,
            quant_scale0=quant_scale0,
            quant_offset0=quant_offset0,
            bias0=bias0,
            quant_scale1=quant_scale0,
            quant_offset1=quant_offset1,
            bias1=bias1,
            ctkv_scale=ctkv_scale,
            q_nope_scale=qnope_scale,
            cache_mode="int8_nzcache",
            quant_mode="per_tensor_quant_asymm"
        )
        self.assertEqual(out[0], q_out0)
        self.assertEqual(out[1], kv_cache_out0)
        self.assertEqual(out[2], q_out1)
        self.assertEqual(out[3], kv_cache_out1)


if __name__ == "__main__":
    run_tests()
