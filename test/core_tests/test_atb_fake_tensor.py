
import unittest
import torch
import torch_npu
from torch import distributed as dist
from torch._dynamo.testing import rand_strided
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase, run_tests


class TestAtbFakeTensor(TestCase):
    def test_npu_multi_head_latent_attentionn(self):
        with FakeTensorMode():
            block_size = 128
            num_tokens = 32
            num_heads = 32
            kv_heads = 1
            head_size_qk = 576
            head_size_vo = 512
            batch = num_tokens
            num_blocks = 64
            max_num_blocks_per_query = 16
            q_nope = torch.randn((num_tokens, num_heads, head_size_vo), dtype=torch.float16).npu()
            q_rope = torch.randn((num_tokens, num_heads, head_size_qk - head_size_vo), dtype=torch.float16).npu()
            ctkv = torch.randn((num_blocks, block_size, kv_heads, 512), dtype=torch.float16).npu()
            k_rope = torch.randn((num_blocks, block_size, kv_heads, 64), dtype=torch.float16).npu()
            block_tables = torch.randint(0, 10, (batch, max_num_blocks_per_query), dtype=torch.int32).npu()
            context_lens = [2] * batch
            atten_out = torch_npu.atb.npu_multi_head_latent_attention(q_nope, q_rope, ctkv, k_rope, block_tables, context_lens, 32, 1.0, 1)
            
            self.assertTrue(atten_out.shape == q_nope.shape)
            self.assertEqual(atten_out.dtype, q_rope.dtype)

    def test_npu_mla_preprocess(self):
        with FakeTensorMode():
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
            quant_offset0 = torch.randint(0, 7, (1,), dtype=torch.int8, device=device)

            wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8, device=device)
            de_scale0 = torch.rand((2112, ), dtype=torch.float, device=device)
            bias0 = torch.randint(0, 7, (2112, ), dtype=torch.int32, device=device)

            gamma1 = torch.randn((1536), dtype=dtype, device=device)
            beta1 = torch.randn((1536), dtype=dtype, device=device)
            quant_scale1 = torch.randn((1,), dtype=dtype, device=device)
            quant_offset1 = torch.randint(0, 7, (1,), dtype=torch.int8, device=device)

            wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8, device=device)
            de_scale1 = torch.rand((head_num * 192, ), dtype=torch.float, device=device)
            bias1 = torch.randint(0, 7, (head_num * 192, ),
                                dtype=torch.int32, device=device)

            gamma2 = torch.randn((512), dtype=dtype, device=device)

            cos = torch.randn((token_num, 64), dtype=dtype, device=device)
            sin = torch.randn((token_num, 64), dtype=dtype, device=device)

            wuk = torch.randn((head_num, 128, 512), dtype=dtype, device=device)

            kv_cache = torch.randint(0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
            kv_cache_rope = torch.randn((block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)

            slotmapping = torch.randint(0, 7, (token_num,), dtype=torch.int32, device=device)

            ctkv_scale = torch.randn((1,), dtype=dtype, device=device)
            qnope_scale = torch.randn((head_num), dtype=dtype, device=device)
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
            token_num = input1.size(0)
            head_num = wuk.size(0)
            self.assertTrue(list(out[0].shape) == [token_num, head_num, 512])
            self.assertTrue(out[1].shape == kv_cache.shape)
            self.assertTrue(list(out[2].shape) == [token_num, head_num, 64])
            self.assertTrue(out[3].shape == kv_cache_rope.shape)
            self.assertEqual(out[0].dtype, kv_cache.dtype)
            self.assertEqual(out[1].dtype, kv_cache.dtype)
            self.assertEqual(out[2].dtype, input1.dtype)
            self.assertEqual(out[3].dtype, kv_cache_rope.dtype)

    def test_npu_self_attention_prefix_encoder(self):
        with FakeTensorMode():
            batch = 4
            qseqlen = 128
            headnum = 28
            headsize = 128
            numblocks = 64
            block_size = 128
            q_seqlens = [32] * batch
            kv_seqLen = [32] * batch
            dtype = torch.float16

            query = torch.randn((batch * qseqlen, headnum, headsize), dtype=dtype).npu()
            key = torch.randn((numblocks, block_size, headnum, headsize), dtype=dtype).npu()
            value = torch.randn((numblocks, block_size, headnum, headsize), dtype=dtype).npu()
            block_tables = torch.randint(0, 10, (batch, 64), dtype=torch.int32).npu()

            output = torch_npu.atb.npu_self_attention_prefix_encoder(query, key, value, block_tables, q_seqlens, kv_seqLen, 28, 0.0883, 4)
            self.assertTrue(output.shape == query.shape)
            self.assertEqual(output.dtype, query.dtype)           

    def test_npu_fused_add_topk_div_meta(self):
        with FakeTensorMode():
            a = 16
            b = 256
            k = 8
            c = 64

            x = torch.randn(a, b, dtype=torch.float16).npu()
            add_num = torch.randn(b, dtype=torch.float16).npu()
            mapping_num = torch.randint(1, c + 1, (b,), dtype=torch.int32).npu()
            mapping_table = torch.randint(0, 10, (b, c), dtype=torch.int32).npu()
            y = torch.empty(a, k, dtype=torch.float32).npu()
            indices = torch.empty(a, k, dtype=torch.int32).npu()
            output = torch_npu.atb.npu_fused_add_topk_div(x, add_num, mapping_num=mapping_num, mapping_table=mapping_table, activation_type='activation_sigmoid', group_num=8, group_topk=4, n=2, k=k, is_norm=True, scale=1, enable_expert_mapping=True)
            self.assertTrue(output[0].shape == y.shape)
            self.assertTrue(output[1].shape == indices.shape)
            self.assertEqual(output[0].dtype, torch.float32)
            self.assertEqual(output[1].dtype, torch.int32)


if __name__ == "__main__":
    run_tests()
