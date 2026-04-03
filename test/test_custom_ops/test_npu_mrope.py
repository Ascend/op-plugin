import itertools
import unittest

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices, SkipIfNotGteCANNVersion
from torch_npu.testing.testcase import run_tests, TestCase


class TestNPUMrope(TestCase):
    # pylint:disable=huawei-too-many-arguments
    def golden_mrope(
            self,
            positions,
            query,
            key,
            cos_sin_cache,
            mrope_section,
            head_size,
            is_neox_style=True,
            cache_mode='default',
    ):
        num_tokens = positions.shape[-1]
        rotary_dim = cos_sin_cache.shape[-1]

        positions_flatten = positions.flatten()  # [3, num_tokens] -> [3*num_tokens]
        cos_sin = cos_sin_cache.index_select(
            0, positions_flatten
        )  # cos_sin_cache: [num_tokens, rotary_dim] -> [3*num_tokens, rotary_dim]
        cos_sin = cos_sin.reshape(-1, rotary_dim)

        cos, sin = cos_sin.chunk(
            2, dim=-1
        )  # 1D: [num_tokens, rotary_dim // 2]  2D: [3 * num_tokens, rotary_dim // 2]
        if positions.ndim == 2:
            cos = cos.reshape(
                3, -1, rotary_dim // 2
            )  # [3, num_tokens, rotary_dim // 2]
            sin = sin.reshape(
                3, -1, rotary_dim // 2
            )  # [3, num_tokens, rotary_dim // 2]
            
            # Map cache_mode string to int: 'default' -> 0, 'interleave' -> 1
            if cache_mode == 'default':
                cache_mode_value = 0
            elif cache_mode == 'interleave':
                cache_mode_value = 1
            else:
                cache_mode_value = 0  # default fallback
            
            if cache_mode_value == 0:
                # cacheMode为0: 按顺序拼接
                cos_0 = cos[0, :, : mrope_section[0]]
                cos_1 = cos[1, :, mrope_section[0]: (mrope_section[0] + mrope_section[1])]
                cos_2 = cos[2, :, (mrope_section[0] + mrope_section[1]):
                    (mrope_section[0] + mrope_section[1] + mrope_section[2]), ]
                cos = torch.concat((cos_0, cos_1, cos_2), dim=-1)

                sin_0 = sin[0, :, : mrope_section[0]]
                sin_1 = sin[1, :, mrope_section[0]: (mrope_section[0] + mrope_section[1])]
                sin_2 = sin[2, :, (mrope_section[0] + mrope_section[1]):
                    (mrope_section[0] + mrope_section[1] + mrope_section[2]), ]
                sin = torch.concat((sin_0, sin_1, sin_2), dim=-1)
            else:
                # cacheMode为1: 交错排列
                cos_tmp = cos.clone()
                sin_tmp = sin.clone()
                # cos[...,1:mropeSection[1]*3:3]=cosTmp[1,...,1:mropeSection[1]*3:3]
                cos[0, :, 1:mrope_section[1]*3:3] = cos_tmp[1, :, 1:mrope_section[1]*3:3]
                # cos[...,2:mropeSection[1]*3:3]=cosTmp[2,...,2:mropeSection[1]*3:3]
                cos[0, :, 2:mrope_section[1]*3:3] = cos_tmp[2, :, 2:mrope_section[1]*3:3]
                # sin[...,1:mropeSection[1]*3:3]=sinTmp[1,...,1:mropeSection[1]*3:3]
                sin[0, :, 1:mrope_section[1]*3:3] = sin_tmp[1, :, 1:mrope_section[1]*3:3]
                # sin[...,2:mropeSection[1]*3:3]=sinTmp[2,...,2:mropeSection[1]*3:3]
                sin[0, :, 2:mrope_section[1]*3:3] = sin_tmp[2, :, 2:mrope_section[1]*3:3]
                # 使用 cos[0] 作为基础
                cos = cos[0, :, :]
                sin = sin[0, :, :]

        cos = cos.unsqueeze(-2)  # [num_tokens, 1, rotary_dim // 2]
        sin = sin.unsqueeze(-2)  # [num_tokens, 1, rotary_dim // 2]
        query_shape = query.shape
        query = query.view(
            num_tokens, -1, head_size
        )  # [num_tokens, num_heads, head_size]
        query_rot = query[..., :rotary_dim]  # [num_tokens, num_heads, rotary_dim]
        query_pass = query[..., rotary_dim:]  # [num_tokens, num_heads, head_size - rotary_dim]
        if is_neox_style:
            x1, x2 = torch.chunk(query_rot, 2, dim=-1)  # [num_tokens, num_heads, rotary_dim // 2]
        else:
            x1 = query_rot[..., ::2]
            x2 = query_rot[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            query_rot = torch.cat((o1, o2), dim=-1)
        else:
            query_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(
            num_tokens, -1, head_size
        )  # [num_tokens, num_kv_heads, head_size]
        key_rot = key[..., :rotary_dim]  # [num_tokens, num_kv_heads, rotary_dim]
        key_pass = key[
                   ..., rotary_dim:
                   ]  # [num_tokens, num_kv_heads, head_size - rotary_dim]
        if is_neox_style:
            x1, x2 = torch.chunk(key_rot, 2, dim=-1)
        else:
            x1 = key_rot[..., ::2]
            x2 = key_rot[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            key_rot = torch.cat((o1, o2), dim=-1)
        else:
            key_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(
            key_shape
        )  # [num_tokens, num_kv_heads * head_size]
        return query, key

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B"])
    def test_npu_mrope_rope(self):
        num_tokens_list = [8, 16]
        num_q_heads_list = [8, 16]
        head_size_list = [128]
        rotary_mode_list = ['half', 'interleave']
        cache_mode_list = ['default', 'interleave']
        dtype_list = [torch.bfloat16, torch.float16, torch.float32]

        for (
                num_tokens,
                num_q_heads,
                head_size,
                rotary_mode,
                cache_mode,
                dtype,
        ) in itertools.product(
            num_tokens_list, num_q_heads_list, head_size_list, 
            rotary_mode_list, cache_mode_list, dtype_list
        ):
            num_kv_heads = num_q_heads
            max_seq_len = num_tokens
            rotary_dim = head_size
            positions = torch.arange(num_tokens, dtype=torch.int64)
            query = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype)
            key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=dtype)
            cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=dtype)

            positions_npu = positions.npu()
            query_npu = query.npu()
            key_npu = key.npu()
            cos_sin_cache_npu = cos_sin_cache.npu()

            if dtype == torch.float16 or dtype == torch.bfloat16:
                golden_dtype = torch.float32
            else:
                golden_dtype = torch.float64

            query = query.to(golden_dtype)
            key = key.to(golden_dtype)
            cos_sin_cache = cos_sin_cache.to(golden_dtype)
            mrope_section = [0, 0, 0]

            query_out, key_out = torch_npu.npu_mrope(
                positions_npu,
                query_npu,
                key_npu,
                cos_sin_cache_npu,
                head_size,
                mrope_section=mrope_section,
                rotary_mode=rotary_mode,
                cache_mode=cache_mode,
            )
            if rotary_mode == 'half':
                is_neox_style = True
            else:
                is_neox_style = False

            expected_query_out, expected_key_out = self.golden_mrope(
                positions,
                query,
                key,
                cos_sin_cache,
                mrope_section,
                head_size,
                is_neox_style,
                cache_mode,
            )

            self.assertRtolEqual(expected_query_out.to(dtype), query_out)
            self.assertRtolEqual(expected_key_out.to(dtype), key_out)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B"])
    def test_npu_mrope_mrope(self):
        mrope_section = [16, 24, 24]
        num_tokens_list = [8, 16]
        num_q_heads_list = [8, 16]
        head_size_list = [128]
        rotary_mode_list = ['half', 'interleave']
        cache_mode_list = ['default', 'interleave']
        dtype_list = [torch.bfloat16, torch.float16, torch.float32]

        for (
                num_tokens,
                num_q_heads,
                head_size,
                rotary_mode,
                cache_mode,
                dtype,
        ) in itertools.product(
            num_tokens_list, num_q_heads_list, head_size_list, 
            rotary_mode_list, cache_mode_list, dtype_list
        ):
            num_kv_heads = num_q_heads
            max_seq_len = num_tokens
            rotary_dim = head_size
            positions = torch.arange(num_tokens, dtype=torch.int64).repeat(3, 1)
            query = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype)
            key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=dtype)
            cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=dtype)

            positions_npu = positions.npu()
            query_npu = query.npu()
            key_npu = key.npu()
            cos_sin_cache_npu = cos_sin_cache.npu()

            if dtype == torch.float16 or dtype == torch.bfloat16:
                golden_dtype = torch.float32
            else:
                golden_dtype = torch.float64

            query = query.to(golden_dtype)
            key = key.to(golden_dtype)
            cos_sin_cache = cos_sin_cache.to(golden_dtype)
            
            query_out, key_out = torch_npu.npu_mrope(
                positions_npu,
                query_npu,
                key_npu,
                cos_sin_cache_npu,
                head_size,
                mrope_section=mrope_section,
                rotary_mode=rotary_mode,
                cache_mode=cache_mode,
            )
            if rotary_mode == 'half':
                is_neox_style = True
            else:
                is_neox_style = False
            expected_query_out, expected_key_out = self.golden_mrope(
                positions,
                query,
                key,
                cos_sin_cache,
                mrope_section,
                head_size,
                is_neox_style,
                cache_mode,
            )

            self.assertRtolEqual(expected_query_out.to(dtype), query_out)
            self.assertRtolEqual(expected_key_out.to(dtype), key_out)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B"])
    def test_npu_mrope_error_cache_mode(self):
        num_tokens = 8
        num_q_heads = 8
        head_size = 128
        positions = torch.arange(num_tokens, dtype=torch.int64).npu()
        query = torch.randn(num_tokens, num_q_heads * head_size, dtype=torch.float16).npu()
        key = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float16).npu()
        cos_sin_cache = torch.rand(num_tokens, head_size, dtype=torch.float16).npu()
        mrope_section = [0, 0, 0]

        msg = "cache_mode only support default or interleave"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch_npu.npu_mrope(
                positions,
                query,
                key,
                cos_sin_cache,
                head_size,
                mrope_section=mrope_section,
                rotary_mode='half',
                cache_mode='invalid_mode',
            )


if __name__ == "__main__":
    run_tests()

