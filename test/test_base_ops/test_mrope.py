import itertools
import unittest

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import run_tests, TestCase


@unittest.skip("CANN is not ready, skipping MROPE test case for now")
class TestMRope(TestCase):
    # pylint:disable=huawei-too-many-arguments
    def golden_mrope(
            self,
            positions,
            query,
            key,
            cos_sin_cache,
            mrope_section,
            head_size,
            num_q_heads,
            num_kv_heads,
            is_neox_style=True,
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
            cos_0 = cos[0, :, : mrope_section[0]]
            cos_1 = cos[1, :, mrope_section[0]: (mrope_section[0] + mrope_section[1])]
            cos_2 = cos[2, :, (mrope_section[0] + mrope_section[1]):
                (mrope_section[0] + mrope_section[1] + mrope_section[2]), ]
            cos = torch.concat((cos_0, cos_1, cos_2), dim=-1)

            sin = sin.reshape(
                3, -1, rotary_dim // 2
            )  # [3, num_tokens, rotary_dim // 2]
            sin_0 = sin[0, :, : mrope_section[0]]
            sin_1 = sin[1, :, mrope_section[0]: (mrope_section[0] + mrope_section[1])]
            sin_2 = sin[2, :, (mrope_section[0] + mrope_section[1]):
                (mrope_section[0] + mrope_section[1] + mrope_section[2]), ]
            sin = torch.concat((sin_0, sin_1, sin_2), dim=-1)

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

    @SupportedDevices(["Ascend910B"])
    def test_rope(self):
        num_tokens_list = [8, 16, 48]
        num_q_heads_list = [8, 16, 30]
        head_size_list = [128]
        rotary_mode = ['half', 'interleave']
        dtype_list = [torch.bfloat16, torch.float16, torch.float32]

        for (
                num_tokens,
                num_q_heads,
                head_size,
                rotary_mode,
                dtype,
        ) in itertools.product(
            num_tokens_list, num_q_heads_list, head_size_list, rotary_mode, dtype_list
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
                num_q_heads,
                num_kv_heads,
                is_neox_style,
            )

            self.assertRtolEqual(expected_query_out.to(dtype), query_out)
            self.assertRtolEqual(expected_key_out.to(dtype), key_out)

    @SupportedDevices(["Ascend910B"])
    def test_mrope(self):
        mrope_section = [16, 24, 24]
        num_tokens_list = [8, 16, 48]
        num_q_heads_list = [8, 16, 30]
        head_size_list = [128]
        rotary_mode = ['half', 'interleave']
        dtype_list = [torch.bfloat16, torch.float16, torch.float32]

        for (
                num_tokens,
                num_q_heads,
                head_size,
                rotary_mode,
                dtype,
        ) in itertools.product(
            num_tokens_list, num_q_heads_list, head_size_list, rotary_mode, dtype_list
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
                num_q_heads,
                num_kv_heads,
                is_neox_style,
            )

            self.assertRtolEqual(expected_query_out.to(dtype), query_out)
            self.assertRtolEqual(expected_key_out.to(dtype), key_out)


if __name__ == "__main__":
    run_tests()
