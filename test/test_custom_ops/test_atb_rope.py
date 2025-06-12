import unittest
import numpy as np
import torch
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestRope(TestCase):
    rope_theta = 10000
    head_dim = 128
    num_heads = 32
    max_position_embeddings = 8192
    batch_size = 1
    seq_length = 4

    def compute_inv_freq(self, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
        return inv_freq

    def compute_cos_sin_cache(self):
        inv_freq = self.compute_inv_freq(self.rope_theta)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache.to('npu')

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.stack((o1, o2), dim=-1).flatten(-2)

    def native_rope(self, positions, query, key):
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.compute_cos_sin_cache().index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query_rot = query[..., :self.head_dim]
        query_remainder = query[..., self.head_dim:]
        query_rot = self._apply_rotary_emb(query_rot, cos, sin)
        query = torch.cat((query_rot, query_remainder), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key_rot = key[..., :self.head_dim]
        key_remainder = key[..., self.head_dim:]
        key_rot = self._apply_rotary_emb(key_rot, cos, sin)
        key = torch.cat((key_rot, key_remainder), dim=-1).reshape(key_shape)
        return query, key

    @SupportedDevices(['Ascend910B'])
    def test_rope(self):
        cos_sin_cache = self.compute_cos_sin_cache()
        positions = torch.arange(self.seq_length).repeat(self.batch_size).npu()
        query = torch.rand(self.batch_size * self.seq_length, self.num_heads * self.head_dim, dtype=torch.float16).npu()
        key = torch.rand(self.batch_size * self.seq_length, self.num_heads * self.head_dim, dtype=torch.float16).npu()
        is_neox = False
        expected_query, expected_key = self.native_rope(positions, query, key)
        torch_npu._npu_rotary_embedding(positions, query, key, self.head_dim, cos_sin_cache, is_neox)

        self.assertRtolEqual(expected_query, query)
        self.assertRtolEqual(expected_key, key)

    @unittest.skipIf(torch.__version__ < '2.5.1', "This compile ut needs torch version >=2.5.1")
    @SupportedDevices(['Ascend910B'])
    def test_rope_compile(self):
        class RopeModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, positions, query, key, head_dim, cos_sin_cache, is_neox):
                torch_npu._npu_rotary_embedding(positions, query, key, head_dim, cos_sin_cache, is_neox)
                return query, key
        cos_sin_cache = self.compute_cos_sin_cache()
        positions = torch.arange(self.seq_length).repeat(self.batch_size).npu()
        query = torch.rand(self.batch_size * self.seq_length, self.num_heads * self.head_dim, dtype=torch.float16).npu()
        key = torch.rand(self.batch_size * self.seq_length, self.num_heads * self.head_dim, dtype=torch.float16).npu()
        query1, key1 = query.clone(), key.clone()
        is_neox = False
        model = RopeModel()
        compiled_model = torch.compile(
            model,
            backend="aot_eager",
            fullgraph=True,
        )
        compiled_output = compiled_model(positions, query, key, self.head_dim, cos_sin_cache, is_neox)
        torch_npu._npu_rotary_embedding(positions, query1, key1, self.head_dim, cos_sin_cache, is_neox)
        self.assertRtolEqual(compiled_output[0], query1)
        self.assertRtolEqual(compiled_output[1], key1)


if __name__ == '__main__':
    run_tests()
