import itertools
import unittest
from dataclasses import dataclass
import math
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


@dataclass
class GoldenCompareParams:
    x: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    k_cache_ref: torch.Tensor
    v_cache_ref: torch.Tensor
    indices: torch.Tensor
    scale_k: torch.Tensor
    scale_v: torch.Tensor
    size_splits: list
    offset_k_optional: torch.Tensor
    offset_v_optional: torch.Tensor


@dataclass
class UpdateScatterParams:
    key_cache: torch.Tensor
    key: torch.Tensor
    scale: torch.Tensor
    indice: torch.Tensor
    offset: torch.Tensor


class TestNpuRopeQuantKVCache(TestCase):

    def quant_update_scatter(self, params: UpdateScatterParams, ifpa=False):
        key_cache = params.key_cache
        key = params.key
        scale = params.scale
        indice = params.indice
        offset = params.offset

        # quant
        quant_out = []
        if offset is not None:
            quant_out = key.float() * scale + offset
        else:
            quant_out = key.float() * scale

        quant_out = quant_out.round()
        d0 = key_cache.shape[0]
        d1 = key_cache.shape[1]
        d2 = key_cache.shape[2]
        d3 = key_cache.shape[3]

        # scatter
        quant_out1 = torch.clamp(torch.round(quant_out.float()), min=0, max=10)
        quant_out2 = quant_out1.reshape(-1, quant_out1.shape[-2], quant_out1.shape[-1])
        if ifpa:
            key_cachepa = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])

            for b in range(indice.shape[0]):
                indice_value = indice[b]
                key_cachepa[indice_value] = quant_out2[b]
            key_cache = key_cachepa.reshape([d0, d1, d2, d3])
        else:
            for b in range(indice.shape[0]):
                indice_value = indice[b]
                key_cache[b][indice_value: indice_value + quant_out.shape[1]][:][:] = quant_out1[b][:][:][:].reshape(
                    key_cache[b][indice_value: indice_value + quant_out.shape[1]][:][:].shape)

    def rope(self, x, cos, sin):
        d = x.shape[-1]
        rotary_x = torch.concat((-x[..., d // 2:], x[..., : d // 2]), dim=-1)
        return x * cos + rotary_x * sin

    def golden_compare(self, params: GoldenCompareParams):
        x = params.x
        cos = params.cos
        sin = params.sin
        k_cache_ref = params.k_cache_ref
        v_cache_ref = params.v_cache_ref
        indices = params.indices
        scale_k = params.scale_k
        scale_v = params.scale_v
        size_splits = params.size_splits
        offset_k_optional = params.offset_k_optional
        offset_v_optional = params.offset_v_optional
        h = cos.shape[-1]
        b = x.shape[0]
        s = x.shape[1]
        q_headdim = size_splits[0] // cos.shape[-1]
        kv_headdim = k_cache_ref.shape[-2]
        q, k, v = x.split(size_splits, dim=-1)
        q = q.reshape([b, s, q_headdim, h])
        k = k.reshape([b, s, kv_headdim, h])
        v = v.reshape([b, s, kv_headdim, h])
        ropek = self.rope(k, cos, sin)
        ropeq = self.rope(q, cos, sin)

        k_scatter_params = UpdateScatterParams(k_cache_ref, ropek, scale_k, indices, offset_k_optional)
        v_scatter_params = UpdateScatterParams(v_cache_ref, v, scale_v, indices, offset_v_optional)

        self.quant_update_scatter(k_scatter_params)
        self.quant_update_scatter(v_scatter_params)

        if v.dtype == torch.bfloat16:
            v = v.to(torch.float32)
        return (
            ropeq.to(torch.float16).cpu().numpy(),
            ropek.to(torch.float16).cpu().numpy(),
            v.cpu().numpy(),
            k_cache_ref.cpu().numpy(),
            v_cache_ref.cpu().numpy(),
        )

    @unittest.skip("skip test_npu_rope_quant_kvcache_1 now")
    @SupportedDevices(["Ascend910B"])
    def test_npu_rope_quant_kvcache_1(self):
        in_x = torch.randn([1, 1, 128 * 3]).to(torch.bfloat16).npu()
        in_cos = torch.randn([1, 1, 1, 128]).to(torch.bfloat16).npu()
        in_sin = torch.randn([1, 1, 1, 128]).to(torch.bfloat16).npu()
        data_k_cache = np.random.uniform(0, 1, [1, 2, 1, 128]).astype(np.int8)
        in_k_cache = torch.from_numpy(data_k_cache).to(torch.int8).npu()
        data_v_cache = np.random.uniform(0, 1, [1, 2, 1, 128]).astype(np.int8)
        in_v_cache = torch.from_numpy(data_v_cache).to(torch.int8).npu()
        in_indices = torch.tensor([0]).to(torch.int32).npu()
        in_scale_k = torch.randn([128], dtype=torch.float32).npu()
        in_scale_v = torch.randn([128], dtype=torch.float32).npu()
        in_offset_k = torch.randn([128], dtype=torch.float32).npu()
        in_offset_v = torch.randn([128], dtype=torch.float32).npu()
        size_splits = [128, 128, 128]
        nq, nk, nv, nkc, nvc = torch_npu.npu_rope_quant_kvcache(
            in_x,
            in_cos,
            in_sin,
            in_k_cache,
            in_v_cache,
            in_indices,
            in_scale_k,
            in_scale_v,
            size_splits,
            offset_k=in_offset_k,
            offset_v=in_offset_v,
        )

        params = GoldenCompareParams(
            in_x.to(torch.float32),
            in_cos.to(torch.float32),
            in_sin.to(torch.float32),
            in_k_cache,
            in_v_cache,
            in_indices,
            in_scale_k,
            in_scale_v,
            size_splits,
            in_offset_k,
            in_offset_v,
        )
        q, k, v, kc, vc = self.golden_compare(params)

        self.assertRtolEqual(q, nq.to(torch.float16).cpu().numpy(), prec16=1e-2)
        self.assertRtolEqual(vc, nvc.cpu().numpy(), prec16=1e-1)
        self.assertRtolEqual(kc, nkc.cpu().numpy(), prec16=1e-1)


    @unittest.skip("skip test_npu_rope_quant_kvcache_2 now")
    @SupportedDevices(["Ascend910B"])
    def test_npu_rope_quant_kvcache_2(self):
        in_x = torch.randn([1, 1, 128 * 4]).to(torch.float16).npu()
        in_cos = torch.randn([1, 1, 1, 128]).to(torch.float16).npu()
        in_sin = torch.randn([1, 1, 1, 128]).to(torch.float16).npu()
        data_k_cache = np.random.uniform(0, 1, [1, 2, 1, 128]).astype(np.int8)
        in_k_cache = torch.from_numpy(data_k_cache).to(torch.int8).npu()
        data_v_cache = np.random.uniform(0, 1, [1, 2, 1, 128]).astype(np.int8)
        in_v_cache = torch.from_numpy(data_v_cache).to(torch.int8).npu()
        in_indices = torch.tensor([0]).to(torch.int32).npu()
        in_scale_k = torch.randn([128], dtype=torch.float32).npu()
        in_scale_v = torch.randn([128], dtype=torch.float32).npu()
        in_offset_k = torch.randn([128], dtype=torch.float32).npu()
        in_offset_v = torch.randn([128], dtype=torch.float32).npu()
        size_splits = [128 * 2, 128, 128]
        nq, nk, nv, nkc, nvc = torch_npu.npu_rope_quant_kvcache(
            in_x,
            in_cos,
            in_sin,
            in_k_cache,
            in_v_cache,
            in_indices,
            in_scale_k,
            in_scale_v,
            size_splits,
            offset_k=in_offset_k,
            offset_v=in_offset_v,
        )

        params = GoldenCompareParams(
            in_x,
            in_cos,
            in_sin,
            in_k_cache,
            in_v_cache,
            in_indices,
            in_scale_k,
            in_scale_v,
            size_splits,
            in_offset_k,
            in_offset_v,
        )
        q, k, v, kc, vc = self.golden_compare(params)

        self.assertRtolEqual(q, nq.cpu().numpy(), prec16=1e-1)
        self.assertRtolEqual(vc, nvc.cpu().numpy(), prec16=1e-1)
        self.assertRtolEqual(kc, nkc.cpu().numpy(), prec16=1e-1)


if __name__ == "__main__":
    run_tests()
