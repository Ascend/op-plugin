import random
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestReshapeAndCache(TestCase):
    num_tokens = 14
    num_head = 1
    head_size = 16
    block_size = 16
    num_blocks = 53535

    def cal_nd(self, key, value, key_cache, value_cache, slot_mapping):
        key_expect = key_cache.clone()
        value_expect = value_cache.clone()
        for i, slot in enumerate(slot_mapping):
            if slot < 0:
                continue
            block_index = slot // self.block_size
            block_offset = slot % self.block_size
            token_key = key[i]
            token_v = value[i]
            key_expect[block_index][block_offset] = token_key
            value_expect[block_index][block_offset] = token_v
        return key_expect, value_expect

    def cal_nz(self, key, value, key_cache, value_cache, slot_mapping):
        key_expect_nz = key_cache.clone()
        value_expect_nz = value_cache.clone()
        data_type = key.dtype
        k_head_size = key.shape[2]
        v_head_size = value.shape[2]
        last_dim_k = 0
        last_dim_v = 16
        if data_type == torch.int8:
            last_dim_k = 32
        else:
            last_dim_k = 16
        num_blocks, _, block_size, _ = key_cache.shape
        value_expect_nz = value_cache
        for i, slot in enumerate(slot_mapping):
            block_index = slot // block_size
            block_offset = slot % block_size

            token_key = key[i]
            token_v = value[i]
            num_head = self.num_head
            token_key = token_key.reshape(num_head * k_head_size)
            token_v = token_v.reshape(num_head * v_head_size)
            for k in range(num_head * k_head_size // last_dim_k):
                key_expect_nz[block_index][k][block_offset][:] = token_key[k * last_dim_k: k * last_dim_k + last_dim_k]
            for v in range(num_head * v_head_size // last_dim_v):
                value_expect_nz[block_index][v][block_offset][:] = token_v[v * last_dim_v: v * last_dim_v + last_dim_v]
        return [key_expect_nz, value_expect_nz]

    @SupportedDevices(['Ascend910B'])
    def test_reshape_and_cache(self):
        head_size_k = np.random.randint(1, 256)
        head_size_v = np.random.randint(1, 256)
        key = torch.rand((self.num_tokens, self.num_head, head_size_k), dtype=torch.float16)
        value = torch.rand((self.num_tokens, self.num_head, head_size_v), dtype=torch.float16)
        num_slots = self.block_size * self.num_blocks
        slot_list = random.sample(range(num_slots), self.num_tokens)
        slot_mapping = np.array(slot_list).astype(np.int32)
        key_cache = torch.rand((self.num_blocks, self.block_size, self.num_head, head_size_k), dtype=torch.float16)
        value_cache = torch.rand((self.num_blocks, self.block_size, self.num_head, head_size_v), dtype=torch.float16)
        key_expect, value_expect = self.cal_nd(key, value, key_cache, value_cache, slot_mapping)
        key = key.npu()
        value = value.npu()
        key_cache = key_cache.npu()
        value_cache = value_cache.npu()
        slot_mapping = torch.from_numpy(slot_mapping).to(torch.int32).npu()
        torch_npu._npu_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        self.assertRtolEqual(key_expect, key_cache)
        self.assertRtolEqual(value_expect, value_cache)

    @SupportedDevices(['Ascend910B'])
    def test_reshape_and_cache_int8(self):
        head_size_k = 512
        head_size_v = 64
        key = torch.randint(-128, 128, (self.num_tokens, self.num_head, head_size_k), dtype=torch.int8)
        value = torch.rand((self.num_tokens, self.num_head, head_size_v), dtype=torch.bfloat16)
        num_slots = self.block_size * self.num_blocks
        slot_list = random.sample(range(num_slots), self.num_tokens)
        slot_mapping = np.array(slot_list).astype(np.int32)
        key_cache = torch.randint(-128, 128, (self.num_blocks, 16, 128, 32), dtype=torch.int8)
        value_cache = torch.rand((self.num_blocks, 4, 128, 16), dtype=torch.bfloat16)

        key_expect_nz, value_expect_nz = self.cal_nz(key, value, key_cache, value_cache, slot_mapping)
        key_cache_nz = torch_npu.npu_format_cast(key_cache.npu(), 29)
        value_cache_nz = torch_npu.npu_format_cast(value_cache.npu(), 29)
        key = key.npu()
        value = value.npu()
        key_cache = key_cache.npu()
        value_cache = value_cache.npu()
        key_cache = key_cache_nz.npu()
        value_cache = value_cache_nz.npu()
        slot_mapping = torch.from_numpy(slot_mapping).to(torch.int32).npu()
        torch_npu._npu_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        self.assertRtolEqual(key_expect_nz, key_cache)
        self.assertRtolEqual(value_expect_nz, value_cache)

if __name__ == '__main__':
    run_tests()
