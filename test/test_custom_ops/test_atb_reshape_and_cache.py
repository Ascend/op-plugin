import random
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestReshapeAndCache(TestCase):
    num_tokens = 16
    num_head = 16
    head_size = 16
    block_size = 16
    num_blocks = 8

    def cal(self, key, value, key_cache, value_cache, slot_mapping):
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
        key_expect, value_expect = self.cal(key, value, key_cache, value_cache, slot_mapping)
        key = key.npu()
        value = value.npu()
        key_cache = key_cache.npu()
        value_cache = value_cache.npu()
        slot_mapping = torch.from_numpy(slot_mapping).to(torch.int32).npu()
        torch_npu._npu_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        self.assertRtolEqual(key_expect, key_cache)
        self.assertRtolEqual(value_expect, value_cache)

if __name__ == '__main__':
    run_tests()
