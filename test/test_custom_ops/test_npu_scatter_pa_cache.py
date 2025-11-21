import unittest
import copy
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

class TestScatterPaCache(TestCase):

    def supported_op_exec(self, key, keyCache, slotMapping):
        key_shape = key.shape
        num_tokens = key_shape[0]
        num_heads = key_shape[1]
        head_size_k = key_shape[2]
        key_cache_shape = keyCache.shape
        key_cache_out = copy.deepcopy(keyCache)
        for i in range(num_tokens):
            block_idx = slotMapping[i]
            block_offset = slotMapping[i]
            key_cache_out[block_idx, block_offset, :, :] = key[i, :, :]
        return key_cache_out

    def custom_op_exec(self, key, keyCache, slotMapping):
        return torch_npu.npu_scatter_pa_cache(key, slotMapping, key_cache=keyCache)

    @SupportedDevices(['Ascend910_95'])
    def test_npu_scatter_pa_cache(self, device="npu"):
        key = torch.randint(-1, 1, (256, 16, 16), dtype=torch.float32).npu()
        keyCache = torch.randint(-1, 1, (16, 16, 16, 16), dtype=torch.float32).npu()
        slotMapping = torch.arange(0, 256).view(256).to(torch.int32).npu()
        supported_output =  self.supported_op_exec(key, keyCache, slotMapping)
        custom_output = self.custom_op_exec(key, keyCache, slotMapping)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()