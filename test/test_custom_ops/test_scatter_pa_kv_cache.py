import copy
import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestScatterPaKvCache(TestCase):
    def supported_op_exec(self, data, data_info):
        key = data[0]
        value = data[1]
        key_cache = data[2]
        value_cache = data[3]
        slot_mapping = data[4]
        block_size = data_info[0]
        num_head = data_info[1]
        k_head_size = data_info[2]
        v_head_size = data_info[3]
        lastDim_k = data_info[4]

        key_cache_golden = copy.deepcopy(key_cache)
        value_cache_golden = copy.deepcopy(value_cache)
        for i, slot in enumerate(slot_mapping):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size

            token_key = key[i].reshape(num_head * k_head_size)
            for k in range(num_head * k_head_size // lastDim_k):
                key_cache_golden[block_index][k][block_offset][:] = \
                    token_key[k * lastDim_k: k * lastDim_k + lastDim_k]

            token_value = value[i].reshape(num_head * v_head_size)
            for v in range(num_head * v_head_size // lastDim_k):
                value_cache_golden[block_index][v][block_offset][:] = \
                    token_value[v * lastDim_k: v * lastDim_k + lastDim_k]

        return key_cache_golden, value_cache_golden


    def custom_op_exec(self, data_npu):
        key_npu = data_npu[0]
        value_npu = data_npu[1]
        slot_mapping_npu = data_npu[2]
        key_cache_npu = data_npu[3]
        value_cache_npu = data_npu[4]
        torch_npu.npu_scatter_pa_kv_cache(key_npu, value_npu, key_cache_npu, value_cache_npu, slot_mapping_npu)


    def _custom_test(self, bs, num_blocks, data_info):
        block_size = data_info[0]
        num_head = data_info[1]
        k_head_size = data_info[2]
        v_head_size = data_info[3]
        lastDim_k = data_info[4]

        key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
        value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
        key_cache = np.random.randn(
            num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
        value_cache = np.zeros(
            (num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
        slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)

        key_npu = torch.from_numpy(key).npu()
        value_npu = torch.from_numpy(value).npu()
        key_cache_npu = torch.from_numpy(key_cache).npu()
        value_cache_npu = torch.from_numpy(value_cache).npu()
        key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
        value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
        slot_mapping_npu = torch.from_numpy(slot_mapping).npu()

        key_cache_golden, value_cache_golden = \
            self.supported_op_exec([key, value, key_cache, value_cache, slot_mapping],
                                   [block_size, num_head, k_head_size, v_head_size, lastDim_k])

        self.custom_op_exec([key_npu, value_npu, slot_mapping_npu, key_cache_npu_cast, value_cache_npu_cast])

        key_cache_golden_npu = torch.from_numpy(key_cache_golden).npu()
        value_cache_golden_npu = torch.from_numpy(value_cache_golden).npu()

        return key_cache_npu_cast, value_cache_npu_cast, key_cache_golden_npu, value_cache_golden_npu


    @unittest.skip("skip until CANN is updated to support aclnnScatterPaKvCache")
    @SupportedDevices(['Ascend910B'])
    def test_npu_scatter_pa_kv_cache_1(self, device="npu"):
        bs = 16
        num_head = 4
        k_head_size = 32
        v_head_size = 32
        num_blocks = 2
        lastDim_k = 16
        block_size = 32

        key_cache_npu_cast, value_cache_npu_cast, key_cache_golden_npu, value_cache_golden_npu = \
            self._custom_test(bs, num_blocks, [block_size, num_head, k_head_size, v_head_size, lastDim_k])

        self.assertRtolEqual(key_cache_npu_cast, key_cache_golden_npu)
        self.assertRtolEqual(value_cache_npu_cast, value_cache_golden_npu)


    @unittest.skip("skip until CANN is updated to support aclnnScatterPaKvCache")
    @SupportedDevices(['Ascend910B'])
    def test_npu_scatter_pa_kv_cache_2(self, device="npu"):
        bs = 16
        num_head = 4
        k_head_size = 32
        v_head_size = 64
        num_blocks = 2
        lastDim_k = 16
        block_size = 32

        key_cache_npu_cast, value_cache_npu_cast, key_cache_golden_npu, value_cache_golden_npu = \
            self._custom_test(bs, num_blocks, [block_size, num_head, k_head_size, v_head_size, lastDim_k])

        self.assertRtolEqual(key_cache_npu_cast, key_cache_golden_npu)
        self.assertRtolEqual(value_cache_npu_cast, value_cache_golden_npu)


    @unittest.skip("skip until CANN is updated to support aclnnScatterPaKvCache")
    @SupportedDevices(['Ascend910B'])
    def test_npu_scatter_pa_kv_cache_3(self, device="npu"):
        bs = 16
        num_head = 4
        k_head_size = 32
        v_head_size = 64
        num_blocks = 2
        lastDim_k = 16
        block_size = 64

        key_cache_npu_cast, value_cache_npu_cast, key_cache_golden_npu, value_cache_golden_npu = \
            self._custom_test(bs, num_blocks, [block_size, num_head, k_head_size, v_head_size, lastDim_k])
        
        self.assertRtolEqual(key_cache_npu_cast, key_cache_golden_npu)
        self.assertRtolEqual(value_cache_npu_cast, value_cache_golden_npu)


    @unittest.skip("skip until CANN is updated to support aclnnScatterPaKvCache")
    @SupportedDevices(['Ascend910B'])
    def test_npu_scatter_pa_kv_cache_4(self, device="npu"):
        bs = 16
        num_head = 4
        k_head_size = 32
        v_head_size = 64
        num_blocks = 2
        lastDim_k = 16
        block_size = 128

        key_cache_npu_cast, value_cache_npu_cast, key_cache_golden_npu, value_cache_golden_npu = \
            self._custom_test(bs, num_blocks, [block_size, num_head, k_head_size, v_head_size, lastDim_k])

        self.assertRtolEqual(key_cache_npu_cast, key_cache_golden_npu)
        self.assertRtolEqual(value_cache_npu_cast, value_cache_golden_npu)


if __name__ == "__main__":
    run_tests()
