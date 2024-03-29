import random
import unittest
import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class TestNPURmsNorm(TestCase):

    def supported_op_exec(self, sorted_experts, num_expert):
        arr_length = sorted_experts.shape[-1]
        res = np.arange(num_expert)
        for i in range(num_expert):
            target = i
            low = 0
            high = arr_length - 1
            target_location = -1
            while low <= high:
                mid = (low + high) // 2
                if sorted_experts[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
                    target_location = mid
            res[i] = target_location + 1
        return res

    def custom_op_exec(self, sorted_experts, num_expert):
        res_npu = torch_npu.npu_moe_compute_expert_tokens(sorted_experts, num_expert)
        return res_npu.cpu().numpy()

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_compute_expert_tokens(self, device="npu"):
        if device is None:
            device = get_npu_device()
        # gene input data
        sorted_expert_len = 302
        num_expert = 31
        random_int_list = []
        for _ in range(sorted_expert_len):
            random_int_list.append(random.randint(0, num_expert))
        sorted_experts = np.sort(random_int_list).astype(np.int32)

        sorted_experts_npu = torch.from_numpy(sorted_experts).to(device)

        res_cpu = self.supported_op_exec(sorted_experts, num_expert)
        res_npu = self.custom_op_exec(sorted_experts_npu, num_expert)
        self.assertRtolEqual(res_cpu.astype(np.float32), res_npu.astype(np.float32))

if __name__ == "__main__":
    run_tests()
