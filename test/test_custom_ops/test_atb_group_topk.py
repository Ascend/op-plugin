import unittest
import numpy as nps
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestGroupTopk(TestCase):
    def generalize_param(self):
        token_nums = [1, 16, 33]
        expert_nums = [1, 4, 15, 16, 17, 32, 65, 128, 257, 1024]
        k_params = [1, 4, 15, 16, 17, 32, 65, 257, 1024]
        k_inner_params = [1, 2, 3, 4, 16, 32, 65, 1024]

        expert_num_groups = []
        for expert_num in expert_nums:
            factors = set()
            for i in range(1, int(expert_num**0.5) + 1):
                if expert_num % i == 0:
                    factors.add(i)
                    factors.add(int(expert_num / i))
            expert_num_groups.append(list(factors))

        for token_num in token_nums:
            for expert_num, group_nums in zip(expert_nums, expert_num_groups):
                for group_num in group_nums:
                    for k in k_params:
                        for k_inner in k_inner_params:
                            if k > group_num or k_inner > expert_num // group_num:
                                continue
                            yield token_num, expert_num, group_num, k, k_inner

    def golden_calc(self, input0, k, group_num, k_inner):
        token_num, expert_num = input0.shape
        input0 = torch.reshape(input0, (token_num, group_num, expert_num // group_num))
        output = input0.clone()
        input0 = input0.to(torch.float)
        group_tensor = torch.topk(input0, k_inner).values
        group_tensor = torch.sum(group_tensor, dim=-1)
        sort_index = torch.from_numpy(np.argsort(-group_tensor.numpy(), kind='stable'))
        cols_to_use = torch.arange(k, group_num, dtype=torch.long)
        row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(cols_to_use.shape[0])
        col_indices = sort_index.index_select(1, cols_to_use).view(-1)
        output[row_indices, col_indices] = 0
        return [torch.reshape(output, (token_num, expert_num))]

    @unittest.skip("skip test_group_topk now")
    @SupportedDevices(['Ascend910B'])
    def test_group_topk(self):
        for dtype in [torch.float16, torch.bfloat16]:
            for token_num, expert_num, group_num, k, k_inner in self.generalize_param():
                input0 = torch.empty((token_num, expert_num), dtype=dtype, device='npu').uniform_(-2, 2)
                output0 = torch.randn((token_num, expert_num), dtype=dtype, device='npu')
                expect_output = self.golden_calc(input0.cpu(), k, group_num, k_inner)
                torch_npu._npu_group_topk(input0, k=k, group_num=group_num, n=k_inner)

                self.assertRtolEqual(input0, expect_output[0])


if __name__ == "__main__":
    run_tests()
