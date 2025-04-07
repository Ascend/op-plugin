import itertools
import unittest
import numpy
import math
import random
import copy
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices

class TestNpuMoeReRouting(TestCase):
    def moe_re_routing_torch(self, tokens: torch.tensor, expert_token_num_per_rank: torch.tensor,
                             *, per_token_scales: torch.tensor = None, expert_token_num_type: int = 1,
                             idx_type: int = 0) -> tuple:
        token_num, token_length = tokens.shape
        rank_num, expert_num = expert_token_num_per_rank.shape

        per_expert_offset = torch.zeros(expert_num, dtype=torch.int32)
        expert_token_num = torch.zeros(expert_num, dtype=torch.int32)

        # 计算每个expert的总token数
        expert_token_num = torch.sum(expert_token_num_per_rank, dim=0, dtype=expert_token_num_per_rank.dtype)
        # 计算per_expert_offset
        per_expert_offset[1:] = torch.cumsum(expert_token_num[:-1], dim=0)

        permute_tokens = torch.zeros((token_num, token_length), dtype=tokens.dtype, device=tokens.device)
        permute_token_idx = torch.zeros(token_num, dtype=torch.int32, device=tokens.device)
        if per_token_scales is not None:
            permute_per_token_scales = torch.zeros(token_num, dtype=per_token_scales.dtype, device=per_token_scales.device)
        else:
            permute_per_token_scales = None

        src_offset = 0
        for cur_rank in range(rank_num):
            for expert in range(expert_num):
                num_tokens = expert_token_num_per_rank[cur_rank, expert]
                if num_tokens == 0:
                    continue
                dst_start = per_expert_offset[expert]
                dst_end = dst_start + num_tokens

                permute_tokens[dst_start:dst_end] = tokens[src_offset:src_offset + num_tokens]
                permute_token_idx[dst_start:dst_end] = torch.arange(src_offset, src_offset + num_tokens, dtype=torch.int32, device=permute_token_idx.device)

                if per_token_scales is not None:
                    permute_per_token_scales[dst_start:dst_end] = per_token_scales[src_offset:src_offset + num_tokens]

                src_offset += num_tokens
                per_expert_offset[expert] += num_tokens

        return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num
    
    @unittest.skip("skip test_moe_re_routing_1 now")
    @SupportedDevices(['Ascend910B'])
    def test_moe_re_routing_1(self, device="npu"):
        maybe_tokens_num_list = [16384, 10000, 28888, 32768]
        tokens_num = random.choice(maybe_tokens_num_list)
        tokens_length = 7 * 1024
        rank_num = 32
        expert_num = 8
        tokens = torch.randint(low=0, high=20, size=(tokens_num, tokens_length), dtype=torch.int8)
        expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype = torch.int32)
        per_token_scales = torch.randn(tokens_num, dtype = torch.float32)

        tokens_sum = 0
        for i in range(rank_num):
            for j in range(expert_num):
                if i == rank_num - 1 and j == expert_num - 1:
                    expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
                    break
                floor = math.floor(tokens_num / (rank_num * expert_num))
                rand_num = random.randint(1, floor)
                expert_token_num_per_rank[i][j] = rand_num
                tokens_sum += rand_num

        expert_token_num_type = 1
        idx_type = 0

        permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = self.moe_re_routing_torch(
            tokens,
            expert_token_num_per_rank,
            per_token_scales=per_token_scales,
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type)

        permute_tokens_npu, permute_per_token_scales_npu, permute_token_idx_npu, expert_token_num_npu = torch_npu.npu_moe_re_routing(
            tokens.npu(),
            expert_token_num_per_rank.npu(),
            per_token_scales=per_token_scales.npu(),
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type)

        self.assertRtolEqual(permute_tokens, permute_tokens_npu.cpu())
        self.assertRtolEqual(permute_per_token_scales, permute_per_token_scales_npu.cpu())
        self.assertRtolEqual(permute_token_idx, permute_token_idx_npu.cpu())
        self.assertRtolEqual(expert_token_num, expert_token_num_npu.cpu())

    @unittest.skip("skip test_moe_re_routing_2 now")
    @SupportedDevices(['Ascend910B'])
    def test_moe_re_routing_2(self, device="npu"):
        maybe_tokens_num_list = [16384, 10000, 28888, 32768]
        tokens_num = random.choice(maybe_tokens_num_list)
        tokens_length = 7 * 1024
        rank_num = 16
        expert_num = 16
        tokens = torch.randint(low=0, high=20, size=(tokens_num, tokens_length), dtype=torch.bfloat16)
        expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype = torch.int64)
        per_token_scales = torch.randn(tokens_num, dtype = torch.float32)

        tokens_sum = 0
        for i in range(rank_num):
            for j in range(expert_num):
                if i == rank_num - 1 and j == expert_num - 1:
                    expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
                    break
                floor = math.floor(tokens_num / (rank_num * expert_num))
                rand_num = random.randint(1, floor)
                expert_token_num_per_rank[i][j] = rand_num
                tokens_sum += rand_num

        expert_token_num_type = 1
        idx_type = 0

        permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = self.moe_re_routing_torch(
            tokens,
            expert_token_num_per_rank,
            per_token_scales=per_token_scales,
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type)

        permute_tokens_npu, permute_per_token_scales_npu, permute_token_idx_npu, expert_token_num_npu = torch_npu.npu_moe_re_routing(
            tokens.npu(),
            expert_token_num_per_rank.npu(),
            per_token_scales=per_token_scales.npu(),
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type)

        self.assertRtolEqual(permute_tokens, permute_tokens_npu.cpu())
        self.assertRtolEqual(permute_per_token_scales, permute_per_token_scales_npu.cpu())
        self.assertRtolEqual(permute_token_idx, permute_token_idx_npu.cpu())
        self.assertRtolEqual(expert_token_num, expert_token_num_npu.cpu())

    @unittest.skip("skip test_moe_re_routing_3 now")
    @SupportedDevices(['Ascend910B'])
    def test_moe_re_routing_3(self, device="npu"):
        maybe_tokens_num_list = [16384, 10000, 28888, 32768]
        tokens_num = random.choice(maybe_tokens_num_list)
        tokens_length = 7 * 1024
        rank_num = 32
        expert_num = 8
        tokens = torch.randint(low=0, high=20, size=(tokens_num, tokens_length), dtype=torch.float16)
        expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype = torch.int64)
        per_token_scales = torch.randn(tokens_num, dtype = torch.float32)

        tokens_sum = 0
        for i in range(rank_num):
            for j in range(expert_num):
                if i == rank_num - 1 and j == expert_num - 1:
                    expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
                    break
                floor = math.floor(tokens_num / (rank_num * expert_num))
                rand_num = random.randint(1, floor)
                expert_token_num_per_rank[i][j] = rand_num
                tokens_sum += rand_num

        expert_token_num_type = 1
        idx_type = 0

        permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = self.moe_re_routing_torch(
            tokens,
            expert_token_num_per_rank,
            per_token_scales=per_token_scales,
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type)

        permute_tokens_npu, permute_per_token_scales_npu, permute_token_idx_npu, expert_token_num_npu = torch_npu.npu_moe_re_routing(
            tokens.npu(),
            expert_token_num_per_rank.npu(),
            per_token_scales=per_token_scales.npu(),
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type)

        self.assertRtolEqual(permute_tokens, permute_tokens_npu.cpu())
        self.assertRtolEqual(permute_per_token_scales, permute_per_token_scales_npu.cpu())
        self.assertRtolEqual(permute_token_idx, permute_token_idx_npu.cpu())
        self.assertRtolEqual(expert_token_num, expert_token_num_npu.cpu())

if __name__ == "__main__":
    run_tests()