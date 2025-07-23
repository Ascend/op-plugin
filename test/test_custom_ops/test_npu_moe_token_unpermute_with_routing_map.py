import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestMoeUnpermuteWithRoutingMap(TestCase):

    def generate_bool_matrix(self, m, n, k):
        matrix = torch.zeros((m, n), dtype=torch.bool)
        
        for i in range(m):
            indices = torch.randperm(n)[:k]
            matrix[i, indices] = True
        
        return matrix.to(torch.int8)

    # pylint:disable = huawei-too-many-arguments
    def unpermute_with_routing_map_golden(self, permuted_tokens, sorted_indices, routing_map, probs, paddedMode, restore_shape):
        if paddedMode:
            sorted_index, sorted_indices1 = sorted_indices.sort(dim=-1, descending=False, stable=True)
            if probs is not None:
                num_experts = probs.size(1)
                num_tokens = probs.size(0)
                hidden = permuted_tokens.size(1)
                capacity = sorted_indices.size(0) // num_experts

                probs_T_1D = probs.T.contiguous().view(-1)

                indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
                indices_dim1 = sorted_indices.view(num_experts, capacity)
                indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)

                permuted_probs = probs_T_1D.index_select(0, indices_1D)

                permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)


            output_tokens = torch.zeros(
                restore_shape, dtype=torch.float64, device=permuted_tokens.device
            )
            for i in range(sorted_index.size(-1)):
                output_tokens[sorted_index[i]] += permuted_tokens[sorted_indices1[i]]
            if probs is not None:
                return output_tokens.to(dtype=permuted_tokens.dtype),\
                    sorted_indices1.to(torch.int32), sorted_index, permuted_probs
            else:
                return output_tokens.to(dtype=permuted_tokens.dtype),\
                    sorted_indices1.to(torch.int32), sorted_index, None

        else:
            topk = permuted_tokens.shape[0] // routing_map.shape[0]
            unpermuted_tokens = permuted_tokens.index_select(0, sorted_indices)
            unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            if probs is not None:
                unpermuted_tokens = unpermuted_tokens * probs.contiguous().masked_select(
                    routing_map.to(dtype=torch.bool).contiguous()).reshape(-1, topk).unsqueeze(-1)
            unpermuted_tokens = unpermuted_tokens.sum(dim=1)
            if probs is not None:
                return unpermuted_tokens.to(dtype=permuted_tokens.dtype), torch.tensor([1]).to(dtype=torch.int32),\
                    torch.tensor([1]).to(dtype=torch.int32), \
                        probs.contiguous().masked_select(routing_map.to(dtype=torch.bool).contiguous())
            else:
                return unpermuted_tokens.to(dtype=permuted_tokens.dtype), torch.tensor([1]).to(dtype=torch.int32),\
                    torch.tensor([1]).to(dtype=torch.int32), None

    @unittest.skip("skip test_npu_moe_token_unpermute_with_routing_map_internel due to cann version")
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_token_unpermute_with_routing_map_internel(self):
        token_num = 40
        hidden_size = 20
        expert_num = 20
        top_k = 20
        capacity = 20
        out_token_num = token_num * top_k
        out_token_num_pad = expert_num * capacity

        for drop_and_pad in [True, False]:
            for need_probs in [True, False]:
                routing_map = self.generate_bool_matrix(token_num, expert_num, top_k)
                routing_map_npu = routing_map.npu()
                self.assertEqual(routing_map, routing_map_npu.cpu())

                if drop_and_pad:
                    permuted_tokens = torch.randn([out_token_num_pad, hidden_size])
                    permuted_tokens_npu = permuted_tokens.npu()
                    permuted_tokens.requires_grad_(True)
                    permuted_tokens_npu.requires_grad_(True)
                    self.assertEqual(permuted_tokens, permuted_tokens_npu.cpu())

                    routing_map_tmp = routing_map.T.contiguous()
                    sorted_indices = routing_map_tmp.argsort(dim=-1, descending=True, stable=True)[:, :capacity].contiguous().to(torch.int32).view(-1)
                    sorted_indices_npu = sorted_indices.npu()
                    self.assertEqual(sorted_indices, sorted_indices_npu.cpu())
                else:
                    permuted_tokens = torch.randn([out_token_num, hidden_size])
                    permuted_tokens_npu = permuted_tokens.npu()
                    permuted_tokens.requires_grad_(True)
                    permuted_tokens_npu.requires_grad_(True)
                    self.assertEqual(permuted_tokens, permuted_tokens_npu.cpu())

                    routing_map_tmp = routing_map.bool().T.contiguous()
                    token_indices = torch.arange(token_num, device=routing_map_tmp.device).unsqueeze(0).expand(expert_num, -1)
                    sorted_indices_tmp = token_indices.masked_select(routing_map_tmp)
                    sorted_indices = torch.sort(sorted_indices_tmp.float(), stable=True)[1].to(torch.int32)

                    sorted_indices_npu = sorted_indices.npu()
                    self.assertEqual(sorted_indices, sorted_indices_npu.cpu())

                porbs = torch.randn([token_num, expert_num])
                porbs_npu = porbs.npu()
                porbs.requires_grad_(True)
                porbs_npu.requires_grad_(True)
                self.assertEqual(porbs, porbs_npu.cpu())

                restore_shape = [token_num, hidden_size]

                if need_probs:
                    unpermuted_tokens_golden, out_idx_golden, permute_token_id_golden, permute_probs_golden = self.unpermute_with_routing_map_golden(
                        permuted_tokens, sorted_indices, routing_map, porbs, drop_and_pad, restore_shape)
                    unpermuted_tokens, out_idx, permute_token_id, permute_probs = torch_npu._npu_moe_token_unpermute_with_routing_map(
                        permuted_tokens_npu, sorted_indices_npu, restore_shape, probs=porbs_npu, routing_map=routing_map_npu, drop_and_pad=drop_and_pad)
                else:
                    unpermuted_tokens_golden, out_idx_golden, permute_token_id_golden, permute_probs_golden = self.unpermute_with_routing_map_golden(
                        permuted_tokens, sorted_indices, routing_map, None, drop_and_pad, restore_shape)
                    unpermuted_tokens, out_idx, permute_token_id, permute_probs = torch_npu._npu_moe_token_unpermute_with_routing_map(
                        permuted_tokens_npu, sorted_indices_npu, restore_shape, probs=None, routing_map=routing_map_npu, drop_and_pad=drop_and_pad)

                if need_probs:
                    self.assertRtolEqual(unpermuted_tokens_golden, unpermuted_tokens)
                    self.assertRtolEqual(permute_probs_golden, permute_probs)
                    if drop_and_pad:
                        self.assertRtolEqual(out_idx_golden, out_idx)
                        self.assertRtolEqual(permute_token_id_golden, permute_token_id)
                else:
                    self.assertRtolEqual(unpermuted_tokens_golden, unpermuted_tokens)
                    if drop_and_pad:
                        self.assertRtolEqual(out_idx_golden, out_idx)
                        self.assertRtolEqual(permute_token_id_golden, permute_token_id)
                unpermuted_tokens_golden.sum().backward()
                unpermuted_tokens.sum().backward()
                self.assertRtolEqual(permuted_tokens.grad, permuted_tokens_npu.grad.cpu())
                if need_probs:
                    self.assertRtolEqual(porbs.grad, porbs_npu.grad.cpu())

    @unittest.skip("skip test_npu_moe_token_unpermute_with_routing_map due to cann version")
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_token_unpermute_with_routing_map(self):
        token_num = 40
        hidden_size = 20
        expert_num = 20
        top_k = 20
        capacity = 20
        out_token_num = token_num * top_k
        out_token_num_pad = expert_num * capacity

        for drop_and_pad in [True, False]:
            for need_probs in [True, False]:
                routing_map = self.generate_bool_matrix(token_num, expert_num, top_k)
                routing_map_npu = routing_map.npu()
                self.assertEqual(routing_map, routing_map_npu.cpu())

                if drop_and_pad:
                    permuted_tokens = torch.randn([out_token_num_pad, hidden_size])
                    permuted_tokens_npu = permuted_tokens.npu()
                    permuted_tokens.requires_grad_(True)
                    permuted_tokens_npu.requires_grad_(True)
                    self.assertEqual(permuted_tokens, permuted_tokens_npu.cpu())

                    routing_map_tmp = routing_map.T.contiguous()
                    sorted_indices = routing_map_tmp.argsort(dim=-1, descending=True, stable=True)[:, :capacity].contiguous().to(torch.int32).view(-1)
                    sorted_indices_npu = sorted_indices.npu()
                    self.assertEqual(sorted_indices, sorted_indices_npu.cpu())
                else:
                    permuted_tokens = torch.randn([out_token_num, hidden_size])
                    permuted_tokens_npu = permuted_tokens.npu()
                    permuted_tokens.requires_grad_(True)
                    permuted_tokens_npu.requires_grad_(True)
                    self.assertEqual(permuted_tokens, permuted_tokens_npu.cpu())

                    routing_map_tmp = routing_map.bool().T.contiguous()
                    token_indices = torch.arange(token_num, device=routing_map_tmp.device).unsqueeze(0).expand(expert_num, -1)
                    sorted_indices_tmp = token_indices.masked_select(routing_map_tmp)
                    sorted_indices = torch.sort(sorted_indices_tmp.float(), stable=True)[1].to(torch.int32)

                    sorted_indices_npu = sorted_indices.npu()
                    self.assertEqual(sorted_indices, sorted_indices_npu.cpu())

                porbs = torch.randn([token_num, expert_num])
                porbs_npu = porbs.npu()
                porbs.requires_grad_(True)
                porbs_npu.requires_grad_(True)
                self.assertEqual(porbs, porbs_npu.cpu())

                restore_shape = [token_num, hidden_size]

                if need_probs:
                    unpermuted_tokens_golden, _, _, _ = self.unpermute_with_routing_map_golden(
                        permuted_tokens, sorted_indices, routing_map, porbs, drop_and_pad, restore_shape)
                    unpermuted_tokens = torch_npu.npu_moe_token_unpermute_with_routing_map(
                        permuted_tokens_npu, sorted_indices_npu, restore_shape, probs=porbs_npu, routing_map=routing_map_npu, drop_and_pad=drop_and_pad)
                else:
                    unpermuted_tokens_golden, _, _, _ = self.unpermute_with_routing_map_golden(
                        permuted_tokens, sorted_indices, routing_map, None, drop_and_pad, restore_shape)
                    unpermuted_tokens = torch_npu.npu_moe_token_unpermute_with_routing_map(
                        permuted_tokens_npu, sorted_indices_npu, restore_shape, probs=None, routing_map=routing_map_npu, drop_and_pad=drop_and_pad)

                self.assertRtolEqual(unpermuted_tokens_golden, unpermuted_tokens)
                unpermuted_tokens_golden.sum().backward()
                unpermuted_tokens.sum().backward()
                self.assertRtolEqual(permuted_tokens.grad, permuted_tokens_npu.grad.cpu())
                if need_probs:
                    self.assertRtolEqual(porbs.grad, porbs_npu.grad.cpu())


if __name__ == "__main__":
    run_tests()
