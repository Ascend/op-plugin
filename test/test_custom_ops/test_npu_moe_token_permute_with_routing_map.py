import unittest
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestMoePermuteWithRoutingMap(TestCase):

    def cpu_op_exec(
        self,
        tokens,
        routing_map,
        probs,
        num_out_tokens,
        drop_and_pad,
    ):
        num_tokens, hidden = tokens.shape
        num_experts = routing_map.shape[1]

        cap = num_out_tokens // num_experts
        sorted_indices2 = None
        if drop_and_pad:
            permuted_probs = None
            routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
            sorted_indices = routing_map.sort(
                dim=-1, descending=True, stable=True)[1]
            sorted_indices = sorted_indices[:, :cap].contiguous()
            sorted_indices = sorted_indices.view(-1)
            if probs is not None:
                probs_T_1D = probs.T
                probs_T_1D = probs_T_1D.contiguous().view(-1)
                indices_dim0 = torch.arange(
                    num_experts, device=routing_map.device).unsqueeze(-1)
                indices_1D = (indices_dim0 * num_tokens +
                              sorted_indices.view(num_experts, cap)).view(-1)
                permuted_probs = probs_T_1D.index_select(0, indices_1D)
            permuted_input = tokens.index_select(0, sorted_indices)
            return permuted_input, permuted_probs, sorted_indices.to(torch.int32)
        else:
            routing_map = routing_map.bool().T.contiguous()
            token_indices = (
                torch.arange(num_tokens, device=routing_map.device).unsqueeze(
                    0).expand(num_experts, -1)
            )
            sorted_indices = token_indices.masked_select(routing_map)
            sorted_indices2 = torch.sort(
                sorted_indices.float(), stable=True)[1]
            if probs is not None:
                permuted_probs = probs.T.masked_select(routing_map)
            else:
                permuted_probs = None
            permuted_input = tokens.index_select(0, sorted_indices)
            return permuted_input, permuted_probs, sorted_indices2.to(torch.int32)

    @unittest.skip("Skipping due to outdated CANN; please update CANN and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_token_permute_with_routing_map_default(self):
        x = torch.randn((3, 4), dtype=torch.float)
        x.requires_grad = True
        rounting_map = torch.tensor(
            [[True, True], [True, True], [True, True]], dtype=torch.bool)
        numtoken = 6
        padMode = False

        x_npu = x.npu().detach()
        x_npu.requires_grad = True
        rounting_map_npu = rounting_map.npu()
        c1, c2, c3 = self.cpu_op_exec(x, rounting_map, None, numtoken, padMode)
        x1, x2, x3 = torch_npu.npu_moe_token_permute_with_routing_map(x_npu, rounting_map_npu, num_out_tokens=numtoken, drop_and_pad=padMode)
        self.assertRtolEqual(c1, x1)
        c1.sum().backward()
        x1.sum().backward()
        self.assertRtolEqual(x.grad, x_npu.grad)

    @unittest.skip("Skipping due to outdated CANN; please update CANN and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_token_permute_with_routing_map_with_pad_mode(self):
        x = torch.randn((3, 4), dtype=torch.float)
        x.requires_grad = True
        rounting_map = torch.tensor(
            [[True, False], [True, True], [True, True]], dtype=torch.bool)
        numtoken = 3
        padMode = True
        x_npu = x.npu().detach()
        x_npu.requires_grad = True
        rounting_map_npu = rounting_map.npu()
        c1, c2, c3 = self.cpu_op_exec(x, rounting_map, None, numtoken, padMode)
        x1, x2, x3 = torch_npu.npu_moe_token_permute_with_routing_map(
            x_npu, rounting_map_npu, num_out_tokens=numtoken, drop_and_pad=padMode)
        self.assertRtolEqual(c1, x1)
        c1.sum().backward()
        x1.sum().backward()
        self.assertRtolEqual(x.grad, x_npu.grad)

    @unittest.skip("Skipping due to outdated CANN; please update CANN and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_token_permute_with_routing_map_with_probs(self):
        x = torch.randn((3, 4), dtype=torch.float)
        x.requires_grad = True
        rounting_map = torch.tensor(
            [[True, True], [True, True], [True, True]], dtype=torch.bool)
        numtoken = 6
        padMode = True
        probs = torch.randn([3, 2], dtype=torch.float)
        probs.requires_grad = True
        x_npu = x.npu().detach()
        x_npu.requires_grad = True
        rounting_map_npu = rounting_map.npu()
        probs_npu = probs.npu().detach()
        probs_npu.requires_grad = True
        c1, c2, c3 = self.cpu_op_exec(x, rounting_map, probs, numtoken, padMode)
        x1, x2, x3 = torch_npu.npu_moe_token_permute_with_routing_map(
            x_npu, rounting_map_npu, probs=probs_npu, num_out_tokens=numtoken, drop_and_pad=padMode)
        self.assertRtolEqual(c1, x1)
        (c1.sum() + c2.sum()).backward()
        (x1.sum() + x2.sum()).backward()
        self.assertRtolEqual(x.grad, x_npu.grad)
        self.assertRtolEqual(probs.grad, probs_npu.grad)


if __name__ == "__main__":
    run_tests()
