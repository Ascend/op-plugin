import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAtbFusedAddTopkDiv(TestCase):
    def golden_calc(self, x, add_num, mapping_num, mapping_table, enable_expert_mapping):
        group_num = 8
        group_topk = 4
        is_norm = True
        n = 2
        k = 8
        scale = 1
        OFFSET = 7
        PRIME = 100000007
        MAX_INT32 = 2147483647
        MAX_REDUNDANT_EXPERT_NUM = 10
        count_mapping = []
        for i in range(MAX_REDUNDANT_EXPERT_NUM + 1):
            count_mapping.append(torch.zeros(i).to(torch.int32))
        x = x.cpu().to(torch.float32)
        add_num = add_num.cpu().to(torch.float32)
        m = torch.nn.Sigmoid()
        output_sig = m(x)
        # add
        input0 = torch.add(output_sig, add_num)
        # group_topk
        token_num, expert_num = input0.shape
        group_eles = expert_num // group_num
        input0 = torch.reshape(input0, (token_num, group_num, group_eles))
        output = input0.clone()
        group_tensor = torch.topk(input0, n).values
        group_tensor = torch.sum(group_tensor, dim=-1)
        sort_index = torch.from_numpy(np.argsort(-group_tensor.numpy(), kind='stable')) 
        cols_to_use = torch.arange(group_topk, group_num, dtype=torch.long) 
        row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(cols_to_use.shape[0])
        col_indices = sort_index.index_select(1, cols_to_use).view(-1)
        output[row_indices, col_indices] = float(0)
        group_top_k_res = torch.reshape(output, (token_num, expert_num))
        # topk
        sort_output = torch.sort(group_top_k_res, descending=True, stable=True)
        # gather
        gather_res = torch.gather(output_sig, -1, sort_output.indices[:, 0:k])
        if is_norm:
            # reduce_sum
            sum_res = torch.sum(gather_res, -1, keepdim=True)
            # div
            y = torch.div(gather_res, sum_res)
            # mul
            y = y * torch.tensor(scale, dtype=torch.float32)
        else:
            y = gather_res

        out_indices = sort_output.indices.to(torch.int32)
        enableExpertMapping = enable_expert_mapping
        if enableExpertMapping is True:
            offset = OFFSET
            prime = PRIME
            mapping_num = mapping_num.cpu()
            mapping_table = mapping_table.cpu()
            out_indices_clone = out_indices.clone().detach()
            for bs in range(token_num):
                indices_offset = torch.tensor(sort_output.indices[bs][group_topk * group_eles - 1] + offset, dtype=torch.int32)
                rand_value = torch.remainder(prime, indices_offset) / indices_offset.to(torch.float32)
                mapping_indices = torch.floor(mapping_num.to(torch.float32) * rand_value).to(torch.int32)
                count_mapping[mapping_table.shape[1]][mapping_indices[0]] = count_mapping[mapping_table.shape[1]][mapping_indices[0]] + 1
                for ki in range(k):
                    expert_id = out_indices_clone[bs][ki]
                    out_indices[bs][ki] = mapping_table[expert_id][mapping_indices[expert_id]]
        return y, out_indices[:, 0:k]

    def golden_compare(self, out_tensors, golden_out_tensors):
        diff = torch.abs(torch.subtract(out_tensors[0].float(), golden_out_tensors[0].float()))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype),
                                   torch.abs(golden_out_tensors[0])).float()

        err_factor, eb_factor = 2**(-10), 2**(-14)

        if torch.any(torch.greater(diff, err_factor * tensor_max)):
            print("[new standards] output0 accuracy failed")
            return False

        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), eb_factor)):
            print("[new standards] output0 eb failed")
            return False

        diff = torch.abs(torch.subtract(out_tensors[1], golden_out_tensors[1]))
        if torch.any(torch.greater(diff, 1)):
            print("[new standards] output1 accuracy failed")
            return False
        return True

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_atb_fused_add_topk_div_out(self):
        MAX_REDUNDANT_EXPERT_NUM = 10
        LOOP_TIMES = 1
        SUPPORT_DTYPE = [torch.float16]
        enable_expert_mapping = True
        for max_buffer in range(2, MAX_REDUNDANT_EXPERT_NUM + 1):
            for _ in range(LOOP_TIMES):
                for dtype in SUPPORT_DTYPE:
                    a = 16
                    b = 256
                    k = 8
                    input_x_golden = torch.from_numpy(np.random.uniform(1, 10, [a, b])).to(dtype)
                    input_add_num_golden = torch.from_numpy(np.random.uniform(1, 10, b)).to(dtype)
                    mapping_num = torch.ones(b, dtype=torch.int32) * max_buffer
                    mapping_table = torch.randint(0, b, [b, max_buffer], dtype=torch.int32)
                    for i in range(b):
                        for e in range(mapping_num[i], max_buffer):
                            mapping_table[i][e] = MAX_INT32
                y_expect, out_indices = self.golden_calc(input_x_golden, input_add_num_golden, mapping_num, mapping_table, enable_expert_mapping)
                y = torch.empty(a, k, dtype=torch.float32).npu()
                indices = torch.empty(a, k, dtype=torch.int32).npu()
                torch_npu.atb.npu_fused_add_topk_div(input_x_golden.npu(), input_add_num_golden.npu(), mapping_num=mapping_num.npu(), mapping_table=mapping_table.npu(), activation_type='activation_sigmoid', group_num=8, group_topk=4, n=2, k=k, is_norm=True, scale=1, enable_expert_mapping=True, y=y, indices=indices)
                self.assertRtolEqual(y_expect.npu(), y)
                self.assertEqual(out_indices[:, 0:k].npu(), indices)
                res1 = self.golden_compare(y.cpu(), y_expect.cpu())
                res2 = self.golden_compare(indices.cpu(), out_indices[:, 0:k].cpu())
                self.assertEqual(res1, True)
                self.assertEqual(res2, True)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_atb_fused_add_topk_div_out_model_shape(self):
        batch = 2048
        experts = 256
        top_k = 8
        SUPPORT_DTYPE = [torch.float16]
        enable_expert_mapping = False
        for dtype in SUPPORT_DTYPE:
            a = 2048
            b = 256
            k = 8
            input_x_golden = torch.from_numpy(np.random.uniform(1, 10, [a, b])).to(dtype)
            input_add_num_golden = torch.from_numpy(np.random.uniform(1, 10, b)).to(dtype)
            mapping_num = None
            mapping_table = None
            num_expert_group = 8
            topk_group = 4
            renormalize = True
            y_expect, out_indices = self.golden_calc(input_x_golden, input_add_num_golden, mapping_num, mapping_table, enable_expert_mapping)

            y = torch.empty(a, top_k, dtype=torch.float32).npu()
            indices = torch.empty(a, top_k, dtype=torch.int32).npu()
            torch_npu.atb.npu_fused_add_topk_div(input_x_golden.npu(), input_add_num_golden.npu(), activation_type='activation_sigmoid', group_num=num_expert_group, group_topk=topk_group, n=2, k=top_k, is_norm=renormalize, scale=1, enable_expert_mapping=False, y=y, indices=indices)
            self.assertRtolEqual(y_expect.npu(), y)
            self.assertEqual(out_indices[:, 0:k].npu(), indices)
            res1 = self.golden_compare(y.cpu(), y_expect.cpu())
            res2 = self.golden_compare(indices.cpu(), out_indices[:, 0:k].cpu())
            self.assertEqual(res1, True)
            self.assertEqual(res2, True)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_atb_fused_add_topk_div(self):
        MAX_REDUNDANT_EXPERT_NUM = 10
        LOOP_TIMES = 1
        SUPPORT_DTYPE = [torch.float16]
        enable_expert_mapping = True
        for max_buffer in range(2, MAX_REDUNDANT_EXPERT_NUM + 1):
            for _ in range(LOOP_TIMES):
                for dtype in SUPPORT_DTYPE:
                    a = 16
                    b = 256
                    k = 8
                    input_x_golden = torch.from_numpy(np.random.uniform(1, 10, [a, b])).to(dtype)
                    input_add_num_golden = torch.from_numpy(np.random.uniform(1, 10, b)).to(dtype)
                    mapping_num = torch.ones(b, dtype=torch.int32) * max_buffer
                    mapping_table = torch.randint(0, b, [b, max_buffer], dtype=torch.int32)
                    for i in range(b):
                        for e in range(mapping_num[i], max_buffer):
                            mapping_table[i][e] = MAX_INT32
                y_expect, out_indices = self.golden_calc(input_x_golden, input_add_num_golden, mapping_num, mapping_table, enable_expert_mapping)
                y, indices = torch_npu.atb.npu_fused_add_topk_div(input_x_golden.npu(), input_add_num_golden.npu(), mapping_num=mapping_num.npu(), mapping_table=mapping_table.npu(), activation_type='activation_sigmoid', group_num=8, group_topk=4, n=2, k=k, is_norm=True, scale=1, enable_expert_mapping=True)
                self.assertRtolEqual(y_expect.npu(), y)
                self.assertEqual(out_indices[:, 0:k].npu(), indices)
                res1 = self.golden_compare(y.cpu(), y_expect.cpu())
                res2 = self.golden_compare(indices.cpu(), out_indices[:, 0:k].cpu())
                self.assertEqual(res1, True)
                self.assertEqual(res2, True)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_atb_fused_add_topk_div_model_shape(self):
        batch = 2048
        experts = 256
        top_k = 8
        SUPPORT_DTYPE = [torch.float16]
        enable_expert_mapping = False
        for dtype in SUPPORT_DTYPE:
            a = 2048
            b = 256
            k = 8
            input_x_golden = torch.from_numpy(np.random.uniform(1, 10, [a, b])).to(dtype)
            input_add_num_golden = torch.from_numpy(np.random.uniform(1, 10, b)).to(dtype)
            mapping_num = None
            mapping_table = None
            num_expert_group = 8
            topk_group = 4
            renormalize = True
            y_expect, out_indices = self.golden_calc(input_x_golden, input_add_num_golden, mapping_num, mapping_table, enable_expert_mapping)
            y, indices = torch_npu.atb.npu_fused_add_topk_div(input_x_golden.npu(), input_add_num_golden.npu(), activation_type='activation_sigmoid', group_num=num_expert_group, group_topk=topk_group, n=2, k=top_k, is_norm=renormalize, scale=1, enable_expert_mapping=False)
            self.assertRtolEqual(y_expect.npu(), y)
            self.assertEqual(out_indices[:, 0:k].npu(), indices)
            res1 = self.golden_compare(y.cpu(), y_expect.cpu())
            res2 = self.golden_compare(indices.cpu(), out_indices[:, 0:k].cpu())
            self.assertEqual(res1, True)
            self.assertEqual(res2, True)

if __name__ == "__main__":
    run_tests()
