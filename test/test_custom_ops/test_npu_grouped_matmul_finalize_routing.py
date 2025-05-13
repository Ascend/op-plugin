import torch
import torch_npu
import unittest
import torch.nn.functional as F

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestGroupedMatmulFinalizeRouting(TestCase):

    def supported_op_exec(self,
                          topK, x, weight, group_list, scale, pertoken_scale,
                          shared_input=None, logit=None, row_index=None,
                          shared_input_scale=1, shared_input_offset=0):
        x_split = torch.split(x, group_list.tolist(), dim=0)
        pertoken_scale_split = torch.split(pertoken_scale, group_list.tolist(), dim=0)
        mm_outs = []
        for i in range(len(group_list)):
            mm = torch.matmul(x_split[i].to(torch.int32), weight[i].to(torch.int32))
            mm = mm.to(torch.float32) * scale[i].to(torch.float32) * pertoken_scale_split[i]
            mm_outs.append(mm)
        mm_out = torch.cat(mm_outs, dim=0)
        if shared_input is not None:
            out = mm_out * logit.reshape(-1, 1)
            index = torch.argsort(row_index, dim=0)
            out = out[index].reshape(-1, topK, mm_out.shape[-1]).sum(dim=1)
            out[shared_input_offset:shared_input_offset + shared_input.shape[0], :] += \
                shared_input_scale * shared_input.to(torch.float32)
        else:
            index = torch.argsort(row_index, dim=0)
            out = mm_out[index].reshape(-1, topK, mm_out.shape[-1]).sum(dim=1)
        return out
    
    @SupportedDevices(["Ascend910B"])
    def test_npu_grouped_matmul_finalize_routing_1(self, device="npu"):
        m, k, n, batch, topK, group_num, shared_input_scale = 576, 2048, 7168, 72, 8, 8, 1
        x = torch.randint(-10, 10, (m, k), dtype=torch.int8)
        weight = torch.randint(-10, 10, (group_num, k, n), dtype=torch.int8)
        scale = torch.normal(0, 0.01, (group_num, n), dtype=torch.float32)
        pertoken_scale = torch.normal(0, 0.01, (m, 1), dtype=torch.float32)
        group_list = torch.tensor([batch] * group_num, dtype=torch.int64)

        logit_ori = torch.normal(0, 0.1, (batch, group_num), dtype=torch.float32)
        routing = torch.argsort(logit_ori, 1)[:, -topK:]

        shared_input = torch.normal(0, 0.1, (batch // 4, n), dtype=torch.bfloat16)
        logit = F.softmax(
            logit_ori[torch.arange(batch).reshape(-1, 1).repeat(1, topK), routing],
            dim=1,
            dtype=torch.float32
        ).reshape(m)
        row_index = (torch.argsort(routing.reshape(-1)) // topK).to(torch.int64)
        shared_input_offset = batch // 2
        output_bs = batch

        supported_output = self.supported_op_exec(topK, x, weight, group_list, scale, 
                                                  pertoken_scale, shared_input, logit, row_index,
                                                  shared_input_scale, shared_input_offset)
        weightNz = torch_npu.npu_format_cast(weight.npu(), 29)
        pertoken_scale = pertoken_scale.reshape(m)
        custome_output = torch_npu.npu_grouped_matmul_finalize_routing(
            x.npu(), weightNz, group_list.npu(), scale=scale.npu(),
            pertoken_scale=pertoken_scale.npu(), shared_input=shared_input.npu(),
            logit=logit.npu(), row_index=row_index.npu(),
            shared_input_offset=shared_input_offset, output_bs=output_bs
        ).to("cpu")
        self.assertRtolEqual(supported_output, custome_output, 0.001)

    @SupportedDevices(["Ascend910B"])
    def test_npu_grouped_matmul_finalize_routing_2(self, device="npu"):
        m, k, n, batch, topK, group_num = 72, 2048, 7168, 72, 1, 1
        x = torch.randint(-10, 10, (m, k), dtype=torch.int8)
        weight = torch.randint(-10, 10, (group_num, k, n), dtype=torch.int8)
        scale = torch.normal(0, 0.01, (group_num, n), dtype=torch.float32)
        pertoken_scale = torch.normal(0, 0.01, (m, 1), dtype=torch.float32)
        group_list = torch.tensor([batch] * group_num, dtype=torch.int64)

        logit_ori = torch.normal(0, 0.1, (batch, group_num), dtype=torch.float32)
        routing = torch.argsort(logit_ori, 1)[:, -topK:]
        row_index = (torch.argsort(routing.reshape(-1)) // topK).to(torch.int64)
        output_bs = m

        supported_output = self.supported_op_exec(topK, x, weight, group_list,
                                                  scale, pertoken_scale, row_index=row_index)
        weightNz = torch_npu.npu_format_cast(weight.npu(), 29)
        pertoken_scale = pertoken_scale.reshape(m)
        custom_output = torch_npu.npu_grouped_matmul_finalize_routing(
            x.npu(), weightNz, group_list.npu(), scale=scale.npu(),
            pertoken_scale=pertoken_scale.npu(), row_index=row_index.npu(),
            output_bs=output_bs
        ).to("cpu")
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()
