import torch
import torch_npu
import unittest
import numpy as np
import torch.nn.functional as F

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def non_quant_golden_with_offset(x, weight, scale, perTokenScale, groupList, bias, offset):
    groupNum, k, n = weight.shape
    quantGroupNum = scale.shape[1]

    index = np.cumsum(groupList)
    xSplit = np.split(x, index * 2, axis=0)
    perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
    weightGroup = weight.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
    mmOuts = []
    atomic = np.float16
    for i in range(groupNum):
        xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
        mmi = np.zeros([xi.shape[0], n], dtype=atomic)
        for j in range(quantGroupNum):
            mm = np.matmul(xi[:, j, :], weightGroup[i, j, ...])
            mm = mm.astype(np.float32) * scale[i, j].reshape(1, -1)
            mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)

        mmi = mmi.reshape(-1, 2, n).astype(np.float32)
        mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias[i].reshape(1, n)
        if offset is not None:
            mmi_xo = np.zeros(mmi.shape, dtype=np.float32)
            xi_o = xSplit[i].astype(np.int32).reshape(-1, 2, k)
            xi_o = xi_o[:, 0, :] * 16 + xi_o[:, 1, :] + 8
            xi_o = xi_o.astype(np.float16).reshape(-1, quantGroupNum, k // quantGroupNum)
            for j in range(quantGroupNum):
                mm = xi_o[:, j, :].sum(axis=1, keepdims=True).astype(np.float32)
                mm = np.matmul(mm, offset[i, j].reshape(1, -1))
                mmi_xo += mm.astype(np.float32)
            mmi += mmi_xo
        mmi = mmi * perTokenScaleSplit[i]
        mmOuts.append(mmi)
    golden = np.concatenate(mmOuts, axis=0)
    golden_tensor = torch.from_numpy(golden)
    return golden_tensor.to(torch.float32)


def combine_func(x, logits, residual, residScale, sourceRow, topK, offset):
    out = x * logits.reshape(-1, 1)
    index = np.argsort(sourceRow)
    out = out[index].reshape(-1, topK, x.shape[-1]).sum(axis=1)
    out[offset:offset + residual.shape[0], :] += residScale * residual.to(torch.float32)
    return out


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
    
    def supported_a8w4_op_exec(self, topK, x_in, weight_in, groupList_in, scale_in,
                                bias_in, offset, perTokenScale_in, residual, logits,
                                sourceRow, residScale, shared_input_offset=0):
        weightNz = weight_in.astype(np.int8)
        groupNum = groupList_in.shape[0]
        m = x_in.shape[0]
        k = x_in.shape[1]
        n = scale_in.shape[2]

        weight = weightNz.reshape(groupNum, k, n)
        xC12 = np.concatenate([x_in.reshape(m, 1, k) // 16, (x_in.reshape(m, 1, k) & 0x0F) - 8], axis=1).reshape(m * 2, k)
        data = xC12.astype(np.int8)
        data[data < 0] += 16
        xInt4 = (data[..., 1::2] << 4) | (data[..., ::2] & 0x0F)
        xInt4.dtype = np.int8
        scaleUint32 = scale_in.astype(np.uint32)
        scaleUint32.dtype = np.float32

        mm_out = non_quant_golden_with_offset(xC12, weight, scaleUint32, perTokenScale_in, groupList_in, bias_in, offset)
        out = combine_func(mm_out, logits, residual, residScale, sourceRow, topK, shared_input_offset)
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
        custom_output = torch_npu.npu_grouped_matmul_finalize_routing(
            x.npu(), weightNz, group_list.npu(), scale=scale.npu(),
            pertoken_scale=pertoken_scale.npu(), shared_input=shared_input.npu(),
            logit=logit.npu(), row_index=row_index.npu(),
            shared_input_offset=shared_input_offset, output_bs=output_bs
        ).to("cpu")
        self.assertRtolEqual(supported_output, custom_output, 0.001)

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

    @unittest.skip("Skip temporary. The kernel is not supported.")
    @SupportedDevices(["Ascend910B"])
    def test_npu_grouped_matmul_finalize_routing_a8w4(self, device="npu"):
        m, k, n, group_num = 8, 2048, 7168, 8
        batch = m // group_num
        quantGroupSize = k
        topK = 8

        x = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        weight = torch.randint(-5, 5, (group_num, k, n), dtype=torch.int32)
        scale_np = np.random.normal(0, 0.01, (group_num, 1, n)).astype(np.float32)
        perGroupScale = np.ones([group_num, k // quantGroupSize, n]).astype(np.float32)
        scaleUint32 = (scale_np * perGroupScale).astype(np.float16).astype(np.float32)
        scaleUint32.dtype = np.uint32
        scaleUint64 = np.zeros((group_num, k // quantGroupSize, n * 2), dtype=np.uint32)
        scaleUint64[..., ::2] = scaleUint32
        scaleUint64.dtype = np.int64
        scale = torch.from_numpy(scaleUint64)
        bias = torch.normal(0, 0.01, (group_num, n), dtype=torch.float32)
        offset = torch.randint(-5, 5, (group_num, k // quantGroupSize, n), dtype=torch.float32)
        pertoken_scale = torch.normal(0, 0.01, (m, 1), dtype=torch.float32)
        group_list = torch.tensor([batch] * group_num, dtype=torch.int64)

        logit_ori = torch.normal(0, 0.1, (batch, group_num), dtype=torch.float32)
        routing = torch.argsort(logit_ori, 1)[:, -topK:]

        shared_input = torch.normal(0, 0.1, (max(batch // 4, 1), n), dtype=torch.bfloat16)
        logit = F.softmax(
            logit_ori[torch.arange(batch).reshape(-1, 1).repeat(1, topK), routing],
            dim=1,
            dtype=torch.float32
        ).reshape(m)
        row_index = (torch.argsort(routing.reshape(-1)) // topK).to(torch.int64)
        shared_input_scale = 1
        shared_input_offset = 0
        output_bs = batch

        weight_quant = torch_npu.npu_quantize(weight.float().npu(), torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)
        custom_output = torch_npu.npu_grouped_matmul_finalize_routing(
            x.npu(), weight_quant, group_list.npu(), scale=scale.npu(),
            bias=bias.npu(), offset=offset.npu(),
            pertoken_scale=pertoken_scale.reshape(m).npu(), shared_input=shared_input.npu(),
            logit=logit.npu(), row_index=row_index.npu(), shared_input_weight=shared_input_scale,
            shared_input_offset=shared_input_offset, output_bs=output_bs
        ).to("cpu")
        supported_output = self.supported_a8w4_op_exec(
            topK, x.numpy(), weight.numpy(), group_list.numpy(), scale.numpy(), bias.numpy(),
            offset.numpy(), pertoken_scale.numpy(), shared_input, logit.numpy(), row_index.numpy(),
            shared_input_scale, shared_input_offset
        )
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()
