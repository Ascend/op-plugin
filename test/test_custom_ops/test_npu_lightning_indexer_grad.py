import unittest
import torch
import numpy as np
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestCustomLightningIndexerGrad(TestCase):
    batch = 2
    seqlenQ = 20
    seqlenK = 2048
    headNumQ = 64
    headNumK = 1
    groupNum = headNumQ // headNumK
    headDim = 128
    indexTopk = 2048
    inputDataType = torch.bfloat16

    def lightning_indexer_fwd(self, q, k, weights, indexTopk, layout):
        weights = weights.reshape(*weights.shape, 1)
        mask = torch.full((self.seqlenQ, self.seqlenK), float("-inf")).triu_(1) if self.seqlenQ > 1 else None
        query = q.permute(0, 2, 1, 3) # (B, S1, N1, D) -> (B, N1, S1, D)
        key = k.permute(0, 2, 3, 1) # (B, S2, N2, D) -> (B, N2, D, S2)
        p = torch.matmul(query, key) # (B, N1, S1, S2)
        reluOut = torch.nn.functional.relu(p) # (B, N1, S1, S2)
        weightOut = reluOut * weights.permute(0, 2, 1, 3) # (B, N1, S1, S2) * (B, N1, S1, 1) -> (B, N1, S1, S2)
        reduceOut = torch.sum(weightOut, dim=1) # (B, S1, S2)
        indexScore = reduceOut
        if mask is not None:
            indexScore += mask # (B, S1, S2) + (S1, S2) -> (B, S1, S2)

        # N2 = 1
        value2, topkIndices2 = indexScore.topk(min(self.indexTopk, self.seqlenK), dim=-1) # (B, S1, S2) -> (B, S1, topK)
        return value2, topkIndices2

    def lightning_indexer_bwd(self, valueGrad, q, k, weights, topkIndices, indexTopk, layout, isHighPrecision=False,
                              useMatmul=True):
        valueGrad = valueGrad.unsqueeze(1).expand(self.batch, self.headNumQ, self.seqlenQ, self.indexTopk)
        valueGrad = valueGrad.to(self.inputDataType)
        topkIndices = topkIndices.unsqueeze(1).expand(self.batch, self.headNumQ, self.seqlenQ, self.indexTopk)
        dumpOutput = None
        dumpWeightsOutput = None
        mulGrad = torch.zeros((self.batch, self.headNumQ, self.seqlenQ, self.indexTopk)).to(torch.float32)
        if useMatmul:
            print("ENTER MULGRAD CUBE")
            mulGrad = mulGrad.permute(0, 2, 1, 3).reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum,
                                                          self.indexTopk)
            for bIdx in range(self.batch):
                for headNumKIdx in range(self.headNumK):
                    for seqlenIdx in range(self.seqlenQ):
                        weightSelect = weights.reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum)[bIdx,
                                       seqlenIdx, headNumKIdx, :].reshape(self.groupNum, 1)
                        valueGradSelect = valueGrad[bIdx, headNumKIdx, seqlenIdx, :].reshape(1, self.indexTopk)
                        mulGrad[bIdx, seqlenIdx, headNumKIdx, :, :] = torch.matmul(weightSelect.to(self.inputDataType),
                                                                                   valueGradSelect.to(
                                                                                       self.inputDataType))
            mulGrad = mulGrad.reshape(self.batch, self.seqlenQ, self.headNumQ, self.indexTopk).permute(0, 2, 1, 3)
        else:
            print("ENTER MULGRAD VECTOR")
            weights = weights.reshape(*weights.shape, 1)
            mulGrad = valueGrad * weights.permute(0, 2, 1, 3)

        reluIn = torch.zeros((self.batch, self.headNumK, self.seqlenQ, self.groupNum, self.indexTopk)).to(
            torch.float32).npu()
        for bIdx in range(self.batch):
            for headNumKIdx in range(self.headNumK):
                for seqlenIdx in range(self.seqlenQ):
                    indices = topkIndices[bIdx, headNumKIdx, seqlenIdx, :]
                    kSelect = torch.index_select(k.permute(0, 2, 1, 3)[bIdx, headNumKIdx, :], dim=0,
                                                 index=indices).reshape(self.indexTopk, self.headDim)
                    qSelect = q.reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum, self.headDim)[bIdx,
                              seqlenIdx, headNumKIdx, :, :].reshape(self.groupNum, self.headDim)
                    reluInPart = torch.matmul(qSelect.to(self.inputDataType).to(torch.float),
                                              kSelect.permute(1, 0).to(self.inputDataType).to(torch.float))
                    reluIn[bIdx, headNumKIdx, seqlenIdx, :, :] = reluInPart

        reluIn = reluIn.permute(0, 1, 3, 2, 4).reshape(self.batch, self.headNumQ, self.seqlenQ, self.indexTopk).npu()
        mask = reluIn < 0.0
        reluGrad = torch.masked_fill(mulGrad.npu(), mask.npu(), 0.0).npu()

        keyPermute = k.permute(0, 2, 1, 3)
        dq = torch.zeros((self.batch, self.headNumK, self.seqlenQ, self.groupNum, self.headDim)).to(torch.float32).npu()
        for bIdx in range(self.batch):
            for headNumKIdx in range(self.headNumK):
                for seqlenIdx in range(self.seqlenQ):
                    indices = topkIndices[bIdx, headNumKIdx, seqlenIdx, :]
                    selectKey = torch.index_select(keyPermute[bIdx, headNumKIdx, :, :], 0, indices)
                    permuteReluGrad = reluGrad.permute(0, 2, 1, 3).reshape(self.batch, self.seqlenQ, self.headNumK,
                                                                           self.groupNum, self.indexTopk)[bIdx,
                                      seqlenIdx, headNumKIdx, :, :]
                    # dqPart shape is [groupNum, headDim]
                    dqPart = torch.matmul(
                        permuteReluGrad.reshape(self.groupNum, self.indexTopk).to(self.inputDataType).npu(),
                        selectKey.reshape(self.indexTopk, self.headDim).to(self.inputDataType).npu())
                    dq[bIdx, headNumKIdx, seqlenIdx, :, :] = dqPart.reshape(self.groupNum, self.headDim)
        dq = dq.permute(0, 1, 3, 2, 4).reshape(self.batch, self.headNumQ, self.seqlenQ, self.headDim)

        # dk single block output
        dk = torch.zeros((self.batch, self.headNumK, self.seqlenK, self.headDim)).to(torch.float32).npu()
        for bIdx in range(self.batch):
            for headNumKIdx in range(self.headNumK):
                for seqlenIdx in range(self.seqlenQ):
                    indices = topkIndices[bIdx, headNumKIdx, seqlenIdx, :]
                    selectQuery = q.reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum, self.headDim)[bIdx,
                                  seqlenIdx, headNumKIdx, :, :]
                    permuteReluGrad = reluGrad.permute(0, 2, 1, 3).reshape(self.batch, self.seqlenQ, self.headNumK,
                                                                           self.groupNum, self.indexTopk)[bIdx,
                                      seqlenIdx, headNumKIdx, :, :].permute(1, 0)
                    # dqPart shape is [indexTopk, headDim]
                    dkPart = torch.matmul(
                        permuteReluGrad.reshape(self.indexTopk, self.groupNum).to(self.inputDataType).to(
                            torch.float).npu(),
                        selectQuery.reshape(self.groupNum, self.headDim).to(self.inputDataType).to(torch.float).npu())
                    dk[bIdx, headNumKIdx].index_add_(0, indices.npu(), dkPart.float().npu())

        reluOut = torch.nn.functional.relu(reluIn)
        dweights = reluOut * valueGrad.to(torch.float32)
        dweights = self.sum_pairwise_to_eight(dweights.to(torch.float32)).reshape(self.batch, self.headNumQ,
                                                                                  self.seqlenQ, 1)
        return dq.permute(0, 2, 1, 3), dk.permute(0, 2, 1, 3), dweights.permute(0, 2, 1,
                                                                                3), dumpOutput, dumpWeightsOutput

    def sum_pairwise_to_eight(self, tensor, dim=-1, target_size=8, compute_dtype=torch.float32):
        original_dtype = tensor.dtype
        original_device = tensor.device

        size = tensor.shape[dim]
        if size <= target_size:
            return torch.sum(tensor, dim=dim)

        t = tensor.to(dtype=compute_dtype)

        dim_normalized = dim if dim >= 0 else tensor.ndim + dim
        if dim_normalized != tensor.ndim - 1:
            t = t.transpose(dim_normalized, -1)

        current_size = t.shape[-1]
        while current_size > target_size:
            half_size = current_size // 2
            first_half = t[..., :half_size]
            second_half = t[..., half_size:half_size + half_size]

            first_half += second_half
            t = first_half
            current_size = half_size

        result = torch.sum(t, dim=-1)
        result = result.to(original_dtype)
        if dim_normalized != tensor.ndim - 1:
            result = result.transpose(dim_normalized, -1)
        return result

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_bsnd_lightning_indexer_grad(self):
        q = torch.randn(self.batch, self.seqlenQ, self.headNumQ, self.headDim).to(torch.float32)
        k = torch.randn(self.batch, self.seqlenK, self.headNumK, self.headDim).to(torch.float32)
        weights = torch.randn(self.batch, self.seqlenQ, self.headNumQ).to(torch.float32)
        valueGrad = torch.randn((self.batch, self.seqlenQ, self.indexTopk)).to(torch.float32)
        layoutStr = "BSND"
        sparseMode = 3
        isHighPrecision = True

        loss, topkIndices = self.lightning_indexer_fwd(q.clone(), k.clone(), weights.clone(), self.indexTopk, "BSND")
        dq, dk, dweights, dumpOutput, dumpWeightsOutput = self.lightning_indexer_bwd(valueGrad.npu(), q.clone().npu(),
                                                                                     k.clone().npu(),
                                                                                     weights.clone().npu(),
                                                                                     topkIndices.npu(), self.indexTopk,
                                                                                     "BSND", isHighPrecision, True)
        dq = dq.detach().cpu()
        dk = dk.detach().cpu()
        dweights = dweights.detach().cpu()

        if layoutStr == "BSND":
            dq_npu, dk_npu, dweights_npu = torch_npu.npu_lightning_indexer_grad(
                q.to(self.inputDataType).clone().npu(),
                k.to(self.inputDataType).clone().npu(),
                valueGrad.to(self.inputDataType).clone().npu(),
                topkIndices.to(torch.int32).clone().npu(),
                weights.to(self.inputDataType).clone().npu(),
                actual_seq_lengths_query=None,
                actual_seq_lengths_key=None,
                layout=layoutStr,
                sparse_mode=sparseMode
            )
        else:
            actualSeqLengthQ = torch.tensor([seqlenQ * i for i in range(1, batch + 1)], dtype=torch.int32)
            actualSeqLengthK = torch.tensor([seqlenK * i for i in range(1, batch + 1)], dtype=torch.int32)
            dq_npu, dk_npu, dweights_npu = torch_npu.npu_lightning_indexer_grad(
                q.reshape(-1, self.headNumQ, self.headDim).to(self.inputDataType).clone().npu(),
                k.reshape(-1, self.headNumK, self.headDim).to(self.inputDataType).clone().npu(),
                valueGrad.reshape(-1, self.indexTopk).to(self.inputDataType).clone().npu(),
                topkIndices.reshape(-1, self.indexTopk).to(torch.int32).clone().npu(),
                weights.reshape(-1, self.headNumQ).to(self.inputDataType).clone().npu(),
                actual_seq_lengths_query=actualSeqLengthQ.clone().npu(),
                actual_seq_lengths_key=actualSeqLengthK.clone().npu(),
                layout=layoutStr,
                sparse_mode=sparseMode
            )

        # skip res check due to aclnn
        # self.assertRtolEqual(dq_npu.detach().cpu().float().numpy(), dq.detach().float().numpy())
        # self.assertRtolEqual(dk_npu.detach().cpu().float().numpy(), dk.detach().float().numpy())
        # self.assertRtolEqual(dweights_npu.detach().cpu().float().numpy(), dweights.detach().float().numpy())


if __name__ == "__main__":
    run_tests()