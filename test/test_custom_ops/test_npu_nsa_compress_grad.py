# Copyright (c) 2025, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuCompressGrad(TestCase):
    
    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, grad, input_cpu, weight, compress_block_size, compress_stride, actual_seq_len):
        def compress_sample_bwd(outputGrad, input_cpu, weight, block_size, compressBlockStride):
            C, N, D = outputGrad.shape
            T, _, _ = input_cpu.shape
            W = block_size
            S = compressBlockStride
            outputGrad = outputGrad.permute(2, 1, 0)
            input_cpu = input_cpu.permute(2, 1, 0)
            grad_kv = torch.zeros(D, N, T, dtype=input_cpu.dtype)
            for w in range(W):
                weight_w = weight[w, :].reshape(1, N, 1)  # [D, N, C]
                target_indices = torch.arange(C) * S + w
                contribution = outputGrad * weight_w # [D, N, C]
                grad_kv.index_add_(2, target_indices, contribution)
            grad_kv = grad_kv.permute(2, 1, 0)
            kv_unfold = input_cpu.unfold(dimension=2, size=W, step=S)
            assert kv_unfold.shape[2] == C
            grad_kv_cmp_expanded = outputGrad.unsqueeze(-1)
            product = grad_kv_cmp_expanded * kv_unfold
            grad_weight = product.sum(dim=(0, 2)).permute(1, 0)
            return grad_kv, grad_weight

        # pylint:disable = huawei-too-many-arguments
        def compress_tnd_bwd(outputGrad, input_cpu, weight, compressBlockSize, compressBlockStride, actSeqLenOptional):
            _dtype = outputGrad.dtype
            outputGrad = outputGrad.to(torch.float32)
            input_cpu = input_cpu.to(torch.float32)
            weight = weight.to(torch.float32)
            grad_input = torch.zeros_like(input_cpu, dtype=torch.float32)
            grad_weight = torch.zeros_like(weight, dtype=torch.float32)
            blc_start = 0
            for i in range(len(actSeqLenOptional) - 1):
                nblocks = (actSeqLenOptional[i + 1] - actSeqLenOptional[i] - compressBlockSize) // compressBlockStride + 1
                if nblocks <= 0:
                    continue
                lens = nblocks * compressBlockStride + (compressBlockSize - compressBlockStride)
                input_trunc = input_cpu[actSeqLenOptional[i]:actSeqLenOptional[i] + lens, :, :]
                grad_out_trunc = outputGrad[blc_start:blc_start + nblocks, :, :]
                blc_start += nblocks
                grad_in, grad_wt = compress_sample_bwd(grad_out_trunc, input_trunc, weight, compressBlockSize, compressBlockStride)
                grad_input[actSeqLenOptional[i]:actSeqLenOptional[i] + lens, :, :] += grad_in
                grad_weight += grad_wt
            grad_input = grad_input.to(_dtype)
            grad_weight = grad_weight.to(_dtype)

            return grad_input, grad_weight

        return compress_tnd_bwd(grad, input_cpu, weight, compress_block_size, compress_stride, actual_seq_len)

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, grad, input_npu, weight, compress_block_size, compress_stride, actual_seq_len):
        output = torch_npu.npu_nsa_compress_grad(grad, input_npu, weight, compress_block_size, compress_stride, actual_seq_len=actual_seq_len)
        return output

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_compress_grad(self):
        grad = torch.randn([1, 128, 192], dtype=torch.float16)
        input_ori = torch.randn([49, 128, 192], dtype=torch.float16)
        weight = torch.randn([32, 128], dtype=torch.float16)
        actual_seq_len = [0, 5, 49]
        compress_block_size = 32
        compress_stride = 16

        npuout = self.npu_op_exec(grad.npu(), input_ori.npu(), weight.npu(), compress_block_size, compress_stride, actual_seq_len)
        gloden_out = self.cpu_op_exec(grad, input_ori, weight, compress_block_size, compress_stride, actual_seq_len)

        self.assertRtolEqual(gloden_out[0].numpy(), npuout[0].cpu().numpy())
        self.assertRtolEqual(gloden_out[1].numpy(), npuout[1].cpu().numpy())

if __name__ == "__main__":
    run_tests()
