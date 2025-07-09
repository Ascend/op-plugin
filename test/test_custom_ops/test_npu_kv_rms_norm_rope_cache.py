import random
import copy
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class TestNPUKvRmsNormRopeCache(TestCase):

    def generate_inputs(self,
                        batch_size,
                        seq_len,
                        page_num,
                        page_size,
                        quant_mode,
                        cache_mode,
                        output_mode,
                        input_dtype):
        # generate inputs
        kv = torch.randn(batch_size, 1, seq_len, 576, dtype=input_dtype)
        gamma = torch.randn(512, dtype=input_dtype)
        cos = torch.randn(batch_size, 1, seq_len, 64, dtype=input_dtype)
        sin = torch.randn(batch_size, 1, seq_len, 64, dtype=input_dtype)
        if cache_mode != "Norm":
            k_cache = torch.ones(page_num, page_size, 1,
                                 64, dtype=input_dtype) * 9
            ckv_cache = torch.ones(page_num, page_size,
                                   1, 512, dtype=input_dtype) * 9
            if "BLK" in cache_mode:
                index_shape = (
                    batch_size * ((seq_len + page_size - 1) // (page_size)),)
                index = torch.arange(
                    start=0, end=index_shape[0] * page_size, step=page_size, dtype=torch.int64)
            else:
                index_shape = (batch_size * seq_len,)
                index = torch.arange(
                    start=0, end=index_shape[0], step=1, dtype=torch.int64)
        else:
            pass
        if quant_mode == 1:
            k_cache = k_cache.to(torch.int8)
            ckv_cache = ckv_cache.to(torch.int8)
            k_rope_scale = torch.randn(64, dtype=torch.float32)
            c_kv_scale = torch.randn(512, dtype=torch.float32)
        else:
            k_rope_scale = None
            c_kv_scale = None

        # call golden
        kv = (-2 + 10) * kv - 10
        gamma = (-10 + 1000) * gamma - 1000
        sin = (0.01 + 0.01) * sin - 0.01

        return kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, cache_mode, output_mode, input_dtype

    def supported_op_exec(self,
                          kv,
                          gamma,
                          cos,
                          sin,
                          index,
                          k_cache,
                          ckv_cache,
                          k_rope_scale=None,
                          c_kv_scale=None,
                          k_rope_offset=None,
                          c_kv_offset=None,
                          epsilon=1e-05,
                          cache_mode="Norm",
                          is_output_kv=False):

        # golden function
        def round_float_to_int8(src_tensor):
            rounded_tensor = torch.round(src_tensor)
            rounded_tensor = rounded_tensor.clip(-128, 127)
            int8_tensor = rounded_tensor.to(torch.int8)
            return int8_tensor

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        batch_size, _, seq_len, _ = kv.shape
        if "PA" in cache_mode:
            block_num, block_size, _, _ = k_cache.shape
        if "BLK" in cache_mode:
            index_page_id_length = index.shape[0] // batch_size
        if k_rope_scale is None:
            quantMode = 0
            d0 = 16
        else:
            quantMode = 1
            d0 = 32

        # split input
        rms_in, rope_in = kv.split([512, 64], dim=-1)
        rms_in = rms_in.to(torch.float32)
        rope_in = rope_in.to(torch.float32)
        # calc rmsnorm
        y = rms_in / torch.sqrt(torch.mean(rms_in ** 2,
                                dim=-1, keepdim=True) + epsilon)
        y = y * gamma.to(torch.float32)
        # calc rope
        k = rope_in.view(batch_size, 1, seq_len, 32, 2).transpose(
            4, 3).reshape(batch_size, 1, seq_len, 64)
        k_embed = (k * cos.to(torch.float32)) + \
            (rotate_half(k) * sin.to(torch.float32))

        # copy outputs
        k_embed_out = copy.deepcopy(k_embed)
        k_embed_out = k_embed_out.to(kv.dtype)
        y_out = copy.deepcopy(y)
        y_out = y_out.to(kv.dtype)
        # prepare cache
        if quantMode == 1:
            k_embed = k_embed * k_rope_scale
            k_embed = round_float_to_int8(k_embed)
            y = y * c_kv_scale
            y = round_float_to_int8(y)
        else:
            k_embed = k_embed.to(k_cache.dtype)
            y = y.to(k_cache.dtype)

        # scatter update
        if cache_mode == "Norm":
            """ Norm mode """
            pass

        elif cache_mode in ("PA", "PA_BNSD"):
            k_cache = k_cache.reshape(-1, 64)
            ckv_cache = ckv_cache.reshape(-1, 512)
            for batch_id in range(batch_size):
                for token_id in range(seq_len):
                    offset = index[batch_id * seq_len + token_id]
                    if offset >= 0:
                        k_cache[offset, :] = k_embed[batch_id, 0, token_id, :]
                        ckv_cache[offset, :] = y[batch_id, 0, token_id, :]
            k_cache = k_cache.reshape(block_num, block_size, 1, 64)
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, 512)

        elif cache_mode == "PA_BLK_NZ":
            """ PA_BLK_NZ mode """
            k_cache = k_cache.reshape(block_num, 1, -1, block_size, d0)
            ckv_cache = ckv_cache.reshape(block_num, 1, -1, block_size, d0)
            for batch_id in range(batch_size):
                for tokenInCurrentBatch in range(seq_len):
                    tokenId = batch_id * seq_len + tokenInCurrentBatch
                    indexPageId = tokenInCurrentBatch // block_size
                    pageOffset = index[batch_id *
                                       index_page_id_length + indexPageId]
                    if pageOffset < 0:
                        continue
                    pageId = pageOffset // block_size
                    tokenOffsetInCurrentPage = tokenInCurrentBatch % block_size
                    k_cache[pageId, 0, :, tokenOffsetInCurrentPage,
                            :] = k_embed[batch_id, 0, tokenInCurrentBatch, :].reshape(-1, d0)
                    ckv_cache[pageId, 0, :, tokenOffsetInCurrentPage,
                              :] = y[batch_id, 0, tokenInCurrentBatch, :].reshape(-1, d0)
            k_cache = k_cache.reshape(block_num, block_size, 1, 64)
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, 512)

        elif cache_mode == "PA_BLK_BNSD":
            """ PA_BLK_BNSD mode """
            k_cache = k_cache.reshape(block_num, block_size, 1, -1)
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, -1)
            for batch_id in range(batch_size):
                for tokenInCurrentBatch in range(seq_len):
                    tokenId = batch_id * seq_len + tokenInCurrentBatch
                    indexPageId = tokenInCurrentBatch // block_size
                    pageOffset = index[batch_id *
                                       index_page_id_length + indexPageId]
                    if pageOffset < 0:
                        continue
                    pageId = pageOffset // block_size
                    tokenOffsetInCurrentPage = tokenInCurrentBatch % block_size
                    k_cache[pageId, tokenOffsetInCurrentPage, 0,
                            :] = k_embed[batch_id, 0, tokenInCurrentBatch, :]
                    ckv_cache[pageId, tokenOffsetInCurrentPage, 0,
                              :] = y[batch_id, 0, tokenInCurrentBatch, :]
            k_cache = k_cache.reshape(block_num, block_size, 1, 64)
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, 512)

        elif cache_mode == "PA_NZ":
            """ PA_NZ """
            if k_rope_scale is not None:
                d0 = 32
            else:
                d0 = 16
            k_cache = k_cache.reshape(block_num, 1, -1, block_size, d0)
            ckv_cache = ckv_cache.reshape(block_num, 1, -1, block_size, d0)
            for batch_id in range(batch_size):
                for tokenInCurrentBatch in range(seq_len):
                    pageOffset = index[batch_id *
                                       seq_len + tokenInCurrentBatch]
                    if pageOffset >= 0:
                        pageId = pageOffset // block_size
                        tokenOffsetInCurrentPage = pageOffset % block_size
                        k_cache[pageId, 0, :, tokenOffsetInCurrentPage,
                                :] = k_embed[batch_id, 0, tokenInCurrentBatch, :].reshape(-1, d0)
                        ckv_cache[pageId, 0, :, tokenOffsetInCurrentPage,
                                  :] = y[batch_id, 0, tokenInCurrentBatch, :].reshape(-1, d0)
            k_cache = k_cache.reshape(block_num, block_size, 1, 64)
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, 512)

        return k_cache, ckv_cache, k_embed_out, y_out

    def custom_op_exec(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                       k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                       epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
        kv = kv.npu()
        gamma = gamma.npu()
        cos = cos.npu()
        sin = sin.npu()
        index = index.npu()
        k_cache = k_cache.npu()
        ckv_cache = ckv_cache.npu()
        if k_rope_scale is not None:
            k_rope_scale = k_rope_scale.npu()
            c_kv_scale = c_kv_scale.npu()
        k_cache_npu, ckv_cache_npu, k_rope_npu, c_kv_npu = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                               k_cache, ckv_cache,
                                                                                               k_rope_scale=k_rope_scale,
                                                                                               c_kv_scale=c_kv_scale,
                                                                                               k_rope_offset=k_rope_offset,
                                                                                               c_kv_offset=c_kv_offset,
                                                                                               epsilon=epsilon,
                                                                                               cache_mode=cache_mode,
                                                                                               is_output_kv=is_output_kv)
        k_cache_cpu = k_cache_npu.cpu()
        ckv_cache_cpu = ckv_cache_npu.cpu()
        k_rope_cpu = k_rope_npu.cpu()
        c_kv_cpu = c_kv_npu.cpu()
        torch._dynamo.reset()
        return k_cache_cpu, ckv_cache_cpu, k_rope_cpu, c_kv_cpu

    @SupportedDevices(['Ascend910B'])
    def test_npu_kv_rmsnorm_rope_cache_PA_BNSD(self, device="npu"):
        kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, cache_mode, output_mode, input_dtype = self.generate_inputs(
            64, 1, 576, 128, 0, "PA_BNSD", False, torch.bfloat16)
        golden_out = self.supported_op_exec(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                            k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                            k_rope_offset=None, c_kv_offset=None,
                                            epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        # call npu api
        npu_out = self.custom_op_exec(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                      k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                      k_rope_offset=None, c_kv_offset=None,
                                      epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        k_cache_npu, ckv_cache_npu, k_rope_npu, c_kv_npu = npu_out
        k_cache_cpu, ckv_cache_cpu, k_rope_cpu, c_kv_cpu = golden_out

        # comparison
        if input_dtype == torch.float16:
            atol = 0.01
            rtol = 0.01
        elif input_dtype == torch.bfloat16:
            atol = 0.04
            rtol = 0.04
        else:
            atol = 1e-5
            rtol = 1e-5

        self.assertRtolEqual(k_cache_npu.cpu(), k_cache_cpu, prec=rtol)
        self.assertRtolEqual(ckv_cache_npu.cpu(), ckv_cache_cpu, prec=rtol)

        if output_mode:
            self.assertEqual(k_rope_npu.shape, k_rope_cpu.shape)
            self.assertEqual(c_kv_npu.shape, c_kv_cpu.shape)
            self.assertEqual(k_rope_npu.dtype, k_rope_cpu.dtype)
            self.assertEqual(c_kv_npu.dtype, c_kv_cpu.dtype)

            try:
                self.assertRtolEqual(k_rope_npu.cpu(), k_rope_cpu, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(k_rope_npu.cpu() - k_rope_cpu))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

            try:
                self.assertRtolEqual(c_kv_npu.cpu(), c_kv_cpu, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(c_kv_npu.cpu() - c_kv_cpu))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

    @SupportedDevices(['Ascend910B'])
    def test_npu_kv_rmsnorm_rope_cache_PA_BLK_NZ(self, device="npu"):
        kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, cache_mode, output_mode, input_dtype = self.generate_inputs(
            64, 1, 576, 128, 0, "PA_BLK_NZ", False, torch.bfloat16)
        golden_out = self.supported_op_exec(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                            k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                            k_rope_offset=None, c_kv_offset=None,
                                            epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        # call npu api
        npu_out = self.custom_op_exec(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                      k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                      k_rope_offset=None, c_kv_offset=None,
                                      epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        k_cache_npu, ckv_cache_npu, k_rope_npu, c_kv_npu = npu_out
        k_cache_cpu, ckv_cache_cpu, k_rope_cpu, c_kv_cpu = golden_out

        # comparison
        if input_dtype == torch.float16:
            atol = 0.01
            rtol = 0.01
        elif input_dtype == torch.bfloat16:
            atol = 0.04
            rtol = 0.04
        else:
            atol = 1e-5
            rtol = 1e-5

        self.assertRtolEqual(k_cache_npu.cpu(), k_cache_cpu, prec=rtol)
        self.assertRtolEqual(ckv_cache_npu.cpu(), ckv_cache_cpu, prec=rtol)

        if output_mode:
            self.assertEqual(k_rope_npu.shape, k_rope_cpu.shape)
            self.assertEqual(c_kv_npu.shape, c_kv_cpu.shape)
            self.assertEqual(k_rope_npu.dtype, k_rope_cpu.dtype)
            self.assertEqual(c_kv_npu.dtype, c_kv_cpu.dtype)

            try:
                self.assertRtolEqual(k_rope_npu.cpu(), k_rope_cpu, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(k_rope_npu.cpu() - k_rope_cpu))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

            try:
                self.assertRtolEqual(c_kv_npu.cpu(), c_kv_cpu, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(c_kv_npu.cpu() - c_kv_cpu))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

    @SupportedDevices(['Ascend910B'])
    def test_npu_kv_rmsnorm_rope_cache_PA_BLK_BNSD(self, device="npu"):
        kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, cache_mode, output_mode, input_dtype = self.generate_inputs(
            64, 1, 576, 128, 0, "PA_BLK_BNSD", False, torch.bfloat16)
        golden_out = self.supported_op_exec(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                            k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                            k_rope_offset=None, c_kv_offset=None,
                                            epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        # call npu api
        npu_out = self.custom_op_exec(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                      k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                      k_rope_offset=None, c_kv_offset=None,
                                      epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        k_cache_npu, ckv_cache_npu, k_rope_npu, c_kv_npu = npu_out
        k_cache_cpu, ckv_cache_cpu, k_rope_cpu, c_kv_cpu = golden_out

        # comparison
        if input_dtype == torch.float16:
            atol = 0.01
            rtol = 0.01
        elif input_dtype == torch.bfloat16:
            atol = 0.04
            rtol = 0.04
        else:
            atol = 1e-5
            rtol = 1e-5

        self.assertRtolEqual(k_cache_npu.cpu(), k_cache_cpu, prec=rtol)
        self.assertRtolEqual(ckv_cache_npu.cpu(), ckv_cache_cpu, prec=rtol)

        if output_mode:
            self.assertEqual(k_rope_npu.shape, k_rope_cpu.shape)
            self.assertEqual(c_kv_npu.shape, c_kv_cpu.shape)
            self.assertEqual(k_rope_npu.dtype, k_rope_cpu.dtype)
            self.assertEqual(c_kv_npu.dtype, c_kv_cpu.dtype)

            try:
                self.assertRtolEqual(k_rope_npu.cpu(), k_rope_cpu, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(k_rope_npu.cpu() - k_rope_cpu))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

            try:
                self.assertRtolEqual(c_kv_npu.cpu(), c_kv_cpu, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(c_kv_npu.cpu() - c_kv_cpu))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

    @SupportedDevices(['Ascend910B'])
    def test_npu_kv_rmsnorm_rope_cache_PA_NZ(self, device="npu"):
        kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, cache_mode, output_mode, input_dtype = self.generate_inputs(
            64, 1, 576, 128, 0, "PA_NZ", False, torch.bfloat16)
        golden_out = self.supported_op_exec(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                            k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                            k_rope_offset=None, c_kv_offset=None,
                                            epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        # call npu api
        npu_out = self.custom_op_exec(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                      k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                      k_rope_offset=None, c_kv_offset=None,
                                      epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        k_cache_npu, ckv_cache_npu, k_rope_npu, c_kv_npu = npu_out
        k_cache_cpu, ckv_cache_cpu, k_rope_cpu, c_kv_cpu = golden_out

        # comparison
        if input_dtype == torch.float16:
            atol = 0.01
            rtol = 0.01
        elif input_dtype == torch.bfloat16:
            atol = 0.04
            rtol = 0.04
        else:
            atol = 1e-5
            rtol = 1e-5

        self.assertRtolEqual(k_cache_npu.cpu(), k_cache_cpu, prec=rtol)
        self.assertRtolEqual(ckv_cache_npu.cpu(), ckv_cache_cpu, prec=rtol)

        if output_mode:
            self.assertEqual(k_rope_npu.shape, k_rope_cpu.shape)
            self.assertEqual(c_kv_npu.shape, c_kv_cpu.shape)
            self.assertEqual(k_rope_npu.dtype, k_rope_cpu.dtype)
            self.assertEqual(c_kv_npu.dtype, c_kv_cpu.dtype)

            try:
                self.assertRtolEqual(k_rope_npu.cpu(), k_rope_cpu, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(k_rope_npu.cpu() - k_rope_cpu))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

            try:
                self.assertRtolEqual(c_kv_npu.cpu(), c_kv_cpu, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(c_kv_npu.cpu() - c_kv_cpu))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")


if __name__ == "__main__":
    run_tests()
