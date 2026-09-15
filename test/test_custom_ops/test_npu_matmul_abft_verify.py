# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion, SupportedDevices


class TestNpuMatmulAbftVerify(TestCase):
    # Golden-reference strategy: correct C → comp_row all 255; corrupted rows →
    # the covering byte (8 rows/byte) non-255. Byte-value assertions are
    # bit-order independent; padding bits (rows >= M) are undefined, never asserted.

    def generate_abft_inputs(self, m, n, k, dtype):
        a = (torch.randn(m, k, dtype=torch.float32) * 0.1).to(dtype)
        b = (torch.randn(k, n, dtype=torch.float32) * 0.1).to(dtype)
        w = torch.ones(n, dtype=dtype)
        # golden C: computed in float64 so reference rounding cannot cause a false alarm
        c = torch.matmul(a.to(torch.float64), b.to(torch.float64)).to(torch.float32)
        return a, b, c, w

    def cpu_op_exec(self, m, n):
        """Expected comp_row for a correct C: all 255."""
        return np.full((m + 7) // 8 * ((n + 255) // 256), 255, dtype=np.uint8)

    def npu_op_exec(self, a, b, c, w, e_max=None):
        a, b, c, w = a.npu(), b.npu(), c.npu(), w.npu()
        if e_max is None:
            out = torch_npu._npu_matmul_abft_verify(a, b, c, w)
        else:
            out = torch_npu._npu_matmul_abft_verify(a, b, c, w, e_max=e_max)
        return out.cpu().numpy()

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_basic_fp16(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(256, 256, 256, torch.float16)
        expected = self.cpu_op_exec(256, 256)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_basic_bf16(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(256, 256, 256, torch.bfloat16)
        expected = self.cpu_op_exec(256, 256)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_basic_fp32(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(256, 256, 256, torch.float32)
        expected = self.cpu_op_exec(256, 256)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_e_max_default(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_e_max_explicit_bf16(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.bfloat16)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w, e_max=0.001)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_e_max_explicit_fp32(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w, e_max=0.00002)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_ceil_boundary_shape(self, device="npu"):
        torch.manual_seed(0)
        # M%8 and N%256 both non-zero: the trailing byte's padding bits are undefined → assert only its valid bits
        m, n, k = 33, 100, 67
        a, b, c, w = self.generate_abft_inputs(m, n, k, torch.float16)
        expected = self.cpu_op_exec(m, n)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertRtolEqual(expected[:4], actual[:4])
        self.assertTrue(actual[4] > 0)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_large_scale(self, device="npu"):
        torch.manual_seed(0)
        # CANN-recommended matrix-op scale (M, N, K >= 1024)
        a, b, c, w = self.generate_abft_inputs(1024, 1024, 1024, torch.bfloat16)
        expected = self.cpu_op_exec(1024, 1024)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_error_detection(self, device="npu"):
        torch.manual_seed(0)
        # corrupt rows 0..7 → byte 0 must be all-zero bits, others all 255
        m, n, k = 128, 128, 128
        a, b, c, w = self.generate_abft_inputs(m, n, k, torch.float16)
        c[0:8, :] += 50.0
        expected = self.cpu_op_exec(m, n)
        expected[0] = 0
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_output_shape_dtype(self, device="npu"):
        torch.manual_seed(0)
        # splitN=2 and in-byte bit order undefined → full bytes asserted 255, padding bytes only counted (non-255 <= 2)
        m, n, k = 37, 300, 129
        a, b, c, w = self.generate_abft_inputs(m, n, k, torch.float16)
        expected = self.cpu_op_exec(m, n)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, torch.Size([10]))
        self.assertEqual(actual.dtype, np.uint8)
        self.assertGreaterEqual(np.count_nonzero(actual == 255), actual.size - 2)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_a_invalid_dtype(self, device="npu"):
        a = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        b = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.randint(0, 10, (128,), dtype=torch.int64).npu()
        with self.assertRaisesRegex(RuntimeError, "only support float16/bfloat16/float32"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_b_invalid_dtype(self, device="npu"):
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_c_invalid_dtype(self, device="npu"):
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "c only supports float32"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_checksum_weight_invalid_dtype(self, device="npu"):
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.randint(0, 10, (128,), dtype=torch.int64).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_a_invalid_dim(self, device="npu"):
        a = torch.randn(2, 128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "a must be a 2D tensor"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_k_mismatch(self, device="npu"):
        a = torch.randn(128, 64, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "must be equal to the K dim of b"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_c_shape_mismatch(self, device="npu"):
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 127, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "c must have shape"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_checksum_weight_shape(self, device="npu"):
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(127, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "checksum_weight must have shape"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_dtype_linkage(self, device="npu"):
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.bfloat16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_e_max_negative(self, device="npu"):
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "must be non-negative"):
            torch_npu._npu_matmul_abft_verify(a.npu(), b.npu(), c.npu(), w.npu(), e_max=-0.001)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_neg_non_contiguous(self, device="npu"):
        # slice on the NPU: a CPU-side slice + .npu() would be re-contiguous after the H2D copy
        a_base = torch.randn(128, 256, dtype=torch.float16).npu()
        a = a_base[:, :128]
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "only support contiguous tensors"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)


if __name__ == "__main__":
    run_tests()
