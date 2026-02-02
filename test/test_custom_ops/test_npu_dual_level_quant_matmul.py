import math
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUDualLevelQuantMatmul(TestCase):
    def pack_b4_to_b8(self, b4_data: np.ndarray):
        """
        pack b4 numpy array to int8 numpy array
        """
        packed_shape = [b4_data.shape[0], int(b4_data.shape[1] / 2)]
        pack_size = 2
        shift = np.array([0, 4], dtype=np.int8)
        if b4_data.size % pack_size != 0:
            b4_data = np.pad(b4_data.flatten(), (0, pack_size - b4_data.size % pack_size), 'constant')
        b4_data = b4_data.reshape(-1, 2).view(np.int8)
        return np.sum(np.bitwise_and(b4_data, 0b00001111) << shift, axis=1, dtype=np.int8).reshape(packed_shape)

    def mx_mmad(self, x1, x2, x1_level1_scale, x2_level1_scale, l1_group_size=32):
        x1_level1_scale_broadcast = np.repeat(x1_level1_scale, l1_group_size, axis=-1)
        x2_level1_scale_broadcast = np.repeat(x2_level1_scale, l1_group_size, axis=-2)
        x1 = x1 * x1_level1_scale_broadcast
        x2 = x2 * x2_level1_scale_broadcast
        return np.matmul(x1, x2)

    def expected_output(self, x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale, x2_level1_scale, bias=None,
                        l0_group_size=512, l1_group_size=32):
        from ml_dtypes import float8_e8m0fnu, bfloat16
        x1 = x1.astype(np.float32)
        x2 = x2.astype(np.float32).T

        x1_level1_scale = x1_level1_scale.reshape(x1_level1_scale.shape[0],
                                                  x1_level1_scale.shape[1] * x1_level1_scale.shape[2])
        x2_level1_scale = x2_level1_scale.reshape(x2_level1_scale.shape[0],
                                                  x2_level1_scale.shape[1] * x2_level1_scale.shape[2]).T

        x1_level1_scale = (2 ** (x1_level1_scale.astype(np.float32) - 127)).astype(float8_e8m0fnu)
        x2_level1_scale = (2 ** (x2_level1_scale.astype(np.float32) - 127)).astype(float8_e8m0fnu)

        m, k, n = x1.shape[0], x1.shape[1], x2.shape[1]
        out = np.zeros((m, n), dtype=np.float32)
        k_loop_num = math.ceil(k / l0_group_size)
        for i in range(k_loop_num):
            k_start = int(i * l0_group_size)
            k_end = min(int((i + 1) * l0_group_size), k)

            x1_i = x1[:, k_start:k_end]
            x2_i = x2[k_start:k_end, :]

            x1_level0_scale_i = x1_level0_scale[:, i:i + 1]
            x2_level0_scale_i = x2_level0_scale[i:i + 1, :]

            l1_scale_k_start = int(k_start / l1_group_size)
            l1_scale_k_end = int(k_end / l1_group_size)

            x1_level1_scale_i = x1_level1_scale[:, l1_scale_k_start:l1_scale_k_end]
            x2_level1_scale_i = x2_level1_scale[l1_scale_k_start:l1_scale_k_end, :]

            out += self.mx_mmad(x1_i, x2_i, x1_level1_scale_i, x2_level1_scale_i,
                                l1_group_size) * x1_level0_scale_i * x2_level0_scale_i

        if bias is not None:
            out += bias.astype(np.float32)

        return out.astype(bfloat16).astype(np.float32)

    @SupportedDevices(['Ascend950'])
    def test_npu_dual_level_quant_matmul(self, device="npu"):
        from ml_dtypes import float4_e2m1fn
        torch.manual_seed(0)

        m, k, n = 256, 1024, 2048
        l0_group_size = 512
        l1_group_size = 32

        x1 = np.random.randint(-6, 6, (m, k)).astype(float4_e2m1fn)
        x2 = np.random.randint(-6, 6, (n, k)).astype(float4_e2m1fn)

        x1_level0_scale = np.random.uniform(-1, 1, (m, math.ceil(k / l0_group_size))).astype(np.float32)
        x2_level0_scale = np.random.uniform(-1, 1, (math.ceil(k / l0_group_size), n)).astype(np.float32)

        x1_level1_scale = np.random.randint(124, 130, (m, math.ceil(k / l1_group_size / 2), 2), dtype=np.uint8)
        x2_level1_scale = np.random.randint(124, 130, (n, math.ceil(k / l1_group_size / 2), 2), dtype=np.uint8)

        bias = np.random.uniform(-1, 1, (n,)).astype(np.float32)

        x1_npu = torch.from_numpy(self.pack_b4_to_b8(x1)).npu()
        x2_npu = torch.from_numpy(self.pack_b4_to_b8(x2)).npu()
        x2_npu = torch_npu.npu_format_cast(x2_npu, 29, torch.int8)

        x1_level0_scale_npu = torch.from_numpy(x1_level0_scale).npu()
        x2_level0_scale_npu = torch.from_numpy(x2_level0_scale).npu()
        x1_level1_scale_npu = torch.from_numpy(x1_level1_scale).npu()
        x2_level1_scale_npu = torch.from_numpy(x2_level1_scale).npu()
        bias_npu = torch.from_numpy(bias).npu()

        expected_out = self.expected_output(x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale, x2_level1_scale,
                                            bias)

        custom_out = torch_npu.npu_dual_level_quant_matmul(x1_npu, x2_npu, x1_level0_scale_npu, x2_level0_scale_npu,
                                                           x1_level1_scale_npu, x2_level1_scale_npu, bias=bias_npu,
                                                           output_dtype=torch.bfloat16)

        self.assertRtolEqual(expected_out, custom_out.float().cpu().numpy(), 0.001)


if __name__ == "__main__":
    run_tests()
