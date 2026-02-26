import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestTransposeQuantBatchMatmul(TestCase):

    def supported_op_exec(self, x1, x2, scale):
        x1 = x1.transpose(0, 1)
        out = torch.matmul(x1.float(), x2.float())
        out = out.transpose(0, 1)
        out = out.reshape(out.shape[0], 1, out.shape[1] * out.shape[2])
        data = (out * scale).to(torch.int)
        output = torch.clip(data, -128, 127).to(torch.float16)
        return output

    def supported_op_exec_2(self, x1, x2):
        x1 = x1.transpose(0, 1)
        out = torch.matmul(x1.float(), x2.float())
        out = out.transpose(0, 1)
        return out.to(torch.float16)

    def supported_op_exec_3(self, x1, x2, batch_split_factor):
        x1 = x1.transpose(0, 1)
        out = torch.matmul(x1.float(), x2.float())
        out = out.transpose(0, 1)
        output = out.reshape(x1.shape[1], batch_split_factor, -1)
        output = output.transpose(0, 1)
        return output.to(torch.float16)

    @unittest.skip("Skipping test_npu_transpose_quant_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e5m2)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e5m2)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        supported_output = self.supported_op_exec(x1, x2, x1_scale, x2_scale)
        custom_output = torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.float16, 
                                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(), 
                                                            perm_x1=[1, 0, 2], perm_x2=[0, 1, 2], perm_y=[1, 0, 2]).to("cpu")
        self.assertRtolEqual(supported_output, custom_output, 0.01)

    @unittest.skip("Skipping test_npu_transpose_quant_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul_1(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e5m2)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e5m2)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        supported_output = self.supported_op_exec(x1, x2, x1_scale, x2_scale)
        custom_output = torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.bfloat16, 
                                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(), 
                                                            perm_x1=[1, 0, 2], perm_x2=[0, 1, 2], perm_y=[1, 0, 2]).to("cpu")
        self.assertRtolEqual(supported_output, custom_output, 0.01)

    @unittest.skip("Skipping test_npu_transpose_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul_2(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        supported_output = self.supported_op_exec(x1, x2, x1_scale, x2_scale)
        custom_output = torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.float16, 
                                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(), 
                                                            perm_x1=[1, 0, 2], perm_x2=[0, 1, 2], perm_y=[1, 0, 2]).to("cpu")
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @unittest.skip("Skipping test_npu_transpose_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul_3(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        supported_output = self.supported_op_exec(x1, x2, x1_scale, x2_scale)
        custom_output = torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.bfloat16, 
                                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(), 
                                                            perm_x1=[1, 0, 2], perm_x2=[0, 1, 2], perm_y=[1, 0, 2]).to("cpu")
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @unittest.skip("Skipping test_npu_transpose_quant_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul_4(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "perm_x1 should be [1, 0, 2]"):
            torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.float16,
                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(),
                                            perm_x1=[1, 1, 2], perm_x2=[0, 1, 2],
                                            perm_y=[1, 0, 2]).to("cpu")

    @unittest.skip("Skipping test_npu_transpose_quant_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul_5(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "perm_x2 should be [0, 1, 2]"):
            torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.float16,
                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(),
                                            perm_x1=[1, 0, 2], perm_x2=[1, 1, 2],
                                            perm_y=[1, 0, 2]).to("cpu")

    @unittest.skip("Skipping npu_transpose_quant_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul_6(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "perm_y should be [1, 0, 2]"):
            torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.float16,
                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(), 
                                            perm_x1=[1, 0, 2], perm_x2=[0, 1, 2],
                                            perm_y=[1, 1, 2]).to("cpu")

    @unittest.skip("Skipping npu_transpose_quant_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul_7(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "x1's type supported for float8_e5m2 or float8_e4m3fn"):
            torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.float16, 
                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(),
                                            perm_x1=[1, 0, 2], perm_x2=[0, 1, 2],
                                            perm_y=[1, 0, 2]).to("cpu")

    @unittest.skip("Skipping npu_transpose_quant_batchmatmul temporarily")
    @SupportedDevices(["Ascend950"])
    def test_npu_transpose_quant_batchmatmul_8(self, device="npu"):
        M, K, N, Batch = 32, 512, 128, 32
        x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
        x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32)
        x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "x2's type supported for float8_e5m2 or float8_e4m3fn"):
            torch_npu.npu_transpose_quant_batchmatmul(x1.npu(), x2.npu(), dtype=torch.float16, 
                                            x1_scale=x1_scale.npu(), x2_scale=x2_scale.npu(),
                                            perm_x1=[1, 0, 2], perm_x2=[0, 1, 2],
                                            perm_y=[1, 0, 2]).to("cpu")


if __name__ == "__main__":
    run_tests()
