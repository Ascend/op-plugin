import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCopysign(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.copysign(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.copysign(input1, input2)
        self.assertEqual(output.dtype, torch.copysign(input1.cpu(), input2.cpu()).dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, output):
        torch.copysign(input1, input2, out=output)
        self.assertEqual(output.dtype, torch.copysign(input1.cpu(), input2.cpu()).dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1, input2):
        input1.copysign_(input2)
        # inplace 不改变 dtype，与 CPU functional 参考的提升结果一致即可
        self.assertEqual(input1.dtype, torch.copysign(input1.cpu(), input2.cpu()).dtype)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def npu_scalar_op_exec(self, input1, scalar):
        output = torch.copysign(input1, scalar)
        self.assertEqual(output.dtype, torch.copysign(input1.cpu(), scalar).dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_inp_op_exec_scalar(self, input1, scalar):
        expected_dtype = torch.copysign(input1.cpu(), scalar).dtype
        input1.copysign_(scalar)
        self.assertEqual(input1.dtype, expected_dtype)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar_out(self, input1, scalar, output):
        torch.copysign(input1, scalar, out=output)
        self.assertEqual(output.dtype, torch.copysign(input1.cpu(), scalar).dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def copysign_result(self, shape_format, inplace=False):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item, -10, 10)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output = npu_output.astype(cpu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

            if inplace:
                npu_output_inp = self.npu_inp_op_exec(npu_input1.clone(), npu_input2)
                npu_output_inp = npu_output_inp.astype(cpu_output.dtype)
                self.assertRtolEqual(cpu_output, npu_output_inp)

    def test_copysign_shape_format_fp16(self):
        format_list = [0, 3]
        shape_format = [[np.float16, i, j] for i in format_list for j in [[18], [5, 256], [32, 3, 3]]]
        self.copysign_result(shape_format, inplace=True)

    def test_copysign_shape_format_fp32(self):
        format_list = [0, 3]
        shape_format = [[np.float32, i, j] for i in format_list for j in [[18], [5, 256], [32, 3, 3], [64, 112, 7, 7]]]
        self.copysign_result(shape_format, inplace=True)

    def test_copysign_shape_format_int32(self):
        # 整型输入会被提升为 float32（promotes_int_to_float）
        format_list = [0]
        shape_format = [[np.int32, i, j] for i in format_list for j in [[18], [5, 256], [32, 3, 3]]]
        self.copysign_result(shape_format, inplace=False)

    def test_copysign_broadcast(self):
        cpu_input1 = torch.randn(3, 4, dtype=torch.float32)
        cpu_input2 = torch.randn(4, dtype=torch.float32)
        cpu_output = torch.copysign(cpu_input1, cpu_input2).numpy()
        npu_output = torch.copysign(cpu_input1.npu(), cpu_input2.npu()).cpu().numpy()
        self.assertRtolEqual(cpu_output, npu_output)

        cpu_input1 = torch.randn(4, dtype=torch.float32)
        cpu_input2 = torch.randn(3, 1, 4, dtype=torch.float32)
        cpu_output = torch.copysign(cpu_input1, cpu_input2).numpy()
        npu_output = torch.copysign(cpu_input1.npu(), cpu_input2.npu()).cpu().numpy()
        self.assertRtolEqual(cpu_output, npu_output)

    def test_copysign_out(self):
        for shape1, shape2 in [((5, 6), (5, 6)), ((3, 4), (4,))]:
            cpu_input1 = torch.randn(*shape1, dtype=torch.float32)
            cpu_input2 = torch.randn(*shape2, dtype=torch.float32)
            cpu_output = torch.copysign(cpu_input1, cpu_input2).numpy()
            npu_input1 = cpu_input1.npu()
            npu_input2 = cpu_input2.npu()
            npu_out = torch.empty(npu_input1.shape, dtype=torch.float32, device="npu")
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_copysign_scalar(self):
        for dtype in [torch.float32, torch.float64]:
            cpu_input = torch.randn(5, 6, dtype=dtype)
            npu_input = cpu_input.npu()
            for scalar in [2.5, -2.5, 0.0, -0.0, float("inf"), float("-inf"), float("nan")]:
                cpu_output = torch.copysign(cpu_input, scalar).numpy()
                npu_output = self.npu_scalar_op_exec(npu_input, scalar)
                self.assertRtolEqual(cpu_output, npu_output)

        # 整型标量 + 整型张量：结果提升为 float32
        cpu_input = torch.randint(-10, 10, (5, 6), dtype=torch.int32)
        cpu_output = torch.copysign(cpu_input, -3).numpy()
        npu_output = self.npu_scalar_op_exec(cpu_input.npu(), -3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_copysign_scalar_inplace_out(self):
        # copysign_.Scalar 与 copysign.Scalar_out 两种标量变体
        cpu_input = torch.randn(5, 6, dtype=torch.float32)
        npu_input = cpu_input.npu()
        for scalar in [2.5, -2.5, 0.0, -0.0, float("inf"), float("-inf"), float("nan")]:
            cpu_inp = cpu_input.clone()
            cpu_inp.copysign_(scalar)
            npu_output_inp = self.npu_inp_op_exec_scalar(npu_input.clone(), scalar)
            self.assertRtolEqual(cpu_inp.numpy(), npu_output_inp)

            cpu_out = torch.empty_like(cpu_input)
            torch.copysign(cpu_input, scalar, out=cpu_out)
            npu_out = torch.empty(5, 6, dtype=torch.float32, device="npu")
            npu_output = self.npu_op_exec_scalar_out(npu_input, scalar, npu_out)
            self.assertRtolEqual(cpu_out.numpy(), npu_output)

    def test_copysign_inplace_int_error(self):
        # inplace 整型提升与 CPU 一致地报错
        cpu_input = torch.randint(-10, 10, (5, 6), dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            cpu_input.copysign_(3)
        npu_input = torch.randint(-10, 10, (5, 6), dtype=torch.int32, device="npu")
        with self.assertRaises(RuntimeError):
            npu_input.copysign_(3)

    def test_copysign_cpu_scalar_tensor(self):
        # other 为 0 维 CPU 标量张量：按弱标量参与提升，结果保持 fp32
        cpu_input = torch.randn(5, 6, dtype=torch.float32)
        cpu_scalar = torch.tensor(-2.0)
        cpu_output = torch.copysign(cpu_input, cpu_scalar)
        npu_output = torch.copysign(cpu_input.npu(), cpu_scalar)
        self.assertEqual(npu_output.dtype, cpu_output.dtype)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_copysign_dtype_promotion(self):
        # 整型输入提升为 float32，bool 输入同样提升
        a = torch.tensor([1, -2, 3], dtype=torch.int32)
        b = torch.tensor([-1, 1, -1], dtype=torch.int32)
        cpu_out = torch.copysign(a, b)
        npu_out = torch.copysign(a.npu(), b.npu()).cpu()
        self.assertTrue(cpu_out.dtype == torch.float32)
        self.assertTrue(npu_out.dtype == cpu_out.dtype)
        self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

        a = torch.tensor([True, False, True])
        b = torch.tensor([True, True, False])
        cpu_out = torch.copysign(a, b)
        npu_out = torch.copysign(a.npu(), b.npu()).cpu()
        self.assertTrue(cpu_out.dtype == torch.float32)
        self.assertTrue(npu_out.dtype == cpu_out.dtype)
        self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

    def test_copysign_bf16(self):
        # bf16 走 int16 位宽路径（copysign_bits_type -> kShort），与 CPU 位级对齐
        # ±0/±inf 的符号位逐位比对
        cases = [0.0, -0.0, float("inf"), float("-inf")]
        for case in cases:
            a = torch.tensor([case], dtype=torch.bfloat16)
            b = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
            cpu_bits = torch.copysign(a, b).view(torch.int16).numpy()
            npu_bits = torch.copysign(a.npu(), b.npu()).cpu().view(torch.int16).numpy()
            self.assertTrue((cpu_bits == npu_bits).all(), msg=f"bf16 case={case} as self cpu_bits={cpu_bits} npu_bits={npu_bits}")
            cpu_bits = torch.copysign(b, a).view(torch.int16).numpy()
            npu_bits = torch.copysign(b.npu(), a.npu()).cpu().view(torch.int16).numpy()
            self.assertTrue((cpu_bits == npu_bits).all(), msg=f"bf16 case={case} as other cpu_bits={cpu_bits} npu_bits={npu_bits}")

        a = torch.tensor([float("nan")], dtype=torch.bfloat16)
        b = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
        # NaN 作为 other：符号位取自 NaN，结果与 CPU 逐位一致
        cpu_bits = torch.copysign(b, a).view(torch.int16).numpy()
        npu_bits = torch.copysign(b.npu(), a.npu()).cpu().view(torch.int16).numpy()
        self.assertTrue((cpu_bits == npu_bits).all(), msg=f"bf16 nan as other cpu_bits={cpu_bits} npu_bits={npu_bits}")
        # NaN 作为 self：copysign(|nan|, ±1.0) 的符号位按 IEEE-754 应取自 other，结果为
        # [+nan, -nan]。但 CPU 的 bf16 路径不传播该符号位（fp32 路径会传播，参见
        # test_copysign_nan_payload），属 CPU bf16 的版本相关差异，故不逐位对 CPU 比对，
        # 直接按 IEEE 断言 NPU 结果 = |nan| 幅度位 | other 符号位。
        nan_mag = int(a.view(torch.int16).item()) & 0x7FFF
        expected_bits = [nan_mag, nan_mag | 0x8000]
        npu_bits = torch.copysign(a.npu(), b.npu()).cpu().view(torch.int16).numpy()
        self.assertTrue([int(x) & 0xFFFF for x in npu_bits] == expected_bits,
                        msg=f"bf16 nan as self npu_bits={npu_bits} expected={expected_bits}")

        # 随机值：dtype 保持 bf16，位级一致
        x = torch.randn(5, 6).to(torch.bfloat16)
        y = torch.randn(5, 6).to(torch.bfloat16)
        cpu_out = torch.copysign(x, y)
        npu_out = torch.copysign(x.npu(), y.npu()).cpu()
        self.assertEqual(npu_out.dtype, cpu_out.dtype)
        self.assertTrue((cpu_out.view(torch.int16).numpy() == npu_out.view(torch.int16).numpy()).all())

    def test_copysign_promotion_int_more(self):
        # int8 / uint8 / int64 输入均提升为 float32（promotes_int_to_float），与 CPU 对齐
        # 显式符号向量：copysign 取 |self|、附 other 符号位
        a8 = torch.tensor([1, -2, 3, -4], dtype=torch.int8)
        b8 = torch.tensor([-1, 1, -1, 1], dtype=torch.int8)
        a64 = torch.tensor([1, -2, 3, -4], dtype=torch.int64)
        b64 = torch.tensor([-1, 1, -1, 1], dtype=torch.int64)
        a_u8 = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        b_i8 = torch.tensor([-1, 1, -1, 1], dtype=torch.int8)  # uint8 x int8 -> float32

        for a, b in [(a8, b8), (a64, b64), (a_u8, b_i8)]:
            cpu_out = torch.copysign(a, b)
            npu_out = torch.copysign(a.npu(), b.npu()).cpu()
            self.assertEqual(npu_out.dtype, cpu_out.dtype)
            self.assertTrue(cpu_out.dtype == torch.float32, f"{a.dtype} x {b.dtype} should promote to float32")
            self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

        # 随机矩阵：各 dtype 与 CPU 对齐（uint8 用非负范围）
        for a_dtype, b_dtype in [(torch.int8, torch.int8), (torch.int64, torch.int64),
                                 (torch.uint8, torch.int8)]:
            a_lo, a_hi = (0, 255) if a_dtype == torch.uint8 else (-10, 10)
            ra = torch.randint(a_lo, a_hi, (5, 6), dtype=a_dtype)
            rb = torch.randint(-10, 10, (5, 6), dtype=b_dtype)
            cpu_out = torch.copysign(ra, rb)
            npu_out = torch.copysign(ra.npu(), rb.npu()).cpu()
            self.assertEqual(npu_out.dtype, cpu_out.dtype)
            self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

    def test_copysign_default_dtype(self):
        # 整型输入提升应遵循 torch.set_default_dtype（promote_integer_inputs_to_float
        # 用的是 get_default_dtype，而非硬编码 float32）。NPU 不支持 double，故用
        # 受支持的 float16 验证提升逻辑；float64 场景应明确报错。
        a = torch.tensor([1, -2, 3], dtype=torch.int32)
        b = torch.tensor([-1, 1, -1], dtype=torch.int32)
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            cpu_out = torch.copysign(a, b)
            npu_out = torch.copysign(a.npu(), b.npu()).cpu()
            self.assertEqual(cpu_out.dtype, torch.float16)
            self.assertEqual(npu_out.dtype, cpu_out.dtype)
            self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())
        finally:
            torch.set_default_dtype(old_dtype)

        # NPU 不支持 double，整型输入在 default dtype 为 float64 时应明确报错，
        # 而不是静默降精度产生错误结果。三条会触发 double 的路径都应命中 guard：
        #   int32×int32（get_default_dtype 路径）、int32×float标量（result_type 标量路径）、
        #   int64×int64（确认非 int32 也生效）。
        a64 = torch.tensor([1, -2, 3], dtype=torch.int64)
        b64 = torch.tensor([-1, 1, -1], dtype=torch.int64)
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with self.assertRaises(RuntimeError):
                torch.copysign(a.npu(), b.npu())
            with self.assertRaises(RuntimeError):
                torch.copysign(a.npu(), 2.5)
            with self.assertRaises(RuntimeError):
                torch.copysign(a64.npu(), b64.npu())
        finally:
            torch.set_default_dtype(old_dtype)

    def test_copysign_special_values(self):
        # 边界值：+0/-0、+inf/-inf、nan 作为 self 或 other
        # 位级断言：±0/±NaN 的符号位与 NaN payload 必须逐位一致（assertRtolEqual 对
        # NaN 相减与 -0/+0 数值相等均无法区分，故用 int32 位模式比对）
        cases = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
        for case in cases:
            a = torch.tensor([case], dtype=torch.float32)
            b = torch.tensor([1.0, -1.0], dtype=torch.float32)

            cpu_bits = torch.copysign(a, b).view(torch.int32).numpy()
            npu_bits = torch.copysign(a.npu(), b.npu()).cpu().view(torch.int32).numpy()
            self.assertTrue((cpu_bits == npu_bits).all(), msg=f"case={case} as self")

            cpu_bits = torch.copysign(b, a).view(torch.int32).numpy()
            npu_bits = torch.copysign(b.npu(), a.npu()).cpu().view(torch.int32).numpy()
            self.assertTrue((cpu_bits == npu_bits).all(), msg=f"case={case} as other")

    def test_copysign_zero_sign(self):
        # copysign(x, +0) = +|x|，copysign(x, -0) = -|x|，需区分符号位
        x = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.float32)
        y = torch.tensor([0.0, -0.0, -0.0, 0.0], dtype=torch.float32)
        out = torch.copysign(x.npu(), y.npu()).cpu().numpy()
        self.assertTrue(out[0] == 1.0 and not np.signbit(out[0]))
        self.assertTrue(out[1] == -1.0 and np.signbit(out[1]))
        self.assertTrue(out[2] == 0.0 and np.signbit(out[2]))
        self.assertTrue(out[3] == 0.0 and not np.signbit(out[3]))

    def test_copysign_nan_sign(self):
        # IEEE-754: copysign 需保留 NaN 的符号位（regression for pytorch#181804）
        pos_nan_bits = torch.tensor([0x7FC00000], dtype=torch.uint32)
        neg_nan_bits = torch.tensor([0xFFC00000], dtype=torch.uint32)
        pos_nan = pos_nan_bits.view(torch.float32)
        neg_nan = neg_nan_bits.view(torch.float32)
        mag = torch.tensor([1.0], dtype=torch.float32)

        result_pos = torch.copysign(mag.npu(), pos_nan.npu()).cpu().item()
        result_neg = torch.copysign(mag.npu(), neg_nan.npu()).cpu().item()
        self.assertTrue(result_pos == 1.0 and not np.signbit(result_pos))
        self.assertTrue(result_neg == -1.0 and np.signbit(result_neg))

    def test_copysign_nan_payload(self):
        # 位精确：NaN payload 必须逐位保留（整数位操作实现可保证；浮点组合实现会改写 payload）
        pos_nan = torch.tensor([0x7FC00000], dtype=torch.uint32).view(torch.float32)
        neg_nan = torch.tensor([0xFFC00000], dtype=torch.uint32).view(torch.float32)
        one = torch.tensor([1.0], dtype=torch.float32)
        for self_t in (pos_nan, neg_nan):
            for other_t in (one, -one, torch.tensor([-0.0])):
                cpu_bits = torch.copysign(self_t, other_t).view(torch.int32).item()
                npu_bits = torch.copysign(self_t.npu(), other_t.npu()).cpu().view(torch.int32).item()
                self.assertEqual(npu_bits, cpu_bits)

    def test_copysign_mixed_dtype_out_inplace(self):
        # out=/inplace 的混合 dtype 规则与 CPU 对齐：浮点间窄化允许，浮点->整型拒绝
        cpu16 = torch.randn(5, 6, dtype=torch.float16)
        npu16 = cpu16.npu()
        cpu32 = torch.randn(5, 6, dtype=torch.float32)
        npu32 = cpu32.npu()

        # out 变体：fp32 out + fp16 输入（提升后窄化）
        cpu_out = torch.empty(5, 6, dtype=torch.float32)
        torch.copysign(cpu16, cpu16, out=cpu_out)
        npu_out = torch.empty(5, 6, dtype=torch.float32, device="npu")
        torch.copysign(npu16, npu16, out=npu_out)
        self.assertEqual(npu_out.dtype, cpu_out.dtype)
        self.assertRtolEqual(cpu_out.numpy(), npu_out.cpu().numpy())

        # inplace：fp16 self + fp32 other（CPU 允许，结果窄化回 fp16）
        cpu_inp = cpu16.clone()
        cpu_inp.copysign_(cpu32)
        npu_inp = npu16.clone()
        npu_inp.copysign_(npu32)
        self.assertEqual(npu_inp.dtype, cpu_inp.dtype)
        self.assertRtolEqual(cpu_inp.numpy().astype(np.float32), npu_inp.cpu().numpy().astype(np.float32))

        # inplace：int32 self 提升为 fp32，CPU 拒绝，NPU 须一致拒绝
        cpu_int = torch.randint(-9, 9, (5, 6), dtype=torch.int32)
        npu_int = cpu_int.npu()
        with self.assertRaises(RuntimeError):
            cpu_int.copysign_(3)
        with self.assertRaises(RuntimeError):
            npu_int.copysign_(3)

        # out：fp32 输入 + int32 out，CPU 拒绝，NPU 须一致拒绝
        with self.assertRaises(RuntimeError):
            torch.copysign(cpu32, cpu32, out=torch.empty(5, 6, dtype=torch.int32))
        with self.assertRaises(RuntimeError):
            torch.copysign(npu32, npu32, out=torch.empty(5, 6, dtype=torch.int32, device="npu"))


if __name__ == "__main__":
    run_tests()
