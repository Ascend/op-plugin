import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]

# launch 数上界（RFC §2.2 / §4.1，按实测值取上界，不做精确断言）
LAUNCH_UPPER_BOUND = {
    "float_same": 4,
    "float_mixed": 6,
    "float16_bf16": 8,
    "int_same": 1,
    "int_mixed": 2,
}


def _nan(shape, dtype=torch.float32):
    tensor = torch.randn(shape, dtype=dtype)
    if tensor.numel() > 0:
        flat = tensor.reshape(-1)
        flat[0] = float("nan")
        flat[-1] = float("nan")
    return tensor


def _pair_tensor(dtype, shape):
    """按 dtype 生成含 NaN 的浮点 / 整数 / bool 输入。"""
    if dtype.is_floating_point:
        return _nan(shape, torch.float32).to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool)
    low = 0 if dtype == torch.uint8 else -5
    return torch.randint(low, 5, shape, dtype=dtype)


def _count_device_kernels(prof):
    """统计 device kernel 数，剔除 aten::* 等 host 侧事件（RFC 分析文档 §1 的坑）。

    NPU 侧 device kernel 事件名以 aclnn 开头；探测不到 aclnn* 事件时返回 0，
    由调用方跳过断言——退化口径（剔除 host 事件后的条目数）会把 runtime/memcpy
    等非 kernel 事件计入，计数不可信，据此断言会产生误报或空过。
    """
    items = list(prof.key_averages())
    kernels = [item for item in items if item.key.startswith("aclnn")]
    return sum(item.count for item in kernels)


class TestFmax(TestCase):

    def _assert_same(self, npu_out, cpu_out):
        self.assertEqual(npu_out.dtype, cpu_out.dtype)
        torch.testing.assert_close(npu_out.cpu(), cpu_out, rtol=0, atol=0, equal_nan=True)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_nan_matrix(self):
        nan = float("nan")
        inf = float("inf")
        cases = [
            ([nan, 1.0, 2.0], [1.0, 2.0, 3.0]),
            ([1.0, 2.0, 3.0], [nan, 2.0, 3.0]),
            ([nan, nan], [nan, nan]),
            ([inf, -inf, 0.0], [-inf, inf, -0.0]),
            ([nan, inf], [inf, nan]),
        ]
        for data_a, data_b in cases:
            cpu_a = torch.tensor(data_a, dtype=torch.float32)
            cpu_b = torch.tensor(data_b, dtype=torch.float32)
            cpu_out = torch.fmax(cpu_a, cpu_b)
            npu_out = torch.fmax(cpu_a.npu(), cpu_b.npu())
            self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_float_dtypes(self):
        for dtype in FLOAT_DTYPES:
            cpu_a = _nan((5, 6), dtype)
            cpu_b = _nan((5, 6), dtype)
            cpu_out = torch.fmax(cpu_a, cpu_b)
            npu_out = torch.fmax(cpu_a.npu(), cpu_b.npu())
            self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_integer_and_bool(self):
        for dtype in INT_DTYPES:
            low = 0 if dtype == torch.uint8 else -8
            cpu_a = torch.randint(low, 8, (5, 6), dtype=dtype)
            cpu_b = torch.randint(low, 8, (5, 6), dtype=dtype)
            cpu_out = torch.fmax(cpu_a, cpu_b)
            npu_out = torch.fmax(cpu_a.npu(), cpu_b.npu())
            self._assert_same(npu_out, cpu_out)

        cpu_a = torch.randint(0, 2, (4, 4), dtype=torch.bool)
        cpu_b = torch.randint(0, 2, (4, 4), dtype=torch.bool)
        cpu_out = torch.fmax(cpu_a, cpu_b)
        npu_out = torch.fmax(cpu_a.npu(), cpu_b.npu())
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_fp64_fallback(self):
        # 本期 fp64 走 CPU fallback（RFC §7.1），结果仍需与 CPU 逐元素一致、留在 NPU 上。
        # fallback 告警由 TORCH_NPU_WARN_ONCE 发出，走 C++ 日志通道（不进 Python
        # warnings，无法用 assertWarns 断言），日志实测见 RFC 分析文档 §10.5。
        cpu_a = _nan((4, 5), torch.float64)
        cpu_b = _nan((4, 5), torch.float64)
        cpu_out = torch.fmax(cpu_a, cpu_b)
        npu_out = torch.fmax(cpu_a.npu(), cpu_b.npu())
        self.assertEqual(npu_out.device.type, "npu")
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_complex_error(self):
        for dtype in [torch.complex64, torch.complex128]:
            cpu_a = torch.randn(4, dtype=dtype)
            cpu_b = torch.randn(4, dtype=dtype)
            # 上游 TORCH_META_FUNC(fmax) 用 TORCH_CHECK_TYPE 抛 c10::TypeError，但 torch 侧
            # 把 c10::TypeError 翻译成 Python TypeError 是较新版本才有的；老版本（2.10~2.12）
            # 会落到通用分支变成 RuntimeError。文案一致，这里两类都接受。
            with self.assertRaisesRegex(
                (TypeError, RuntimeError), "fmax not implemented for complex tensors"):
                torch.fmax(cpu_a.npu(), cpu_b.npu())
            with self.assertRaisesRegex(
                (TypeError, RuntimeError), "fmax not implemented for complex tensors"):
                out = torch.empty(4, dtype=dtype).npu()
                torch.fmax(cpu_a.npu(), cpu_b.npu(), out=out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_type_promotion(self):
        # 对齐上游 test_maximum_minimum_type_promotion：双输入 dtype 笛卡尔积。
        # fp64（fallback）与 complex（报错）由独立用例覆盖，不进本矩阵。
        dtypes = FLOAT_DTYPES + INT_DTYPES + [torch.bool]
        for dtype_a in dtypes:
            for dtype_b in dtypes:
                cpu_a = _pair_tensor(dtype_a, (4, 4))
                cpu_b = _pair_tensor(dtype_b, (4, 4))
                cpu_out = torch.fmax(cpu_a, cpu_b)
                npu_out = torch.fmax(cpu_a.npu(), cpu_b.npu())
                self.assertEqual(npu_out.dtype, torch.result_type(cpu_a, cpu_b))
                self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_mixed_dtype_broadcast_and_layout(self):
        # 交叉用例：混合 dtype × 广播（2 组）/ 非连续（2 组）
        cpu_a = torch.randn(4, 1, dtype=torch.float32)
        cpu_b = torch.randn(1, 5, dtype=torch.float16)
        self._assert_same(torch.fmax(cpu_a.npu(), cpu_b.npu()), torch.fmax(cpu_a, cpu_b))

        cpu_a = torch.randn(8, 1, 3, dtype=torch.bfloat16)
        cpu_b = torch.randint(-5, 5, (1, 4, 3), dtype=torch.int64)
        self._assert_same(torch.fmax(cpu_a.npu(), cpu_b.npu()), torch.fmax(cpu_a, cpu_b))

        base = torch.randn(6, 6, dtype=torch.float32)
        cpu_a = base.t()
        cpu_b = torch.randn(6, 6, dtype=torch.float16)
        self._assert_same(torch.fmax(cpu_a.npu(), cpu_b.npu()), torch.fmax(cpu_a, cpu_b))

        base = torch.randn(12, 6, dtype=torch.float16)
        cpu_a = base[::2]
        cpu_b = torch.randn(6, 6, dtype=torch.float32)
        self._assert_same(torch.fmax(cpu_a.npu(), cpu_b.npu()), torch.fmax(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_channels_last_unsupported(self):
        # 布局：通道格式转换后的输入
        # 已知限制：NPU 不支持 ChannelsLast（已与社区确认），fmax 只注册 .out 变体后
        # 由上游 structured_delegate 转发，create_out 会以 memory_format=ChannelsLast
        # 调 at::empty，被 torch_npu 拒绝。此处断言显式报错，确保不会静默产生
        # 与 CPU 不一致的布局。
        for dtype in (torch.float32, torch.float16):
            cpu_a = torch.randn(2, 3, 4, 5, dtype=dtype).to(memory_format=torch.channels_last)
            cpu_b = torch.randn(2, 3, 4, 5, dtype=dtype).to(memory_format=torch.channels_last)
            self.assertTrue(cpu_a.is_contiguous(memory_format=torch.channels_last))
            with self.assertRaisesRegex(
                    RuntimeError,
                    "Only c10::MemoryFormat::Contiguous is supported"):
                torch.fmax(cpu_a.npu(), cpu_b.npu())

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_extremal(self):
        # 边界：浮点极值与 ±inf、整型极值
        for dtype in FLOAT_DTYPES:
            info = torch.finfo(dtype)
            vals = [info.max, info.min, float("inf"), -float("inf"), 0.0]
            cpu_a = torch.tensor(vals, dtype=dtype)
            cpu_b = torch.tensor(vals[::-1], dtype=dtype)
            self._assert_same(torch.fmax(cpu_a.npu(), cpu_b.npu()), torch.fmax(cpu_a, cpu_b))
        for dtype in INT_DTYPES:
            info = torch.iinfo(dtype)
            cpu_a = torch.tensor([info.max, info.min, 0, -1], dtype=dtype)
            cpu_b = torch.tensor([info.min, info.max, 0, 1], dtype=dtype)
            self._assert_same(torch.fmax(cpu_a.npu(), cpu_b.npu()), torch.fmax(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_broadcast(self):
        cases = [((4, 1), (1, 4)), ((), (4, 4)), ((4, 4), ()), ((0, 3), (0, 3)), ((1, 0, 4), (3, 1, 4))]
        for shape_a, shape_b in cases:
            cpu_a = torch.randn(shape_a)
            cpu_b = torch.randn(shape_b)
            cpu_out = torch.fmax(cpu_a, cpu_b)
            npu_out = torch.fmax(cpu_a.npu(), cpu_b.npu())
            self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_non_contiguous(self):
        base_a = torch.randn(6, 6)
        base_b = torch.randn(6, 6)
        cpu_a = base_a.t()
        cpu_b = base_b.t()
        self.assertFalse(cpu_a.is_contiguous() or cpu_b.is_contiguous())
        self._assert_same(torch.fmax(cpu_a.npu(), cpu_b.npu()), torch.fmax(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_cpu_zero_dim(self):
        npu_a = _nan((4, 4)).npu()
        cpu_b = torch.tensor(float("nan"))
        self._assert_same(torch.fmax(npu_a, cpu_b), torch.fmax(npu_a.cpu(), cpu_b))
        self._assert_same(torch.fmax(cpu_b, npu_a), torch.fmax(cpu_b, npu_a.cpu()))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_normal(self):
        cpu_a = _nan((4, 4))
        cpu_b = _nan((4, 4))
        cpu_out = torch.empty(4, 4)
        npu_out = torch.empty(4, 4).npu()
        torch.fmax(cpu_a, cpu_b, out=cpu_out)
        ret = torch.fmax(cpu_a.npu(), cpu_b.npu(), out=npu_out)
        self.assertEqual(ret.data_ptr(), npu_out.data_ptr())
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_non_contiguous(self):
        cpu_a = torch.randn(6, 6)
        cpu_b = torch.randn(6, 6)
        cpu_out = torch.empty(12, 6)[::2]
        npu_out = torch.empty(12, 6).npu()[::2]
        torch.fmax(cpu_a, cpu_b, out=cpu_out)
        torch.fmax(cpu_a.npu(), cpu_b.npu(), out=npu_out)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_dtype_mismatch(self):
        # 上游语义：out dtype 与提升结果不同时不报错，按 cast 写入
        cpu_a = _nan((4, 4), torch.float16)
        cpu_b = _nan((4, 4), torch.float16)
        cpu_out = torch.empty(4, 4, dtype=torch.float32)
        npu_out = torch.empty(4, 4, dtype=torch.float32).npu()
        torch.fmax(cpu_a, cpu_b, out=cpu_out)
        torch.fmax(cpu_a.npu(), cpu_b.npu(), out=npu_out)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_complex(self):
        # out 为 complex 时上游按 cast 写入实部，虚部为 0
        cpu_a = torch.tensor([1.0, -2.0])
        cpu_b = torch.tensor([2.0, -3.0])
        cpu_out = torch.empty(2, dtype=torch.complex64)
        npu_out = torch.empty(2, dtype=torch.complex64).npu()
        torch.fmax(cpu_a, cpu_b, out=cpu_out)
        torch.fmax(cpu_a.npu(), cpu_b.npu(), out=npu_out)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_shape_mismatch(self):
        # 上游语义：shape 不匹配按 resize 处理并给弃用告警
        cpu_a = torch.randn(4, 4)
        cpu_b = torch.randn(4, 4)
        cpu_out = torch.empty(2, 2)
        npu_out = torch.empty(2, 2).npu()
        torch.fmax(cpu_a, cpu_b, out=cpu_out)
        # torch_npu 的 assertWarns 是函数式签名 assertWarns(call_fn)
        self.assertWarns(lambda: torch.fmax(cpu_a.npu(), cpu_b.npu(), out=npu_out))
        self.assertEqual(npu_out.shape, cpu_out.shape)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_shape_mismatch_float64_fallback(self):
        # fp64 走 CPU fallback 路径，out 语义必须与原生路径/上游一致：
        # resize + 弃用告警，而不是 "expanded size ... must match" 的报错
        cpu_a = torch.randn(4, 4, dtype=torch.float64)
        cpu_b = torch.randn(4, 4, dtype=torch.float64)
        cpu_out = torch.empty(2, 2, dtype=torch.float64)
        npu_out = torch.empty(2, 2, dtype=torch.float64).npu()
        torch.fmax(cpu_a, cpu_b, out=cpu_out)
        self.assertWarns(
            lambda: torch.fmax(cpu_a.npu(), cpu_b.npu(), out=npu_out))
        self.assertEqual(npu_out.shape, cpu_out.shape)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_zero_element_resize(self):
        # 零元素 out：上游按 resize 处理且不告警（PyTorch 弃用告警推荐用
        # t.resize_(0) 复用 out），fallback 路径曾因 copy_ 对空张量早退而静默返回空结果
        for dtype in (torch.float32, torch.float64):
            cpu_a = torch.randn(4, 4, dtype=dtype)
            cpu_b = torch.randn(4, 4, dtype=dtype)
            cpu_out = torch.empty(0, dtype=dtype)
            npu_out = torch.empty(0, dtype=dtype).npu()
            torch.fmax(cpu_a, cpu_b, out=cpu_out)
            torch.fmax(cpu_a.npu(), cpu_b.npu(), out=npu_out)
            self.assertEqual(npu_out.shape, cpu_out.shape)
            self.assertEqual(npu_out.numel(), 16)
            self._assert_same(npu_out, cpu_out)
        # 上游 TensorIterator 语义：可安全 cast（int→int、int→float）按 cast 写入；
        # 不可安全 cast（int→bool、float→int）报与 CPU 一致的 RuntimeError
        cpu_a = torch.tensor([2, 1, 3], dtype=torch.int32)
        cpu_b = torch.tensor([1, 2, 3], dtype=torch.int32)

        out64 = torch.empty(3, dtype=torch.int64).npu()
        torch.fmax(cpu_a.npu(), cpu_b.npu(), out=out64)
        self.assertEqual(out64.dtype, torch.int64)
        self._assert_same(out64, torch.fmax(cpu_a, cpu_b).to(torch.int64))

        out_bool = torch.empty(3, dtype=torch.bool).npu()
        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
            torch.fmax(cpu_a.npu(), cpu_b.npu(), out=out_bool)

        cpu_f = torch.randn(3)
        out_int = torch.empty(3, dtype=torch.int32).npu()
        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
            torch.fmax(cpu_f.npu(), cpu_f.npu(), out=out_int)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_same_tensor(self):
        # out 与 self 为同一 tensor 时允许，返回即 out
        cpu_a = torch.tensor([1.0, 5.0, float("nan")])
        cpu_b = torch.tensor([2.0, 3.0, 1.0])
        cpu_expected = torch.fmax(cpu_a.clone(), cpu_b)
        npu_a = cpu_a.npu()
        npu_b = cpu_b.npu()
        ret = torch.fmax(npu_a, npu_b, out=npu_a)
        self.assertEqual(ret.data_ptr(), npu_a.data_ptr())
        self._assert_same(npu_a, cpu_expected)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_out_partial_overlap(self):
        # 部分重叠：复刻上游 RuntimeError，不静默 clone。
        # out 形状必须与结果一致，否则先触发 resize 而非重叠校验。
        base = torch.tensor([1.0, 5.0, 2.0, -1.0, 9.0])
        cpu_b = torch.tensor([2.0, 3.0, 1.0, 0.0])
        cpu_a = base[:4]
        npu_base = base.npu()
        npu_a = npu_base[:4]
        npu_b = cpu_b.npu()
        with self.assertRaisesRegex(RuntimeError, "refer to a single memory location"):
            torch.fmax(cpu_a, cpu_b, out=base[1:])
        with self.assertRaisesRegex(RuntimeError, "refer to a single memory location"):
            torch.fmax(npu_a, npu_b, out=npu_base[1:])

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_zero_sign(self):
        # ±0 不做 bitwise 断言（IEEE 754 未规定，CPU x86/aarch64 本身即不一致）
        cpu_a = torch.tensor([0.0, -0.0, 0.0, -0.0])
        cpu_b = torch.tensor([-0.0, 0.0, 1.0, -1.0])
        self._assert_same(torch.fmax(cpu_a.npu(), cpu_b.npu()), torch.fmax(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_backward(self):
        nan = float("nan")
        cpu_self = torch.tensor([nan, 1.0, 5.0, 2.0], requires_grad=True)
        cpu_other = torch.tensor([2.0, nan, 1.0, 2.0], requires_grad=True)
        cpu_out = torch.fmax(cpu_self, cpu_other)
        cpu_out.sum().backward()

        npu_self = cpu_self.detach().clone().npu().requires_grad_(True)
        npu_other = cpu_other.detach().clone().npu().requires_grad_(True)
        npu_out = torch.fmax(npu_self, npu_other)
        npu_out.sum().backward()

        self._assert_same(npu_out, cpu_out.detach())
        self._assert_same(npu_self.grad, cpu_self.grad)
        self._assert_same(npu_other.grad, cpu_other.grad)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_backward_tie(self):
        # 平局（含 ±0）两侧各得 grad/2
        cpu_self = torch.ones(4, requires_grad=True)
        cpu_other = torch.ones(4, requires_grad=True)
        torch.fmax(cpu_self, cpu_other).sum().backward()

        npu_self = torch.ones(4).npu().requires_grad_(True)
        npu_other = torch.ones(4).npu().requires_grad_(True)
        torch.fmax(npu_self, npu_other).sum().backward()

        self._assert_same(npu_self.grad, cpu_self.grad)
        self._assert_same(npu_other.grad, cpu_other.grad)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmax_launch_upper_bound(self):
        # 上界告警口径：观测值写入报告，超出上界即失败（RFC §2.2 / §4.1）
        # 5 类路径 × 每类 2 种 dtype 组合
        cases = [
            ("float_same", [(torch.float32, torch.float32), (torch.float16, torch.float16)]),
            ("float_mixed", [(torch.float32, torch.float16), (torch.bfloat16, torch.float32)]),
            ("float16_bf16", [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)]),
            ("int_same", [(torch.int32, torch.int32), (torch.int64, torch.int64)]),
            ("int_mixed", [(torch.int32, torch.int64), (torch.int16, torch.int64)]),
        ]
        for name, pairs in cases:
            for dtype_a, dtype_b in pairs:
                cpu_a = torch.randn(64, 64).to(dtype_a) if dtype_a.is_floating_point \
                    else torch.randint(0, 5, (64, 64)).to(dtype_a)
                cpu_b = torch.randn(64, 64).to(dtype_b) if dtype_b.is_floating_point \
                    else torch.randint(0, 5, (64, 64)).to(dtype_b)
                npu_a, npu_b = cpu_a.npu(), cpu_b.npu()
                with torch.autograd.profiler.profile(use_device='npu') as prof:
                    torch.fmax(npu_a, npu_b)
                launch = _count_device_kernels(prof)
                if launch == 0:
                    self.skipTest(f"profiler 未捕获 aclnn 事件，计数不可信，跳过 {name} 上界断言")
                self.assertLessEqual(
                    launch, LAUNCH_UPPER_BOUND[name],
                    f"{name} launch {launch} exceeds upper bound {LAUNCH_UPPER_BOUND[name]}")


if __name__ == "__main__":
    run_tests()
