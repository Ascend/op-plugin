import numpy as np
import torch

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


def _jit_modes():
    # A2/A3 等支持 acl_op 的老 SOC 上两种模式都验证（op_api 入口内部回落到 acl_op）；
    # Ascend950(A5) 等 aclnn-only 设备上 jit_compile=True 会被 dispatch 拒绝，只验证 op_api 路径。
    device_name = torch_npu.npu.get_device_name(0)
    if device_name.startswith(("Ascend910", "Ascend310")):
        return (False, True)
    return (False,)


class TestNpuStrideCopy(TestCase):
    def cpu_ref(self, input1, size, stride, storage_offset):
        # 主参考值：CPU 上等价变换（as_strided 视图 -> 连续化）
        return torch.as_strided(input1, size, stride, storage_offset).contiguous().numpy()

    def npu_exec(self, input1, size, stride, storage_offset):
        return torch_npu.npu_stride_copy(input1, size, stride, storage_offset).cpu().numpy()

    def _check(self, item):
        # item: [[dtype, format, shape], size, stride, storage_offset]
        # format 固定 0(ND)：非 base format 会静默走 acl_op，测不到 op_api 路径
        cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
        expected = self.cpu_ref(cpu_input, item[1], item[2], item[3])
        for jit in _jit_modes():  # False -> op_api 路径；True -> acl_op 路径
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            got = self.npu_exec(npu_input, item[1], item[2], item[3])
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, got)

    def _check_out(self, item, jit_modes=None):
        # .out 版：预分配 out -> npu_stride_copy.out 原地填充 -> 与参考值比对
        # jit_modes: False 走 op_api，True 走 acl_op；默认按设备能力选择
        jit_modes = _jit_modes() if jit_modes is None else jit_modes
        cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
        expected = self.cpu_ref(cpu_input, item[1], item[2], item[3])
        for jit in jit_modes:
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            out = torch.empty(item[1], dtype=npu_input.dtype, device=npu_input.device)
            torch.ops.npu.npu_stride_copy.out(npu_input, item[1], item[2], item[3], out=out)
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, out.cpu().numpy())

    def test_stride_copy_main_path(self):
        self._check([[np.float32, 0, [3, 3]], (2, 2), (1, 2), 0])
        self._check([[np.int32, 0, [5, 5]], (3, 3), (1, 2), 1])

    def test_stride_copy_small_bytes(self):
        # 小张量（nbytes<32）：acl_op 会兜底到 ViewCopy，验证 op_api 无需兜底也正确
        self._check([[np.int8, 0, [5, 5]], (5, 5), (1, 5), 0])

    def test_stride_copy_small_bytes_bool(self):
        # 1 字节 bool：acl_op 特意避开 ViewCopy 以防 bool 字节被规整
        cpu_input = torch.arange(25).reshape(5, 5) % 3 == 0
        npu_input = cpu_input.npu()
        size, stride, offset = (5, 5), (1, 5), 0
        expected = self.cpu_ref(cpu_input, size, stride, offset).astype(np.uint8)
        for jit in _jit_modes():
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            got = self.npu_exec(npu_input, size, stride, offset).astype(np.uint8)
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, got)

    def test_stride_copy_quant_int_repr(self):
        # 量化 view/ravel 的真实调用场景（TensorFactories.cpp / TensorShape.cpp）
        self._check([[np.int8, 0, [4, 4]], (4, 4), (1, 4), 0])

    def test_stride_copy_big_stride(self):
        # 末维 stride >= 256：acl_op 走 “AsStrided + Transpose” 性能分支，老 SOC 上验证两条路径结果一致
        self._check([[np.float16, 0, [1, 300]], (2, 2), (1, 256), 0])

    def test_stride_copy_offset(self):
        self._check([[np.float16, 0, [13, 23]], (10, 15), (1, 2), 1])

    def test_stride_copy_broadcast(self):
        # stride 含 0：广播式取数
        self._check([[np.float32, 0, [4]], (3, 4), (0, 1), 0])

    def test_stride_copy_big_numel(self):
        self._check([[np.float32, 0, [1000]], (500, 2), (2, 1), 0])

    def test_stride_copy_ndim(self):
        self._check([[np.float32, 0, [10]], (4,), (2,), 1])
        self._check([[np.float32, 0, [32, 8, 2]], (8, 6, 2), (5, 4, 1), 1])
        self._check([[np.float32, 0, [2, 3, 4, 5, 6]], (2, 3, 4, 5, 3), (360, 120, 30, 6, 2), 0])

    def test_stride_copy_numpy_ref(self):
        # 独立参考值：numpy 手算，避免与算子用同一变换自证
        cpu_input = torch.randn(1, 300)
        npu_input = cpu_input.npu()
        size, stride, offset = (2, 2), (1, 256), 0
        flat = cpu_input.reshape(-1).numpy()
        expected = np.zeros(size, dtype=flat.dtype)
        for idx in np.ndindex(*size):
            expected[idx] = flat[offset + sum(i * s for i, s in zip(idx, stride))]
        for jit in _jit_modes():
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            got = self.npu_exec(npu_input, size, stride, offset)
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, got)

    def test_stride_copy_out_variant(self):
        # .out 版是框架热路径实际调用的入口（copy_d2d_dtype_baseformat）
        self._check_out([[np.float32, 0, [3, 3]], (2, 2), (1, 2), 0])
        self._check_out([[np.int8, 0, [5, 5]], (5, 5), (1, 5), 0])

    @SupportedDevices(['Ascend950'])
    def test_stride_copy_out_big_stride(self):
        # A5 走 op_api 路径，out 与返回值均正确；
        # A2/A3 保持原有 acl_op 实现，其大 stride 分支重绑定 result 而不写入传入的 out，故仅在 A5 断言
        self._check_out([[np.float16, 0, [1, 300]], (2, 2), (1, 256), 0], jit_modes=(False,))

    def test_stride_copy_via_copy_of_noncontiguous(self):
        # 端到端：非连续源(offset!=0).contiguous() -> npu_stride_copy_out
        # 可能被 contiguous 优化器接管，仅作正确性检查
        cpu_input = torch.randn(6, 8)
        npu_input = cpu_input.npu()
        cpu_src = cpu_input.t()[1:]
        npu_src = npu_input.t()[1:]
        torch_npu.npu.set_compile_mode(jit_compile=False)
        self.assertRtolEqual(cpu_src.contiguous().numpy(), npu_src.contiguous().cpu().numpy())


if __name__ == "__main__":
    run_tests()
