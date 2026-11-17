import numpy as np
import torch

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


def _jit_modes():
    # On socs that still support acl_op both modes run (the op_api entry falls back to
    # acl_op); on aclnn-only devices jit_compile=True is rejected, so only op_api runs.
    device_name = torch_npu.npu.get_device_name(0)
    if device_name.startswith(("Ascend910", "Ascend310")):
        return (False, True)
    return (False,)


class TestNpuStrideCopy(TestCase):
    def cpu_ref(self, input1, size, stride, storage_offset):
        # reference: the equivalent transform on CPU (as_strided view -> contiguous)
        return torch.as_strided(input1, size, stride, storage_offset).contiguous().numpy()

    def npu_exec(self, input1, size, stride, storage_offset):
        return torch_npu.npu_stride_copy(input1, size, stride, storage_offset).cpu().numpy()

    def _check(self, item):
        # item: [[dtype, format, shape], size, stride, storage_offset]
        # format is fixed to 0 (ND) to keep non-base formats out of the way
        cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
        expected = self.cpu_ref(cpu_input, item[1], item[2], item[3])
        for jit in _jit_modes():  # False -> op_api path; True -> acl_op path
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            got = self.npu_exec(npu_input, item[1], item[2], item[3])
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, got)

    def _check_out(self, item, jit_modes=None):
        # .out variant: preallocate out, fill it in place, then compare with the reference
        # jit_modes: False -> op_api, True -> acl_op; defaults to the device capability
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
        # small tensor (nbytes < 32): acl_op falls back to ViewCopy, op_api must match it
        self._check([[np.int8, 0, [5, 5]], (5, 5), (1, 5), 0])

    def test_stride_copy_small_bytes_bool(self):
        # 1-byte bool: acl_op deliberately avoids ViewCopy, which can normalize bool bytes
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
        # real call sites of quantized view/ravel (TensorFactories.cpp / TensorShape.cpp)
        self._check([[np.int8, 0, [4, 4]], (4, 4), (1, 4), 0])

    def test_stride_copy_big_stride(self):
        # last-dim stride >= 256: covers the AsStrided + Transpose fast path in acl_op
        self._check([[np.float16, 0, [1, 300]], (2, 2), (1, 256), 0])

    def test_stride_copy_offset(self):
        self._check([[np.float16, 0, [13, 23]], (10, 15), (1, 2), 1])

    def test_stride_copy_broadcast(self):
        # stride contains 0: broadcast-style gather
        self._check([[np.float32, 0, [4]], (3, 4), (0, 1), 0])

    def test_stride_copy_big_numel(self):
        self._check([[np.float32, 0, [1000]], (500, 2), (2, 1), 0])

    def test_stride_copy_ndim(self):
        self._check([[np.float32, 0, [10]], (4,), (2,), 1])
        self._check([[np.float32, 0, [32, 8, 2]], (8, 6, 2), (5, 4, 1), 1])
        self._check([[np.float32, 0, [2, 3, 4, 5, 6]], (2, 3, 4, 5, 3), (360, 120, 30, 6, 2), 0])

    def test_stride_copy_numpy_ref(self):
        # independent reference computed with numpy, so the op is not validated against itself
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

    def test_stride_copy_view_source(self):
        # source is a non-contiguous view with a storage offset: storage_offset is
        # storage-relative, not view-relative
        base = torch.randn(4, 8)
        npu_base = base.npu()
        src = npu_base.t()[1:]           # shape (7,4), strides (1,8), storage_offset=1
        size, stride, offset = (2, 3), (1, 8), 1
        expected = torch.as_strided(base, size, stride, offset).numpy()
        for jit in _jit_modes():
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            got = torch_npu.npu_stride_copy(src, size, stride, offset).cpu().numpy()
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, got)

    def test_stride_copy_out_view_source(self):
        # .out with a view source and a non-zero storage offset, the call shape used by
        # copy_d2d_dtype_baseformat
        base = torch.randn(4, 8)
        npu_base = base.npu()
        src = npu_base.t()[1:]           # shape (7,4), strides (1,8), storage_offset=1
        size, stride, offset = (2, 3), (1, 8), 1
        expected = torch.as_strided(base, size, stride, offset).numpy()
        for jit in _jit_modes():
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            out = torch.empty(size, dtype=npu_base.dtype, device=npu_base.device)
            torch.ops.npu.npu_stride_copy.out(src, size, stride, offset, out=out)
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, out.cpu().numpy())

    def test_stride_copy_out_variant(self):
        # .out is the entry the framework hot path actually calls (copy_d2d_dtype_baseformat)
        self._check_out([[np.float32, 0, [3, 3]], (2, 2), (1, 2), 0])
        self._check_out([[np.int8, 0, [5, 5]], (5, 5), (1, 5), 0])

    @SupportedDevices(['Ascend950'])
    def test_stride_copy_out_big_stride(self):
        # A5 goes through op_api, where both out and the return value are correct; acl_op's
        # big-stride branch rebinds result instead of writing the passed-in out, so A5 only
        self._check_out([[np.float16, 0, [1, 300]], (2, 2), (1, 256), 0], jit_modes=(False,))

    def test_stride_copy_via_copy_of_noncontiguous(self):
        # end to end: contiguous() on a non-contiguous source (offset != 0) -> npu_stride_copy_out
        # the contiguous optimizer may take over, so this only checks correctness
        cpu_input = torch.randn(6, 8)
        npu_input = cpu_input.npu()
        cpu_src = cpu_input.t()[1:]
        npu_src = npu_input.t()[1:]
        torch_npu.npu.set_compile_mode(jit_compile=False)
        self.assertRtolEqual(cpu_src.contiguous().numpy(), npu_src.contiguous().cpu().numpy())


if __name__ == "__main__":
    run_tests()
