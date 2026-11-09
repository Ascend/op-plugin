import subprocess
import sys

import torch

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


FAILURE_SCRIPT = """
import torch
import torch_npu

value = {value}
tensor = torch.tensor(value, dtype={dtype}, device="npu")
try:
    {assert_call}
except RuntimeError:
    print("ASSERT_HOST_ERROR", flush=True)
    raise
print("ASSERT_ENQUEUED", flush=True)
torch.npu.synchronize()
print("ASSERT_COMPLETED", flush=True)
"""


class TestAssertAsync(TestCase):
    @SupportedDevices(["Ascend950"])
    def test_assert_async_success(self):
        cases = [
            (True, torch.bool),
            (1, torch.int8),
            (1, torch.int32),
            (1, torch.int64),
            (1.0, torch.float16),
            (1.0, torch.bfloat16),
            (1.0, torch.float32),
            (1.0 + 1.0j, torch.complex64),
        ]
        for value, dtype in cases:
            with self.subTest(dtype=dtype):
                input_tensor = torch.tensor(value, dtype=dtype, device="npu")
                torch._assert_async(input_tensor)
                torch._assert_async(input_tensor, "value is nonzero")
                torch.npu.synchronize()

    @SupportedDevices(["Ascend950"])
    def test_assert_async_boundary_pass(self):
        # aclnnAssertAsync judges bit-exactly: NaN/+-inf pass, and a complex value
        # passes unless both parts are zero.
        cases = [
            (float("nan"), torch.float32),
            (float("inf"), torch.float32),
            (float("-inf"), torch.float32),
            (float("nan"), torch.float16),
            (float("inf"), torch.bfloat16),
            (-1, torch.int64),
            (complex(0.0, 1.0), torch.complex64),
            (complex(1.0, 0.0), torch.complex64),
            (complex(float("nan"), 0.0), torch.complex64),
            (complex(0.0, float("inf")), torch.complex64),
        ]
        for value, dtype in cases:
            with self.subTest(value=value, dtype=dtype):
                input_tensor = torch.tensor(value, dtype=dtype, device="npu")
                torch._assert_async(input_tensor)
                torch.npu.synchronize()

    @SupportedDevices(["Ascend950"])
    def test_assert_async_single_element_view(self):
        input_tensor = torch.arange(5, dtype=torch.int32, device="npu")[2:3]
        self.assertEqual(input_tensor.numel(), 1)
        self.assertEqual(input_tensor.storage_offset(), 2)
        torch._assert_async(input_tensor)

        # A single-element tensor is always contiguous in PyTorch, so check the
        # stride instead: the kernel still has to handle offset and non-unit stride.
        strided_view = torch.arange(6, dtype=torch.int32, device="npu")[::2][1:2]
        self.assertEqual(strided_view.numel(), 1)
        self.assertEqual(strided_view.storage_offset(), 2)
        self.assertEqual(strided_view.stride(), (2,))
        torch._assert_async(strided_view)
        torch.npu.synchronize()

    @SupportedDevices(["Ascend950"])
    def test_assert_async_failure_in_subprocess(self):
        # One representative per zero-detection path; a device-side failure kills
        # the process, so every case runs in its own subprocess and the count is
        # kept low to bound the runtime.
        cases = [
            ("False", "torch.bool", "torch._assert_async(tensor)"),
            ("0", "torch.int32", 'torch._assert_async(tensor, "assert_async failure")'),
            ("-0.0", "torch.float32", 'torch._assert_async(tensor, "assert_async failure")'),
            ("complex(0.0, 0.0)", "torch.complex64", 'torch._assert_async(tensor, "assert_async failure")'),
            ("complex(-0.0, 0.0)", "torch.complex64", 'torch._assert_async(tensor, "assert_async failure")'),
        ]
        for value, dtype, assert_call in cases:
            script = FAILURE_SCRIPT.format(value=value, dtype=dtype, assert_call=assert_call)
            with self.subTest(value=value, dtype=dtype):
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                self.assertNotIn("ASSERT_COMPLETED", result.stdout)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                if "ASSERT_ENQUEUED" in result.stdout:
                    # Device path: the check traps the AI Core at synchronize.
                    # assertMsg is printed by AscendC::printf on the device, so it
                    # is only visible in the device log; stderr only carries the NPU
                    # error. Every alternative below is a device-origin signature,
                    # so a host-side check cannot satisfy this assertion.
                    self.assertRegex(
                        result.stderr,
                        r"(device task abort|107022|NPU function error|"
                        r"The process exits for this inner error|EZ9999)",
                        result.stdout + result.stderr)
                else:
                    # CPU fallback: no aclnnAssertAsync on this SoC or CANN version,
                    # so the check runs synchronously on the host and raises at the
                    # call site instead of trapping the device.
                    self.assertIn("ASSERT_HOST_ERROR", result.stdout)

    @SupportedDevices(["Ascend950"])
    def test_assert_async_invalid_numel(self):
        cases = [
            (torch.empty(0, dtype=torch.int32, device="npu"), "empty"),
            (torch.tensor([1, 0], dtype=torch.int32, device="npu"), "int32"),
            (torch.tensor([True, False], dtype=torch.bool, device="npu"), "bool"),
            (torch.tensor([1.0, 0.0], dtype=torch.float32, device="npu"), "float32"),
        ]
        for input_tensor, name in cases:
            with self.subTest(case=name):
                with self.assertRaises(RuntimeError):
                    torch._assert_async(input_tensor)

    @SupportedDevices(["Ascend950"])
    def test_assert_async_msg_length(self):
        input_tensor = torch.tensor(1, dtype=torch.int32, device="npu")
        torch._assert_async(input_tensor, "a" * 254)
        torch.npu.synchronize()
        with self.assertRaises(RuntimeError):
            torch._assert_async(input_tensor, "a" * 255)


if __name__ == "__main__":
    run_tests()
