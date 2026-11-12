# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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

"""DVM no-graph fusion regressions, isolated from ordinary operator tests."""

import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch_npu


def _check_lazy_fusion_training_backward():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    cases = [
        ((8, 3), 0.4, 2.0),
        ((2, 4, 6), 0.4, 2.0),
        ((4, 3, 5, 5), 0.4, 2.0),
        ((4, 3, 5, 5), 2.0, 0.0),
    ]
    checked = 0
    for shape, scale, shift in cases:
        for eps in (1e-5, 1e-2):
            channels = shape[1]
            x = (torch.randn(shape) * scale + shift).requires_grad_()
            weight = (torch.randn(channels) + 1).requires_grad_()
            bias = torch.randn(channels, requires_grad=True)
            grad = torch.randn_like(x)
            expected = torch.nn.functional.batch_norm(
                x, torch.zeros(channels), torch.ones(channels),
                weight, bias, True, 0.1, eps,
            )
            expected.backward(grad)
            x_npu = x.detach().npu()
            weight_npu = weight.detach().npu()
            actual, mean, saved_stat = torch.ops.aten.native_batch_norm.default(
                x_npu, weight_npu, bias.detach().npu(),
                torch.zeros(channels).npu(), torch.ones(channels).npu(),
                True, 0.1, eps,
            )
            torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=1e-4, atol=1e-4)
            # Consume the statistic produced by the real forward. Its meaning
            # differs between BNTrainingUpdate (variance) and BatchNormV3 (invstd).
            for mask in itertools.product((False, True), repeat=3):
                if not any(mask):
                    continue
                actual_grads = torch.ops.aten.native_batch_norm_backward.default(
                    grad.npu(), x_npu, weight_npu, None, None,
                    mean, saved_stat, True, eps, list(mask),
                )
                for enabled, actual_grad, expected_grad in zip(
                    mask, actual_grads, (x.grad, weight.grad, bias.grad)
                ):
                    if enabled:
                        torch.testing.assert_close(
                            actual_grad.cpu(), expected_grad, rtol=1e-4, atol=1e-4,
                            msg=lambda msg: f"shape={shape}, eps={eps}, mask={mask}: {msg}",
                        )
                    else:
                        assert actual_grad is None
                checked += 1
    print(f"BN_FORWARD_BACKWARD_PASS cases={checked}")


def _check_lazy_fusion_inference_statistics():
    torch.manual_seed(7)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    cases = [
        ((4, 32), True, True),
        ((2, 32, 4, 8), False, True),
        ((2, 32, 4, 8), True, False),
    ]
    checked = 0
    for shape, with_weight, with_bias in cases:
        for eps in (1e-5, 1e-2):
            channels = shape[1]
            x = torch.randn(shape)
            weight = torch.randn(channels) if with_weight else None
            bias = torch.randn(channels) if with_bias else None
            running_mean = torch.randn(channels)
            running_var = torch.rand(channels) + 0.5
            expected = torch.nn.functional.batch_norm(
                x, running_mean, running_var, weight, bias, False, 0.1, eps)
            actual, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
                x.npu(),
                None if weight is None else weight.npu(),
                None if bias is None else bias.npu(),
                running_mean.npu(), running_var.npu(), False, 0.1, eps)
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
            # Inference must return the statistics the fused graph actually
            # uses (running mean and rsqrt(running_var + eps)) in the input
            # dtype, not the placeholder zeros it used to return.
            assert save_mean.shape == (channels,) and save_mean.dtype == torch.float32
            assert save_invstd.shape == (channels,) and save_invstd.dtype == torch.float32
            torch.testing.assert_close(save_mean.cpu(), running_mean, rtol=0, atol=0)
            torch.testing.assert_close(
                save_invstd.cpu(), torch.rsqrt(running_var + eps), rtol=1e-5, atol=1e-6)
            checked += 1
    print(f"BN_INFERENCE_STATISTICS_PASS cases={checked}")


def _check_gelu_modes(output):
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    torch.npu.set_device(0)
    fusion = os.environ["TORCH_NPU_LAZY_FUSION"].startswith("True")
    compatible = os.environ["TORCH_NPU_USE_COMPATIBLE_IMPL"] == "1"
    records = {}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for transposed in (False, True):
            x = (torch.randn(17, 31) * 2).to(dtype)
            grad = torch.randn_like(x)
            if transposed:
                x, grad = x.t(), grad.t()
            for mode in ("none", "tanh"):
                xn = x.npu().detach().requires_grad_(True)
                yn = torch.nn.functional.gelu(xn, approximate=mode)
                yn.backward(grad.npu())
                y, dx = yn.detach().cpu(), xn.grad.cpu()
                # The fused tanh formula is checked against CPU FP64. The
                # mode-aware native configuration also provides a CPU control.
                # Default native API accuracy is outside this DVM regression.
                if (fusion and mode == "tanh") or compatible:
                    expected_y = torch.nn.functional.gelu(x.double(), approximate=mode)
                    expected_dx = torch.ops.aten.gelu_backward.default(
                        grad.double(), x.double(), approximate=mode)
                    tolerance = {torch.float32: 1e-5 if mode == "tanh" else 1e-6,
                                 torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
                    torch.testing.assert_close(y.double(), expected_y, rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(dx.double(), expected_dx, rtol=tolerance, atol=tolerance)
                records[f"{dtype}-{transposed}-{mode}"] = {
                    "input": x, "grad": grad, "output": y, "grad_input": dx}
    torch.save(records, output)
    print(f"DVM_GELU_PASS cases={len(records)}")


class TestDVMLazyFusion(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        if not torch.npu.get_device_name(0).startswith(("Ascend910B", "Ascend910_93", "Ascend950")):
            self.skipTest("No-graph fusion is not supported on this device")

    def _run_worker(self, worker, env, output=None):
        command = [sys.executable, os.path.abspath(__file__), "--worker", worker]
        if output is not None:
            command.append(str(output))
        result = subprocess.run(command, cwd="/tmp", env=env, capture_output=True,
                                text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        markers = {
            "bn": "BN_FORWARD_BACKWARD_PASS cases=56",
            "bn_inference": "BN_INFERENCE_STATISTICS_PASS cases=6",
            "gelu": "DVM_GELU_PASS cases=12",
        }
        self.assertIn(markers[worker], result.stdout)

    def test_bn_training_saved_statistics(self):
        env = os.environ.copy()
        env["TORCH_NPU_LAZY_FUSION"] = (
            "True enable_ops_only=native_batch_norm_backward,batch_norm_backward_elemt")
        self._run_worker("bn", env)

    def test_bn_inference_saved_statistics(self):
        env = os.environ.copy()
        env["TORCH_NPU_LAZY_FUSION"] = "True enable_ops_only=native_batch_norm"
        self._run_worker("bn_inference", env)

    def test_gelu_approximate_modes(self):
        # Fusion and compatible-mode switches are cached: use fresh processes.
        with tempfile.TemporaryDirectory(prefix="dvm-gelu-") as directory:
            for compatible in ("0", "1"):
                with self.subTest(compatible=compatible):
                    for fusion in ("False", "True"):
                        env = os.environ.copy()
                        env["TORCH_NPU_LAZY_FUSION"] = (
                            "True enable_ops_only=gelu,gelu_backward" if fusion == "True" else "False")
                        env["TORCH_NPU_USE_COMPATIBLE_IMPL"] = compatible
                        output = Path(directory) / f"{compatible}-{fusion}.pt"
                        self._run_worker("gelu", env, output)
                    eager = torch.load(Path(directory) / f"{compatible}-False.pt", weights_only=True)
                    fused = torch.load(Path(directory) / f"{compatible}-True.pt", weights_only=True)
                    self.assertEqual(set(eager), set(fused))
                    for key in eager:
                        for name in ("input", "grad"):
                            torch.testing.assert_close(fused[key][name], eager[key][name], rtol=0, atol=0)
                        if key.endswith("-none"):
                            # Non-tanh dispatch must preserve op_api behavior,
                            # including the default native path on older SoCs.
                            for name in ("output", "grad_input"):
                                torch.testing.assert_close(fused[key][name], eager[key][name],
                                                           rtol=0, atol=0)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker", "bn"]:
        _check_lazy_fusion_training_backward()
    elif sys.argv[1:] == ["--worker", "bn_inference"]:
        _check_lazy_fusion_inference_statistics()
    elif len(sys.argv) == 4 and sys.argv[1:3] == ["--worker", "gelu"]:
        _check_gelu_modes(sys.argv[3])
    else:
        unittest.main()
