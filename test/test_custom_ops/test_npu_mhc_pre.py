import torch
import torch_npu
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuMhcPre(TestCase):
    @staticmethod
    def fp32_to_hf32(value):
        value_bits = value.contiguous().view(torch.int32)
        hf32_bits = torch.bitwise_right_shift(torch.bitwise_right_shift(value_bits, 12) + 1, 1)
        hf32_bits = torch.bitwise_left_shift(hf32_bits, 13)
        return hf32_bits.view(torch.float32)

    def cpu_op_exec(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        gamma: torch.Tensor = None,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
        inner_precise: int = 0,
    ):
        T, N, D = x.shape
        ND = N * D

        x = x.reshape(T, ND).float()
        inv_rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)

        if gamma is not None:
            matmul_x = x * gamma.reshape(ND).float()
        else:
            matmul_x = x
        matmul_phi = phi.float()
        if inner_precise == 1:
            matmul_x = self.fp32_to_hf32(matmul_x)
            matmul_phi = self.fp32_to_hf32(matmul_phi)

        h_mix = F.linear(matmul_x, matmul_phi)
        weight = h_mix * inv_rms
        h_pre, h_post, h_res = weight.split([N, N, N * N], dim=-1)
        h_res = h_res.unflatten(-1, (N, N))
        h_pre = torch.sigmoid(h_pre * alpha[0] + bias[:N].unsqueeze(0)) + hc_eps
        h_post = 2 * torch.sigmoid(h_post * alpha[1] + bias[N:2 * N].unsqueeze(0))
        h_res = h_res * alpha[2] + bias[2 * N:].view(N, N).unsqueeze(0)
        h_in = torch.sum(
            h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(N, -1)),
            dim=1
        ).bfloat16()

        return (h_in, h_post, h_res, inv_rms[:, 0], h_mix, h_pre)

    def cpu_op_exec_hy(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        gamma: torch.Tensor = None,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ):
        T, N, D = x.shape
        ND = N * D

        x = x.reshape(T, ND).float()
        inv_rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)

        if gamma is not None:
            gamma = gamma.reshape(ND).float()
            h_mix = F.linear(x * gamma, phi.float())
            weight = h_mix * inv_rms
        else:
            h_mix = F.linear(x, phi.float())
            weight = h_mix * inv_rms

        h_pre, h_post = weight.split([N, N], dim=-1)
        h_pre = torch.sigmoid(h_pre * alpha[0] + bias[:N].unsqueeze(0)) + hc_eps
        h_post = 2 * torch.sigmoid(h_post * alpha[1] + bias[N:2 * N].unsqueeze(0))
        h_in = torch.sum(
            h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(N, -1)),
            dim=1
        ).bfloat16()

        return (h_in, h_post, inv_rms[:, 0], h_mix, h_pre)

    def custom_op_exec(self, x, phi, alpha, bias, gamma, out_flag, inner_precise=0):
        return torch_npu.npu_mhc_pre(
            x,
            phi,
            alpha,
            bias,
            gamma=gamma,
            out_flag=out_flag,
            inner_precise=inner_precise
        )

    def build_input_tensors(self, T, n, D):
        x = torch.randn(T, n, D, dtype=torch.bfloat16)
        phi = torch.randn(n * n + 2 * n, n * D, dtype=torch.float32)
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        gamma = torch.ones(n, D, dtype=torch.float32)

        bias_pre = torch.full((n,), 0.01, dtype=torch.float32)
        bias_post = torch.full((n,), 0.01, dtype=torch.float32)
        bias_res = torch.full((n, n), 0.01, dtype=torch.float32)
        bias = torch.cat([bias_pre, bias_post, bias_res.reshape(-1)], dim=0)

        return x, phi, alpha, bias, gamma

    def build_hy_input_tensors(self, T, n, D):
        x = torch.randn(T, n, D, dtype=torch.bfloat16)
        phi = torch.randn(2 * n, n * D, dtype=torch.float32)
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float32)
        gamma = torch.ones(n, D, dtype=torch.float32)

        bias_pre = torch.full((n,), 0.01, dtype=torch.float32)
        bias_post = torch.full((n,), 0.01, dtype=torch.float32)
        bias = torch.cat([bias_pre, bias_post.reshape(-1)], dim=0)

        return x, phi, alpha, bias, gamma

    def run_and_check(self, T, n, D, out_flag, output_names, tol_map, inner_precise=0):
        with torch.no_grad():
            x, phi, alpha, bias, gamma = self.build_input_tensors(T, n, D)

            expected_output = self.cpu_op_exec(
                x, phi, alpha, bias, gamma, inner_precise=inner_precise
            )[:len(output_names)]
            actual_output = self.custom_op_exec(
                x.npu(), phi.npu(), alpha.npu(), bias.npu(), gamma.npu(), out_flag=out_flag,
                inner_precise=inner_precise
            )[:len(output_names)]

            for name, exp, act in zip(output_names, expected_output, actual_output):
                try:
                    self.assertRtolEqual(
                        exp.float().numpy(),
                        act.float().cpu().numpy(),
                        prec=tol_map[name]
                    )
                except AssertionError as e:
                    raise AssertionError(
                        f"Output {name} compare failed for shape (T={T}, n={n}, D={D}), "
                        f"out_flag={out_flag}, inner_precise={inner_precise}: {e}"
                    )

    def run_hy_and_check(self, T, n, D, out_flag, output_names, tol_map):
        with torch.no_grad():
            x, phi, alpha, bias, gamma = self.build_hy_input_tensors(T, n, D)

            expected_output = self.cpu_op_exec_hy(x, phi, alpha, bias, gamma)
            actual_all = self.custom_op_exec(
                x.npu(), phi.npu(), alpha.npu(), bias.npu(), gamma.npu(), out_flag=out_flag
            )
            # alpha=[2] 时 h_res(index 2) 为空，跳过
            actual_output = [actual_all[i] for i in [0, 1, 3, 4, 5]]

            for name, exp, act in zip(output_names, expected_output, actual_output):
                try:
                    self.assertRtolEqual(
                        exp.float().numpy(),
                        act.float().cpu().numpy(),
                        prec=tol_map[name]
                    )
                except AssertionError as e:
                    raise AssertionError(
                        f"Output {name} compare failed for shape (T={T}, n={n}, D={D}), "
                        f"out_flag={out_flag}: {e}"
                    )

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_prefill_training(self, device="npu"):
        # 训练场景的 prefill 模式: T >= 512, out_flag=1, 有效输出 h_in, h_post, h_res, inv_rms, h_mix, h_pre
        T, n, D = (4096, 4, 5120)
        out_flag = 1
        output_names = ["h_in", "h_post", "h_res", "inv_rms", "h_mix", "h_pre"]
        tol_map = {
            "h_in": 2 ** -7,
            "h_post": 1e-3,
            "h_res": 1e-3,
            "inv_rms": 1e-3,
            "h_mix": 1e-3,
            "h_pre": 1e-3,
        }
        self.run_and_check(T, n, D, out_flag, output_names, tol_map)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_prefill_inference(self, device="npu"):
        # 推理场景的 prefill 模式: T >= 512, out_flag=0, 有效输出 h_in, h_post, h_res
        T, n, D = (1024, 4, 2560)
        out_flag = 0
        output_names = ["h_in", "h_post", "h_res"]
        tol_map = {
            "h_in": 2 ** -7,
            "h_post": 1e-3,
            "h_res": 1e-3,
        }
        self.run_and_check(T, n, D, out_flag, output_names, tol_map)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_decode_inference(self, device="npu"):
        # 推理场景的 decode 模式: T < 512, out_flag=0，有效输出 h_in, h_post, h_res
        T, n, D = (64, 4, 2560)
        out_flag = 0
        output_names = ["h_in", "h_post", "h_res"]
        tol_map = {
            "h_in": 2 ** -7,
            "h_post": 1e-3,
            "h_res": 1e-3,
        }
        for inner_precise in (0, 1):
            self.run_and_check(T, n, D, out_flag, output_names, tol_map, inner_precise=inner_precise)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_prefill_hy(self, device="npu"):
        # hy场景的 prefill 模式: T >= 512, out_flag=1, alpha=[2] 无 h_res, 有效输出 h_in, h_post, inv_rms, h_mix, h_pre
        T, n, D = (1024, 4, 2560)
        out_flag = 1
        output_names = ["h_in", "h_post", "inv_rms", "h_mix", "h_pre"]
        tol_map = {
            "h_in": 2 ** -7,
            "h_post": 1e-3,
            "inv_rms": 1e-3,
            "h_mix": 1e-3,
            "h_pre": 1e-3,
        }
        self.run_hy_and_check(T, n, D, out_flag, output_names, tol_map)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_decode_hy(self, device="npu"):
        # hy场景的 decode 模式: T < 512, out_flag=1, alpha=[2] 无 h_res, 有效输出 h_in, h_post, inv_rms, h_mix, h_pre
        T, n, D = (64, 4, 2560)
        out_flag = 1
        output_names = ["h_in", "h_post", "inv_rms", "h_mix", "h_pre"]
        tol_map = {
            "h_in": 2 ** -7,
            "h_post": 1e-3,
            "inv_rms": 1e-3,
            "h_mix": 1e-3,
            "h_pre": 1e-3,
        }
        self.run_hy_and_check(T, n, D, out_flag, output_names, tol_map)

if __name__ == "__main__":
    run_tests()
