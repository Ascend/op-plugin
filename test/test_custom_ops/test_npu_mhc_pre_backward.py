import torch
import torch_npu
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuMhcPre(TestCase):

    def mhc_forward_pre(self, x, phi, alpha, bias, gamma, outflag: bool=False, norm_eps: float=1e-6, hc_eps: float=1e-6):

        # ---- shape ----
        is_tnd = (x.dim() == 3)
        if is_tnd:
            # TND: [T, n, D] -> treat as [1, T, n, D] for unified computation
            x = x.unsqueeze(0)
        B, S, n, D = x.shape
        nD = n * D

        # ---- flatten ----
        # [B, S, nD]
        vecX = x.reshape(B, S, nD).float()


        # ----gamma加权后再做GEMM ----
        # [B, S, n^2 + 2n]
        if gamma is not None:
            h_mix = torch.matmul(vecX * gamma.reshape(1, 1, nD), phi.t())
        else:
            h_mix = torch.matmul(vecX, phi.t())

        # ---- RMSNorm (delayed) ----
        # [B, S, 1]
        inv_rms = torch.rsqrt(vecX.square().mean(-1, keepdim=True) + norm_eps)
        # [B, S, n^2 + 2n]
        h_mix_tmp = h_mix * inv_rms
        # [B, S, n], [B, S, n], [B, S, n*n]
        h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
        # [B, S, n, n]
        h_res2 = h_res1.reshape(B, S, n, n)
        a_pre, a_post, a_res = alpha

        # ---- split ----
        h_pre2  = a_pre * h_pre1    + bias[:n]                 # [B, S, n]
        h_post2 = a_post * h_post1   + bias[n:2*n]              # [B, S, n]
        h_res  = a_res   * h_res2    + bias[2*n:].view(n, n)    # [B, S, n, n]

        # ---- nonlinear ----
        h_pre  = torch.sigmoid(h_pre2) + hc_eps                 # [B, S, n]
        h_post = 2.0 * torch.sigmoid(h_post2)                   # [B, S, n]

        # ---- h_in = h_pre @ x ----
        # [B, S, D] = [B, S, n, 1] * [B, S, n, D]
        h_in_fp = (h_pre.unsqueeze(-1) * x.float()).sum(dim=2)
        h_in = h_in_fp.to(torch.bfloat16)

        if is_tnd:
            # Squeeze back to TND format
            h_in = h_in.squeeze(0)        # [T, D]
            h_post = h_post.squeeze(0)    # [T, n]
            h_res = h_res.squeeze(0)      # [T, n, n]

        if not outflag:
            return h_in, h_post, h_res
        else:
            inv_rms = inv_rms.squeeze(-1) # [B, S] or [1, T]
            if is_tnd:
                inv_rms = inv_rms.squeeze(0)  # [T]
                h_mix = h_mix.squeeze(0)      # [T, n^2 + 2n]
                h_pre = h_pre.squeeze(0)      # [T, n]
            return h_in, h_post, h_res, inv_rms, h_mix, h_pre

    # 手动反向传播
    def cpu_op_exec(self,
        x, phi, alpha, bias,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma,
        norm_eps: float=1e-6, hc_eps: float=1e-6, grad_x_post=None
    ):
        is_tnd = (x.dim() == 3)
        if is_tnd:
            # TND -> BSND: [T, n, D] -> [1, T, n, D]
            x = x.unsqueeze(0)
            inv_rms = inv_rms.unsqueeze(0)
            h_mix = h_mix.unsqueeze(0)
            h_pre = h_pre.unsqueeze(0)
            h_post = h_post.unsqueeze(0)
            dh_in = dh_in.unsqueeze(0)
            dh_post = dh_post.unsqueeze(0)
            dh_res = dh_res.unsqueeze(0)
            if grad_x_post is not None:
                grad_x_post = grad_x_post.unsqueeze(0)

        B, S, n, D = x.shape
        nD = n * D
        vecX = x.reshape(B, S, nD).float()
        a_pre, a_post, a_res = alpha

        dx = torch.zeros_like(x)
        dphi = torch.zeros_like(phi)
        dalpha = torch.zeros_like(alpha)
        dbias = torch.zeros_like(bias)

        # ========================================================
        # ---- 计算导数 ----
        # (V0) 反推sigmod/线性变换梯度
        # ========================================================
        x_fp = x.float() # [B, S, n, D]
        h_in_fp_grad = dh_in.float() # [B, S, D]
        dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1) # [B, S, n]

        s_pre2 = h_pre - hc_eps                                     # [B, S, n]
        dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)                  # [B, S, n]
        dh_post2 = dh_post * h_post * (1.0 - h_post / 2)            # [B, S, n]

        dh_pre1 = a_pre * dh_pre2                                   # [B, S, n]
        dh_post1 = a_post * dh_post2                                # [B, S, n]
        dh_res1 = a_res * dh_res                                    # [B, S, n, n]

        dh_mix_tmp = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1)   # [B, S, n^2 + 2n]
        dh_mix = dh_mix_tmp * inv_rms[:, :, None]                   # [B, S, n^2 + 2n]

        # ========================================================
        # (C0)
        # ========================================================
        dvecX_mm = torch.matmul(dh_mix, phi)                     # [B, S, nD]

        # ========================================================
        # (C1)
        # ========================================================
        if gamma is not None:
            xrs = x_fp.reshape(B*S, n*D) * gamma.reshape(1, n*D)            # [BS, nD]
        else:
            xrs = x_fp.reshape(B*S, n*D)                                    # [BS, nD]
        dphi = torch.matmul(dh_mix.reshape(B*S, n*n + 2*n).t(), xrs)    # [n^2 + 2n, nD]

        # ========================================================
        # (V1)
        # ========================================================
        h_mix_tmp = h_mix * inv_rms[:, :, None]                         # [B, S, n^2 + 2n]
        h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
        dinv_rms = (dh_mix_tmp * h_mix).sum(dim=-1, keepdim=True)       # [B, S, 1]

        dalpha_pre = (dh_pre2 * h_pre1).sum()
        dalpha_post = (dh_post2 * h_post1).sum()
        dalpha_res = (dh_res.reshape(B, S, n * n) * h_res1).sum()
        dalpha = torch.stack([dalpha_pre, dalpha_post, dalpha_res])

        dbias_pre = dh_pre2.sum(dim=(0, 1))                         # [n]
        dbias_post = dh_post2.sum(dim=(0, 1))                       # [n]
        dbias_res = dh_res.reshape(B, S, n * n).sum(dim=(0, 1))     # [n*n]
        dbias = torch.cat([dbias_pre, dbias_post, dbias_res], dim=0)

        # ========================================================
        # (V2)
        # ========================================================
        dvecX_inv = -(dinv_rms * inv_rms[:, :, None].pow(3) / nD) * vecX    # [B, S, nD]
        dvecX_hin = h_pre.unsqueeze(-1) * dh_in.unsqueeze(2)                # [B, S, n, D]
        dvecX_inv = dvecX_inv.reshape(B, S, n, D) + dvecX_hin               # [B, S, n, D]

        if gamma is not None:
            dgamma = (x_fp.reshape(B * S, n * D) * dvecX_mm.reshape(B * S, n * D)).sum(dim=-2)
            dx = dvecX_mm.reshape(B, S, n, D) * gamma.reshape(1, 1, n, D) + dvecX_inv
        else:
            dgamma = torch.zeros(n, D, dtype=torch.float32)
            dx = dvecX_mm.reshape(B, S, n, D) + dvecX_inv
        if grad_x_post != None:
            dx = dx + grad_x_post.reshape(B, S, n, D).float()

        if is_tnd:
            # Squeeze back to TND: [1, T, n, D] -> [T, n, D]
            dx = dx.squeeze(0)

        return dx.to(torch.bfloat16), dphi, dalpha, dbias, dgamma   

    def npu_op_exec(self, x, phi, alpha,
            dh_in, dh_post, dh_res,
            inv_rms, h_mix, h_pre, h_post, 
            gamma, hc_eps, grad_x_post=None):
        return torch_npu.npu_mhc_pre_backward(
            x, phi, alpha, dh_in, dh_post, dh_res,
            inv_rms, h_mix, h_pre, h_post, 
            gamma=gamma, hc_eps=hc_eps, grad_x_post=grad_x_post
        )

    def build_input_tensors(self, B, S, n, D, with_grad_x_post=False, tnd_format=False, with_gamma=True):
        if tnd_format:
            # TND format: [T, n, D]
            T = B * S
            x = torch.randn(T, n, D, dtype=torch.bfloat16)
        else:
            # BSND format: [B, S, n, D]
            x = torch.randn(B, S, n, D, dtype=torch.bfloat16)

        phi = torch.randn(n * n + 2 * n, n * D, dtype=torch.float32)
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1
        gamma = torch.randn(n, D, dtype=torch.float32) if with_gamma else None

        if tnd_format:
            # TND format gradients
            dh_in = torch.randn(T, D).bfloat16()
            dh_post = torch.randn(T, n)
            dh_res = torch.randn(T, n, n)
        else:
            # BSND format gradients
            dh_in = torch.randn(B, S, D).bfloat16()
            dh_post = torch.randn(B, S, n)
            dh_res = torch.randn(B, S, n, n)


        # 正向传播
        h_in, h_post, h_res, inv_rms, h_mix, h_pre = self.mhc_forward_pre(
            x, phi, alpha, bias, gamma, outflag=True
        )

        grad_x_post = None
        if with_grad_x_post:
            if tnd_format:
                grad_x_post = torch.randn(T, n, D, dtype=torch.bfloat16)
            else:
                grad_x_post = torch.randn(B, S, n, D, dtype=torch.bfloat16)

        return (x, phi, alpha, bias, gamma,
                dh_in, dh_post, dh_res, inv_rms, h_mix, h_pre, h_post, grad_x_post)

    def run_and_check(self, B, S, n, D, with_grad_x_post=False, tnd_format=False, with_gamma=True):
        with torch.no_grad():
            (x, phi, alpha, bias, gamma,
                dh_in, dh_post, dh_res, inv_rms, h_mix, h_pre, h_post,
                grad_x_post) = self.build_input_tensors(B, S, n, D, with_grad_x_post,
                                                         tnd_format=tnd_format, with_gamma=with_gamma)

            expected_output = self.cpu_op_exec(x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
                dh_in, dh_post, dh_res, gamma, hc_eps=1e-6, grad_x_post=grad_x_post)

            npu_args = [t.npu() for t in [ x, phi, alpha, dh_in, dh_post, dh_res,
                                        inv_rms, h_mix, h_pre, h_post]]
            actual_output = self.npu_op_exec(
                *npu_args,
                gamma=gamma.npu() if gamma is not None else None,
                hc_eps=1e-6,
                grad_x_post=grad_x_post.npu() if grad_x_post is not None else None,
            )

            output_names = ["dx", "dphi", "dalpha", "dbias", "dgamma"]
            tol_map = {
                "dx": 2 ** -7,
                "dphi": 1e-3,
                "dalpha": 1e-3,
                "dbias": 1e-3,
                "dgamma": 1e-3,
            }

            for name, exp, act in zip(output_names, expected_output, actual_output):
                act_cpu = act.float().cpu()
                if name == "dgamma":
                    act_cpu = act_cpu.reshape(exp.shape)
                self.assertRtolEqual(exp.float().numpy(), act_cpu.numpy(), prec=tol_map[name])

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_without_grad_x_post(self, device="npu"):
        self.run_and_check(B=1, S=128, n=4, D=512)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_with_grad_x_post(self, device="npu"):
        self.run_and_check(B=1, S=128, n=4, D=512, with_grad_x_post=True)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_tnd_without_grad_x_post(self, device="npu"):
        self.run_and_check(B=1, S=128, n=4, D=512, tnd_format=True)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_tnd_with_grad_x_post(self, device="npu"):
        self.run_and_check(B=1, S=128, n=4, D=512, with_grad_x_post=True, tnd_format=True)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_no_gamma(self, device="npu"):
        self.run_and_check(B=1, S=128, n=4, D=512, with_gamma=False)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_no_gamma_with_grad_x_post(self, device="npu"):
        self.run_and_check(B=1, S=128, n=4, D=512, with_grad_x_post=True, with_gamma=False)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_tnd_no_gamma(self, device="npu"):
        self.run_and_check(B=1, S=128, n=4, D=512, tnd_format=True, with_gamma=False)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_tnd_no_gamma_with_grad_x_post(self, device="npu"):
        self.run_and_check(B=1, S=128, n=4, D=512, with_grad_x_post=True, tnd_format=True, with_gamma=False)


if __name__ == "__main__":
    run_tests()
