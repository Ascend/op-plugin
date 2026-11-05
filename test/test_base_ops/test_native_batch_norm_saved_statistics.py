import torch
import torch_npu
import torch_npu.testing
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests


class TestNativeBatchNormSavedStatistics(TestCase):
    @parametrize("jit", [False, True])
    @parametrize("wrong_size", [False, True])
    @parametrize("internal_format", [False, True])
    def test_inference_saved_statistics_edges(self, device, jit, wrong_size, internal_format):
        # internal_format only guards against a configuration regression here: the saved statistics
        # are 1D, so they stay ND under both settings and do not cover non-ND storage.
        previous_jit = torch.npu.is_jit_compile_false()
        option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        previous_format = option is not None and option.decode() == "enable"
        try:
            torch.npu.set_compile_mode(jit_compile=jit)
            torch.npu.config.allow_internal_format = internal_format
            x = torch.arange(24, device=device, dtype=torch.float32).reshape(2, 3, 4) / 7
            mean = torch.tensor([-0.5, 0.75, 1.25], device=device)
            variance = torch.tensor([0.0, 1e-8, 1.0], device=device)
            original_mean, original_variance = mean.clone(), variance.clone()
            saved_mean = torch.full((3,), float("nan"), device=device)
            saved_rstd = torch.full((4 if wrong_size else 3,), float("nan"), device=device)
            mean_ptr, rstd_ptr = saved_mean.data_ptr(), saved_rstd.data_ptr()
            out = torch.empty_like(x)
            if wrong_size:
                with self.assertRaisesRegex(RuntimeError, "(?i)size|shape|broadcast|expand|dimension"):
                    torch.ops.aten.native_batch_norm.out(
                        x, None, None, mean, variance, False, 0.1, 1e-8,
                        out=out, save_mean=saved_mean, save_invstd=saved_rstd)
                torch.npu.synchronize()
                self.assertEqual(saved_rstd.shape, (4,))
                self.assertEqual(saved_rstd.data_ptr(), rstd_ptr)
                return
            result = torch.ops.aten.native_batch_norm.out(
                x, None, None, mean, variance, False, 0.1, 1e-8,
                out=out, save_mean=saved_mean, save_invstd=saved_rstd)
            reference_rstd = (variance.cpu() + 1e-8).rsqrt()
            self.assertEqual(result[0].data_ptr(), out.data_ptr())
            self.assertEqual(result[1].data_ptr(), mean_ptr)
            self.assertEqual(result[2].data_ptr(), rstd_ptr)
            self.assertEqual(saved_mean.cpu(), mean.cpu(), atol=0, rtol=0)
            self.assertEqual(saved_rstd.cpu(), reference_rstd, atol=1.5e-5, rtol=1.3e-5)
            expected = torch.nn.functional.batch_norm(x.cpu(), mean.cpu(), variance.cpu(), training=False, eps=1e-8)
            self.assertEqual(out.cpu(), expected, atol=1.5e-5, rtol=1.3e-5)
            self.assertEqual(mean, original_mean, atol=0, rtol=0)
            self.assertEqual(variance, original_variance, atol=0, rtol=0)
        finally:
            torch.npu.config.allow_internal_format = previous_format
            torch.npu.set_compile_mode(jit_compile=not previous_jit)

    def test_inference_5d_out(self, device):
        previous_jit = torch.npu.is_jit_compile_false()
        option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        previous_format = option is not None and option.decode() == "enable"
        try:
            torch.npu.set_compile_mode(jit_compile=True)
            torch.npu.config.allow_internal_format = False
            x = torch.arange(96, device=device, dtype=torch.float32).reshape(2, 3, 4, 2, 2) / 7
            weight = torch.tensor([0.75, 1.25, 0.5], device=device)
            bias = torch.tensor([-0.25, 0.5, 0.75], device=device)
            mean = torch.tensor([-0.5, 0.75, 1.25], device=device)
            variance = torch.tensor([0.5, 1.5, 2.0], device=device)
            out = torch.full_like(x, float("nan"))
            saved_mean = torch.full_like(mean, float("nan"))
            saved_rstd = torch.full_like(variance, float("nan"))
            out_ptr = out.data_ptr()
            mean_ptr, rstd_ptr = saved_mean.data_ptr(), saved_rstd.data_ptr()
            result = torch.ops.aten.native_batch_norm.out(
                x, weight, bias, mean, variance, False, 0.1, 0.01,
                out=out, save_mean=saved_mean, save_invstd=saved_rstd)
            expected = torch.nn.functional.batch_norm(
                x.cpu(), mean.cpu(), variance.cpu(), weight.cpu(), bias.cpu(), training=False, eps=0.01)
            self.assertEqual(out.data_ptr(), out_ptr)
            self.assertEqual(result[0].data_ptr(), out_ptr)
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(result[0].shape, x.shape)
            self.assertEqual(out.cpu(), expected, atol=1.5e-5, rtol=1.3e-5)
            self.assertEqual(saved_mean.data_ptr(), mean_ptr)
            self.assertEqual(saved_rstd.data_ptr(), rstd_ptr)
            self.assertEqual(result[1].data_ptr(), mean_ptr)
            self.assertEqual(result[2].data_ptr(), rstd_ptr)
            self.assertEqual(saved_mean.cpu(), mean.cpu(), atol=0, rtol=0)
            self.assertEqual(saved_rstd.cpu(), (variance.cpu() + 0.01).rsqrt(), atol=1.5e-5, rtol=1.3e-5)
        finally:
            torch.npu.config.allow_internal_format = previous_format
            torch.npu.set_compile_mode(jit_compile=not previous_jit)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("jit", [False, True])
    def test_no_stats_training(self, device, dtype, jit):
        self.assertTrue(torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_native_batch_norm_legit.no_stats", "PrivateUse1"))
        previous_jit = torch.npu.is_jit_compile_false()
        try:
            torch.npu.set_compile_mode(jit_compile=jit)
            values = torch.sin(torch.arange(96, dtype=torch.float32)).reshape(4, 3, 8) * 2
            x = values.to(device=device, dtype=dtype).requires_grad_()
            reference = x.detach().cpu().float().requires_grad_()
            result = torch.ops.aten._native_batch_norm_legit.no_stats(x, None, None, True, 0.5, 1e-5)
            expected = torch.nn.functional.batch_norm(reference, None, None, training=True)
            atol, rtol = {torch.float16: (1e-5, 1e-3), torch.bfloat16: (1e-5, 1.6e-2),
                          torch.float32: (1.5e-5, 1.3e-5)}[dtype]
            self.assertEqual(result[0].cpu(), expected.to(dtype), atol=atol, rtol=rtol)
            self.assertEqual(result[1].cpu(), reference.mean(dim=(0, 2)), atol=1.5e-5, rtol=1.3e-5)
            self.assertEqual(result[2].cpu(), reference.var(dim=(0, 2), correction=0), atol=1.5e-5, rtol=1.3e-5)
            grad = torch.cos(torch.arange(96, dtype=torch.float32)).reshape(4, 3, 8)
            result[0].backward(grad.to(device=device, dtype=dtype))
            expected.backward(grad.to(dtype).float())
            self.assertEqual(x.grad.cpu(), reference.grad.to(dtype), atol=atol, rtol=rtol)
        finally:
            torch.npu.set_compile_mode(jit_compile=not previous_jit)

    @parametrize("jit", [False, True])
    def test_no_stats_requires_training(self, device, jit):
        previous_jit = torch.npu.is_jit_compile_false()
        try:
            torch.npu.set_compile_mode(jit_compile=jit)
            x = torch.ones((4, 3, 8), device=device)
            with self.assertRaisesRegex(RuntimeError, "no_stats requires training=True"):
                torch.ops.aten._native_batch_norm_legit.no_stats(x, None, None, False, 0.3, 0.01)
        finally:
            torch.npu.set_compile_mode(jit_compile=not previous_jit)

    @parametrize("jit", [False, True])
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("use_out", [False, True])
    def test_inference_saved_statistics(self, device, jit, dtype, use_out):
        previous_jit = torch.npu.is_jit_compile_false()
        option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        previous_format = option is not None and option.decode() == "enable"
        try:
            torch.npu.set_compile_mode(jit_compile=jit)
            torch.npu.config.allow_internal_format = False
            x = torch.arange(24, device=device, dtype=dtype).reshape(3, 2, 4) / 7
            weight = torch.tensor([0.75, 1.25], device=device, dtype=dtype)
            bias = torch.tensor([-0.25, 0.5], device=device, dtype=dtype)
            mean = torch.tensor([-0.5, 0.75], device=device, dtype=dtype)
            variance = torch.tensor([0.5, 1.5], device=device, dtype=dtype)
            original_mean, original_variance = mean.clone(), variance.clone()
            args = (x, weight, bias, mean, variance, False, -1.2, 0.01)
            if use_out:
                out = torch.empty_like(x)
                saved_mean = torch.full_like(mean, float("nan"))
                saved_rstd = torch.full_like(variance, float("nan"))
                result = torch.ops.aten.native_batch_norm.out(
                    *args, out=out, save_mean=saved_mean, save_invstd=saved_rstd)
                self.assertEqual(result[0].data_ptr(), out.data_ptr())
                self.assertEqual(result[0].shape, out.shape)
                self.assertEqual(result[1].data_ptr(), saved_mean.data_ptr())
                self.assertEqual(result[2].data_ptr(), saved_rstd.data_ptr())
            else:
                result = torch.ops.aten.native_batch_norm.default(*args)
            reference_rstd = (variance.cpu().float() + 0.01).rsqrt().to(dtype)
            expected = ((x.cpu().float() - mean.cpu().float()[None, :, None])
                        * (variance.cpu().float() + 0.01).rsqrt()[None, :, None]
                        * weight.cpu().float()[None, :, None] + bias.cpu().float()[None, :, None]).to(dtype)
            atol, rtol = {torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-5, 1.6e-2),
                          torch.float32: (1.5e-5, 1.3e-5)}[dtype]
            self.assertEqual(result[0].cpu(), expected, atol=atol, rtol=rtol)
            if use_out:
                self.assertEqual(out.cpu(), expected, atol=atol, rtol=rtol)
            self.assertEqual(result[1].cpu(), mean.cpu(), atol=0, rtol=0)
            self.assertEqual(result[2].cpu(), reference_rstd, atol=atol, rtol=rtol)
            self.assertEqual(result[1].dtype, dtype)
            self.assertEqual(result[2].dtype, dtype)
            self.assertEqual(result[1].shape, mean.shape)
            self.assertEqual(result[2].shape, variance.shape)
            self.assertEqual(mean, original_mean, atol=0, rtol=0)
            self.assertEqual(variance, original_variance, atol=0, rtol=0)
            self.assertNotEqual(result[1].data_ptr(), mean.data_ptr())
        finally:
            torch.npu.config.allow_internal_format = previous_format
            torch.npu.set_compile_mode(jit_compile=not previous_jit)


instantiate_device_type_tests(TestNativeBatchNormSavedStatistics, globals(), only_for=("npu",))

if __name__ == "__main__":
    run_tests()
