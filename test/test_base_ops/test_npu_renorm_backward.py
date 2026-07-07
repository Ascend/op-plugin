import torch
import torch_npu
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests


class TestNPURenormBackward(TestCase):

    def test_basic_forward(self):
        """基础功能：直接调用 npu_renorm_backward"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        result = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, 1.0)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.device, x.device)
        self.assertEqual(result.dtype, x.dtype)

    def test_compare_with_autograd_p_2(self):
        """对比自动微分：p=2"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 2.0, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_p_1(self):
        """对比自动微分：p=1"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 1.0, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 1.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_p_3(self):
        """对比自动微分：p>2"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 3.0, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 3.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_p_0_5(self):
        """对比自动微分：p<1"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 0.5, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 0.5, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_p_1_5(self):
        """对比自动微分：1<p<2"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 1.5, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 1.5, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_3d(self):
        """3D 张量测试"""
        x = torch.randn(3, 4, 5, device='npu', requires_grad=True)
        grad = torch.randn(3, 4, 5, device='npu')
        y = torch.renorm(x, 2.0, 1, 0.8)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 1, 0.8)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_4d(self):
        """4D 张量测试"""
        x = torch.randn(2, 3, 4, 5, device='npu', requires_grad=True)
        grad = torch.randn(2, 3, 4, 5, device='npu')
        y = torch.renorm(x, 2.0, 2, 0.8)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 2, 0.8)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_negative_dim(self):
        """负数 dim 测试"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 2.0, -1, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, -1, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_different_maxnorm(self):
        """不同 maxnorm 值测试"""
        for maxnorm in [0.5, 1.0, 2.0, 5.0]:
            x = torch.randn(4, 8, device='npu', requires_grad=True)
            grad = torch.randn(4, 8, device='npu')
            y = torch.renorm(x, 2.0, 0, maxnorm)
            y.backward(grad)
            grad_auto = x.grad.clone()
            x.grad.zero_()
            grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, maxnorm)
            self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_float16(self):
        """float16 精度测试"""
        x = torch.randn(4, 8, device='npu', dtype=torch.float16, requires_grad=True)
        grad = torch.randn(4, 8, device='npu', dtype=torch.float16)
        y = torch.renorm(x.float(), 2.0, 0, 1.0).to(torch.float16)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-3, rtol=1e-3))

    def test_compare_with_autograd_bfloat16(self):
        """bfloat16 精度测试"""
        x = torch.randn(4, 8, device='npu', dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(4, 8, device='npu', dtype=torch.bfloat16)
        y = torch.renorm(x.float(), 2.0, 0, 1.0).to(torch.bfloat16)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-2, rtol=1e-2))

    def test_no_scale_when_norm_less_than_maxnorm(self):
        """norm < maxnorm 时，梯度应等于输入梯度（不缩放）"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        maxnorm = 10.0
        y = torch.renorm(x, 2.0, 0, maxnorm)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, maxnorm)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_zero_tensor(self):
        """全零输入测试"""
        x = torch.zeros(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 2.0, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_large_tensor(self):
        """大张量测试"""
        x = torch.randn(100, 100, device='npu', requires_grad=True)
        grad = torch.randn(100, 100, device='npu')
        y = torch.renorm(x, 2.0, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    run_tests()
