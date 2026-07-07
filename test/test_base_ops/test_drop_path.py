import torch
import torch_npu
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_npu.contrib.module import NpuDropPath


class TestNpuDropPath(TestCase):

    def test_basic_forward(self):
        """基础前向传播"""
        drop_path = NpuDropPath(0).npu()
        x = torch.randn(68, 5, device='npu')
        output = drop_path(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.device, x.device)

    def test_drop_prob_0(self):
        """drop_prob=0 时输出等于输入"""
        drop_path = NpuDropPath(0).npu()
        x = torch.randn(68, 5, device='npu')
        output = drop_path(x)
        self.assertTrue(torch.allclose(output, x))

    def test_backward(self):
        """反向传播：autograd 示例"""
        drop_path = NpuDropPath(0).npu()
        input1 = torch.randn(68, 5, device='npu', requires_grad=True)
        input2 = torch.randn(68, 5, device='npu', requires_grad=True)
        output = input1 + drop_path(input2)
        output.sum().backward()
        self.assertIsNotNone(input1.grad)
        self.assertIsNotNone(input2.grad)

    def test_train_eval_mode(self):
        """训练模式 vs 评估模式"""
        drop_path = NpuDropPath(0.5).npu()
        x = torch.randn(68, 5, device='npu')
        # 训练模式：有随机丢弃
        drop_path.train()
        train_output = drop_path(x)
        has_zero_train = (train_output == 0).any()
        # 评估模式：不丢弃
        drop_path.eval()
        eval_output = drop_path(x)
        self.assertTrue(torch.allclose(eval_output, x))
        print(f"训练模式是否有丢弃: {has_zero_train.item()}")


if __name__ == '__main__':
    run_tests()
