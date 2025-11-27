import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from scipy.stats import kstest
from typing import Optional


class TestNPUSimExponential(TestCase):
    """测试 npu_sim_exponential_ 算子（Ascend910B 专属）"""

    def cal_reject_num(self, alpha: float, n: int) -> float:
        """计算假设检验的拒绝阈值（复用参考用例的计算逻辑）"""
        z = -3.0902
        rate = float((1 - alpha) + z * pow((1 - alpha) * np.divide(alpha, n, where=n != 0), 0.5))
        reject_num = (1 - rate) * n
        return reject_num

    def supported_op_exec(self, tensor_cpu: torch.Tensor, lambd: float) -> np.ndarray:
        """CPU 原生 exponential_ 作为基准（lambda 对应 rate 参数）"""
        tensor_cpu = tensor_cpu.exponential_(lambd)
        return tensor_cpu.numpy()

    def custom_op_exec(self, tensor_cpu: torch.Tensor, lam: float, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        """NPU 自定义 npu_sim_exponential_ 执行逻辑（修正调用方式）"""
        tensor_npu = tensor_cpu.npu()
        # 核心修改：调用 torch_npu.npu_sim_exponential_ 静态方法
        torch_npu.npu_sim_exponential_(tensor_npu, lambd=lam, generator=gen)
        return tensor_npu

    @unittest.skip("skip") # CI版本不支持
    @SupportedDevices(['Ascend910B'])
    def test_npu_sim_exponential_distribution_consistency(self, device: str = "npu"):
        """测试：NPU 算子与 CPU 原生算子的分布一致性（KS检验）"""
        N = 100  # 测试轮数
        alpha = 0.01  # 显著性水平
        count = 0  # 拒绝原假设的次数
        lambd_list = [0.5, 1.0, 2.0, 5.0]  # 覆盖不同 λ 值

        for i in range(N):
            # 随机生成维度和形状
            k = np.random.randint(1, 5)
            shape = tuple(np.random.randint(10, 100, size=(k,)))
            # 随机选择 λ 值
            lambd = np.random.choice(lambd_list)
            
            # 生成输入 tensor
            tensor_cpu = torch.empty(size=shape, dtype=torch.float32)
            
            # CPU 基准结果
            np_cpu = self.supported_op_exec(tensor_cpu.clone(), lambd)
            # NPU 算子结果
            tensor_npu = self.custom_op_exec(tensor_cpu.clone(), lambd)
            np_npu = tensor_npu.cpu().numpy()

            # KS 检验：验证两个分布是否一致
            test_output = kstest(np_cpu.flatten(), np_npu.flatten())
            if test_output.pvalue < alpha:
                count += 1

        # 校验拒绝次数是否在可接受范围内
        reject_num = self.cal_reject_num(alpha, N)
        self.assertTrue(count <= reject_num, 
                        f"Reject count {count} exceeds threshold {reject_num}, distribution mismatch")

    @unittest.skip("skip") # CI版本不支持
    @SupportedDevices(['Ascend910B'])
    def test_npu_sim_exponential_inf_lambda(self):
        """测试：λ 为无穷大时，tensor 被置为全 0"""
        # 覆盖不同形状
        test_shapes = [(10,), (2, 5), (3, 4, 6)]
        inf_lambd = float("inf")

        for shape in test_shapes:
            with self.subTest(shape=shape):
                tensor_npu = torch.empty(shape, device="npu", dtype=torch.float32)
                # 核心修改：调用 torch_npu.npu_sim_exponential_
                torch_npu.npu_sim_exponential_(tensor_npu, inf_lambd)
                # 验证所有元素为 0
                self.assertTrue(torch.all(tensor_npu == 0.0))

    @unittest.skip("skip") # CI版本不支持
    @SupportedDevices(['Ascend910B'])
    def test_npu_sim_exponential_no_zero(self):
        """测试：正常 λ 下，输出值均大于 0（指数分布特性）"""
        test_lambds = [0.1, 1.0, 10.0]
        for lambd in test_lambds:
            with self.subTest(lambd=lambd):
                # 大张量验证统计特性
                tensor_npu = torch.empty(500000, device="npu", dtype=torch.float32)
                # 核心修改：调用 torch_npu.npu_sim_exponential_
                torch_npu.npu_sim_exponential_(tensor_npu, lambd)
                self.assertTrue(tensor_npu.min() > 0.0)

    @unittest.skip("skip") # CI版本不支持
    @SupportedDevices(['Ascend910B'])
    def test_npu_sim_exponential_negative_lambda_fails(self):
        """测试：λ 为负数时抛出 RuntimeError"""
        negative_lambds = [-0.5, -1.0, -10.0]
        for lambd in negative_lambds:
            with self.subTest(lambd=lambd):
                tensor_npu = torch.empty((1,), device="npu", dtype=torch.float32)
                with self.assertRaises(RuntimeError) as cm:
                    # 核心修改：调用 torch_npu.npu_sim_exponential_
                    torch_npu.npu_sim_exponential_(tensor_npu, lambd)
                # 验证错误信息包含关键提示
                self.assertIn("expects lambd > 0.0", str(cm.exception))

    @unittest.skip("skip") # CI版本不支持
    @SupportedDevices(['Ascend910B'])
    def test_npu_sim_exponential_zero_lambda_fails(self):
        """测试：λ 为 0 时抛出 RuntimeError"""
        tensor_npu = torch.empty((10,), device="npu", dtype=torch.float32)
        with self.assertRaises(RuntimeError) as cm:
            # 核心修改：调用 torch_npu.npu_sim_exponential_
            torch_npu.npu_sim_exponential_(tensor_npu, 0.0)
        self.assertIn("expects lambd > 0.0", str(cm.exception))

    @unittest.skip("skip") # CI版本不支持
    @SupportedDevices(['Ascend910B'])
    def test_npu_sim_exponential_generator_consistency(self):
        """测试：指定生成器种子时，结果可复现"""
        lam = 1.5
        shape = (3, 3)
        # 创建固定种子的 NPU 生成器
        gen1 = torch.Generator(device="npu")
        gen1.manual_seed(12345)
        gen2 = torch.Generator(device="npu")
        gen2.manual_seed(12345)

        # 两次执行算子
        tensor1 = torch.empty(shape, device="npu", dtype=torch.float32)
        # 核心修改：调用 torch_npu.npu_sim_exponential_（带 generator）
        torch_npu.npu_sim_exponential_(tensor1, lambd=lam, generator=gen1)

        tensor2 = torch.empty(shape, device="npu", dtype=torch.float32)
        torch_npu.npu_sim_exponential_(tensor2, lambd=lam, generator=gen2)

        # 验证结果完全一致
        self.assertEqual(tensor1, tensor2)

    @unittest.skip("skip") # CI版本不支持
    @SupportedDevices(['Ascend910B'])
    def test_npu_sim_exponential_mean(self):
        """测试：输出值的均值接近 1/λ（指数分布理论均值）"""
        test_cases = [
            (0.5, 2.0),   # λ=0.5 → 均值≈2.0
            (1.0, 1.0),   # λ=1.0 → 均值≈1.0
            (2.0, 0.5)    # λ=2.0 → 均值≈0.5
        ]
        tolerance = 0.05  # 5% 误差容忍度

        for lambd, expected_mean in test_cases:
            with self.subTest(lambd=lambd):
                tensor_npu = torch.empty(100000, device="npu", dtype=torch.float32)
                # 核心修改：调用 torch_npu.npu_sim_exponential_
                torch_npu.npu_sim_exponential_(tensor_npu, lambd)
                actual_mean = tensor_npu.mean().cpu().item()
                # 验证均值在误差范围内
                self.assertAlmostEqual(actual_mean, expected_mean, delta=expected_mean * tolerance)


if __name__ == "__main__":
    # 执行所有测试用例
    run_tests()