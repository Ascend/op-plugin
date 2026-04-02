import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests

class TestComparisonFormatPreservation(TestCase):
    def setUp(self):
        # 保存当前配置
        self.original_allow_internal_format = getattr(torch_npu.npu.config, 'allow_internal_format', None)
        # 启用内部格式
        torch_npu.npu.config.allow_internal_format = True

    def tearDown(self):
        # 恢复原始配置
        if self.original_allow_internal_format is not None:
            torch_npu.npu.config.allow_internal_format = self.original_allow_internal_format

    def create_test_tensor(self, shape, dtype, input_format):
        """创建具有指定格式的测试张量"""
        torch.npu.set_device('npu:0')
        tensor = torch.randn(shape,dtype=dtype,device='npu:0')
        # 转换为指定格式
        if input_format != torch_npu.Format.ND:
            tensor = torch_npu.npu_format_cast(tensor, input_format)
        return tensor

    def test_comparison_operators_format_preservation(self):
        """测试所有比较操作符是否保留输入张量格式"""
        # 测试不同的张量形状和数据类型
        test_cases = [
            # (shape, dtype, input_format)
            ((2, 3, 4, 5), torch.float32, torch_npu.Format.NCHW),      # Non-private format
            ((2, 3, 4, 5), torch.float32, torch_npu.Format.ND),         # Non-private format
            ((2, 3, 4, 5), torch.float32, torch_npu.Format.FRACTAL_Z),  # Private format
            ((2, 3, 4, 5), torch.float32, torch_npu.Format.FRACTAL_NZ), # Private format
            ((2, 3, 4, 5), torch.float32, torch_npu.Format.NC1HWC0),    # Private format

        ]

        # 所有比较操作符
        comparison_ops = [
            (torch.gt, "gt"),
            (torch.ge, "ge"),
            (torch.lt, "lt"),
            (torch.le, "le"),
            (torch.eq, "eq"),
            (torch.ne, "ne"),
        ]

        for shape, dtype, input_format in test_cases:
            with self.subTest(shape=shape, dtype=dtype, input_format=input_format):
                # 创建两个输入张量
                tensor1 = self.create_test_tensor(shape, dtype, input_format)
                tensor2 = self.create_test_tensor(shape, dtype, input_format)
                
                # 保存原始格式
                original_format1 = torch_npu.get_npu_format(tensor1)
                original_format2 = torch_npu.get_npu_format(tensor2)
                                
                for op_func, op_name in comparison_ops:
                    with self.subTest(op_name=op_name):
                        # 测试 tensor-tensor 比较
                        result = op_func(tensor1, tensor2)
                        
                        # 验证输入张量格式未改变
                        current_format1 = torch_npu.get_npu_format(tensor1)
                        current_format2 = torch_npu.get_npu_format(tensor2)
                        
                        self.assertEqual(current_format1, original_format1, 
                                        f"{op_name}: Tensor1 format changed from {original_format1} to {current_format1}")
                        self.assertEqual(current_format2, original_format2, 
                                        f"{op_name}: Tensor2 format changed from {original_format2} to {current_format2}")

if __name__ == "__main__":
    run_tests()