import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

def split_and_merge_dim(x, dim):
    """
    将张量在指定维度 split_dim 处分界：
    - 前面的维度合并为一个
    - 后面的维度合并为一个
    - split_dim 本身保留

    Args:
        x: 输入张量
        split_dim: 分界维度（从1开始计数，比如第3维传入3）

    Returns:
        重新排列后的张量，shape: (before_total, split_dim_size, after_total)
    """
    # 前面维度合并
    before_shape = x.shape[:dim]
    before_total = int(torch.Size(before_shape).numel())

    # 后面维度合并
    after_shape = x.shape[dim:]
    after_total = int(torch.Size(after_shape).numel())

    # 重塑
    return x.reshape(before_total, after_total)

class TestClippedSwiglu(TestCase):
    def get_golden(self, input_x, group_index, dim, alpha, limit, bias, interleaved):
        x = split_and_merge_dim(input_x, dim)
        if group_index is not None:
            group_sum = min(torch.sum(group_index), x.shape[0])
        else:
            group_sum = x.shape[0]
        x_tensor = x[:group_sum]
        if interleaved:
            x_glu = x_tensor[..., ::2]
            x_linear = x_tensor[..., 1::2]
        else:
            out = torch.chunk(x_tensor, 2, dim=-1)
            x_glu = out[0]
            x_linear = out[1]
        x_glu = x_glu.clamp(min=None, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
        sigmoid_part = torch.sigmoid(alpha * x_glu)
        result = x_glu * sigmoid_part * (x_linear + bias)
        y = torch.zeros((x.shape[0], x.shape[1] // 2), dtype=input_x.dtype)
        y[:group_sum] = result
        shape = list(input_x.shape)
        shape[dim] = shape[dim] // 2
        return y.reshape(shape)

    @unittest.skip("Skip test_npu_clipped_swiglu now")
    @SupportedDevices(['Ascend910B'])
    def test_swiglu(self):
        shape = [8192, 3904 * 2]
        input_x = torch.randn(shape, dtype=torch.float32)
        dim = -1
        interleaved = True
        limit = 1.0
        alpha = 0.7
        bias = 1.2
        dim = -1
        group_num = 5
        x = split_and_merge_dim(input_x, dim)
        group_index = torch.randint(1, 100, (group_num, ), dtype=torch.int64)
        total = torch.sum(group_index)

        out = torch_npu.npu_clipped_swiglu(
            input_x.npu(),
            group_index=group_index.npu(),
            dim=dim,
            alpha=alpha,
            limit=limit,
            bias=bias,
            interleaved=interleaved
        )
        torch.npu.synchronize()

        golden = self.get_golden(input_x, group_index, dim, alpha, limit, bias, interleaved)
        self.assertRtolEqual(out[:total].cpu(), golden[:total])

if __name__ == "__main__":
    run_tests()