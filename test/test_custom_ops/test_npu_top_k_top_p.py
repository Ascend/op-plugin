import unittest
import numpy as np
import torch

import torch_npu
from torch.testing._internal.common_utils import TestCase, run_tests, parametrize, instantiate_parametrized_tests
from torch_npu.testing.common_utils import SupportedDevices

TOL_MAPPING = {
    torch.float: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=5e-3, rtol=5e-3),
}


class TestNpuTopKTopP(TestCase):
    def cpu_op_exec(self, logits, p, k):
        """
        对模型的原始输出 logits 进行 top-k 和 top-p 采样过滤。
        输入：
          - logits: 原始模型输出的未归一化概率，形状为 [batch_size, vocab_size]
          - p: top-p 阈值，形状为 [batch_size]
          - k: top-k 阈值，形状为 [batch_size]
        返回：
          - 经过 top-k 和 top-p 过滤后的 logits，形状同输入
        """

        # 对 logits 进行升序排序（从小到大），返回排序后的值和对应原索引
        # logits_sort 形状 [batch_size, vocab_size]，升序排列（最后一个元素是最大值）
        logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True) # 形状 [batch_size, vocab_size]

        # 1. 应用 top-k 过滤
        # --------------------------------------------------
        # 计算需要保留的最小索引位置（保留最大的k个值）
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # 例如 vocab_size=5, k=3 → 保留索引2到4 形状 [batch_size]
        # 获取每个样本对应的阈值（第k大的值）
        # gather 操作获取每个样本在 top_k_mask 位置的值，作为阈值分界线
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))  # 形状 [batch_size, 1]

        # 生成 mask：将小于阈值的标记为 True（这些位置后续会被置为 -inf）
        top_k_mask = logits_sort < top_k_mask  # 广播比较，形状 [batch_size, vocab_size]

        # 用 -inf 填充需要过滤的位置（softmax 后会变为0概率）
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

        # 2. 应用 top-p (nucleus) 过滤
        # --------------------------------------------------
        # 将过滤后的 logits 转换为概率分布
        probs_sort = logits_sort.to(torch.float32).softmax(dim=-1)  # 形状 [batch_size, vocab_size]

        # 计算累积概率（从最小的概率开始累加）
        probs_sum = probs_sort.cumsum(dim=-1) # 形状 [batch_size, vocab_size]

        # 生成 mask：累积概率 <= (1 - p) 的位置需要被过滤
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)  # 形状 [batch_size, vocab_size]

        # 确保至少保留一个标记（将最后一个位置的 mask 设为 False）
        top_p_mask[:, -1] = False # 形状 [batch_size, vocab_size]

        # 用 -inf 填充需要过滤的位置
        logits_sort.masked_fill_(top_p_mask, -float("inf")) # 形状 [batch_size, vocab_size]

        # 3. 恢复原始排序
        # --------------------------------------------------
        # 使用 scatter 将排序后的 logits 还原到原始顺序
        logits = torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort) # 形状 [batch_size, vocab_size]
        return logits

    def npu_op_exec(self, logits, p, k):
        return torch_npu.npu_top_k_top_p(logits, p, k).cpu()

    @unittest.skip("skip") # CI版本不支持
    @SupportedDevices(['Ascend910B'])
    @parametrize('vocab_size', [15206, 152064])
    @parametrize('batch_size', [4, 128])
    @parametrize('dtype', [torch.float32])
    def test_npu_apply_top_k_top_p(self, vocab_size, batch_size, dtype):
        tols = TOL_MAPPING.get(dtype)
        k_max = 1024
        shape = [batch_size, vocab_size]
        logits = torch.from_numpy(np.random.uniform(-5, 5, shape)).to(dtype)
        p = torch.rand(batch_size).to(dtype)
        k = torch.randint(10, k_max, (batch_size,)).to(torch.int32)
        out_cpu = self.cpu_op_exec(logits, p, k)
        out_npu = self.npu_op_exec(logits.npu(), p.npu(), k.npu())
        self.assertEqual(out_cpu, out_npu, **tols)


instantiate_parametrized_tests(TestNpuTopKTopP)

if __name__ == "__main__":
    run_tests()
