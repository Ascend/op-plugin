import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuFusedLinearOnlineMaxSum(TestCase):
    def test_npu_fused_linear_online_max_sum_fp16(self):
        batch = 128
        hidden = 64
        vocab_size = 256
        vocab_start = 0
        vocab_end = 64

        input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
        weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
        target_tensor = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()

        logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, vocab_parallel_logits = \
            torch_npu.npu_fused_linear_online_max_sum(
                input_tensor, weight_tensor, target_tensor,
                vocab_start, vocab_end, return_logits=True
            )

        self.assertEqual(logits_max.shape, torch.Size([batch]))
        self.assertEqual(sum_exp_logits.shape, torch.Size([batch]))
        self.assertEqual(predicted_logits.shape, torch.Size([batch]))
        self.assertEqual(target_mask.shape, torch.Size([(batch + 7) // 8]))
        self.assertEqual(masked_target.shape, torch.Size([batch]))
        self.assertEqual(vocab_parallel_logits.shape, torch.Size([batch, vocab_size]))
        self.assertEqual(logits_max.dtype, torch.float32)
        self.assertEqual(sum_exp_logits.dtype, torch.float32)
        self.assertEqual(predicted_logits.dtype, torch.float32)
        self.assertEqual(target_mask.dtype, torch.uint8)
        self.assertEqual(masked_target.dtype, torch.int32)
        self.assertEqual(vocab_parallel_logits.dtype, torch.float16)

    def test_npu_fused_linear_online_max_sum_bf16(self):
        batch = 64
        hidden = 32
        vocab_size = 128
        vocab_start = 0
        vocab_end = 32

        input_tensor = torch.randn(batch, hidden, dtype=torch.bfloat16).npu()
        weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.bfloat16).npu()
        target_tensor = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()

        logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, vocab_parallel_logits = \
            torch_npu.npu_fused_linear_online_max_sum(
                input_tensor, weight_tensor, target_tensor,
                vocab_start, vocab_end, return_logits=True
            )

        self.assertEqual(logits_max.shape, torch.Size([batch]))
        self.assertEqual(sum_exp_logits.shape, torch.Size([batch]))
        self.assertEqual(vocab_parallel_logits.dtype, torch.bfloat16)

    def test_npu_fused_linear_online_max_sum_no_logits(self):
        batch = 32
        hidden = 16
        vocab_size = 64
        vocab_start = 0
        vocab_end = 16

        input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
        weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
        target_tensor = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()

        logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, vocab_parallel_logits = \
            torch_npu.npu_fused_linear_online_max_sum(
                input_tensor, weight_tensor, target_tensor,
                vocab_start, vocab_end, return_logits=False
            )

        self.assertEqual(logits_max.shape, torch.Size([batch]))
        self.assertEqual(sum_exp_logits.shape, torch.Size([batch]))
        self.assertEqual(predicted_logits.shape, torch.Size([batch]))
        self.assertIsNone(vocab_parallel_logits)

    def test_npu_fused_linear_online_max_sum_int64_target(self):
        batch = 16
        hidden = 8
        vocab_size = 32
        vocab_start = 0
        vocab_end = 8

        input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
        weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
        target_tensor = torch.randint(0, vocab_size, (batch,), dtype=torch.int64).npu()

        logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, vocab_parallel_logits = \
            torch_npu.npu_fused_linear_online_max_sum(
                input_tensor, weight_tensor, target_tensor,
                vocab_start, vocab_end, return_logits=True
            )

        self.assertEqual(masked_target.dtype, torch.int64)
        self.assertEqual(logits_max.shape, torch.Size([batch]))


if __name__ == "__main__":
    run_tests()
