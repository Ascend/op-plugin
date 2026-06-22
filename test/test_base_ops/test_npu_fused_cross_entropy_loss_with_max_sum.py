import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuFusedCrossEntropyLossWithMaxSum(TestCase):
    def test_npu_fused_cross_entropy_loss_with_max_sum_basic(self):
        batch = 128

        logits_max = torch.randn(batch, dtype=torch.float32).npu()
        sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0
        predicted_logits = torch.randn(batch, dtype=torch.float32).npu()

        loss, softmax = torch_npu.npu_fused_cross_entropy_loss_with_max_sum(
            logits_max, sum_exp_logits, predicted_logits,
            label_smoothing=0.0, input=None, weight=None, vocab_parallel_logits=None
        )

        self.assertEqual(loss.shape, torch.Size([batch]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertIsNone(softmax)

    def test_npu_fused_cross_entropy_loss_with_max_sum_with_logits(self):
        batch = 64
        vocab_size = 256

        logits_max = torch.randn(batch, dtype=torch.float32).npu()
        sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0
        predicted_logits = torch.randn(batch, dtype=torch.float32).npu()
        vocab_parallel_logits = torch.randn(batch, vocab_size, dtype=torch.float16).npu()

        loss, softmax = torch_npu.npu_fused_cross_entropy_loss_with_max_sum(
            logits_max, sum_exp_logits, predicted_logits,
            label_smoothing=0.0, input=None, weight=None,
            vocab_parallel_logits=vocab_parallel_logits
        )

        self.assertEqual(loss.shape, torch.Size([batch]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(softmax.shape, torch.Size([batch, vocab_size]))
        self.assertEqual(softmax.dtype, torch.float32)

    def test_npu_fused_cross_entropy_loss_with_max_sum_bf16_logits(self):
        batch = 32
        vocab_size = 128

        logits_max = torch.randn(batch, dtype=torch.float32).npu()
        sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0
        predicted_logits = torch.randn(batch, dtype=torch.float32).npu()
        vocab_parallel_logits = torch.randn(batch, vocab_size, dtype=torch.bfloat16).npu()

        loss, softmax = torch_npu.npu_fused_cross_entropy_loss_with_max_sum(
            logits_max, sum_exp_logits, predicted_logits,
            label_smoothing=0.0, vocab_parallel_logits=vocab_parallel_logits
        )

        self.assertEqual(loss.shape, torch.Size([batch]))
        self.assertEqual(softmax.shape, torch.Size([batch, vocab_size]))

    def test_npu_fused_cross_entropy_loss_with_max_sum_default_smoothing(self):
        batch = 16

        logits_max = torch.randn(batch, dtype=torch.float32).npu()
        sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0
        predicted_logits = torch.randn(batch, dtype=torch.float32).npu()

        loss, softmax = torch_npu.npu_fused_cross_entropy_loss_with_max_sum(
            logits_max, sum_exp_logits, predicted_logits
        )

        self.assertEqual(loss.shape, torch.Size([batch]))


if __name__ == "__main__":
    run_tests()
