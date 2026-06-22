import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuFusedLinearCrossEntropyLossWithMaxSumBackward(TestCase):
    def test_backward_high_perf_mode_fp16(self):
        batch = 128
        hidden = 64
        vocab_size = 256

        grad = torch.randn(batch, dtype=torch.float32).npu()
        input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
        weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
        target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
        masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
        softmax = torch.randn(batch, vocab_size, dtype=torch.float32).npu()

        input_grad, weight_grad = torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(
            grad, input_tensor, weight_tensor, target_mask, masked_target,
            label_smoothing=0.0, logits_max=None, sum_exp_logits=None, softmax=softmax
        )

        self.assertEqual(input_grad.shape, torch.Size([batch, hidden]))
        self.assertEqual(weight_grad.shape, torch.Size([vocab_size, hidden]))
        self.assertEqual(input_grad.dtype, torch.float16)
        self.assertEqual(weight_grad.dtype, torch.float16)

    def test_backward_high_perf_mode_bf16(self):
        batch = 64
        hidden = 32
        vocab_size = 128

        grad = torch.randn(batch, dtype=torch.float32).npu()
        input_tensor = torch.randn(batch, hidden, dtype=torch.bfloat16).npu()
        weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.bfloat16).npu()
        target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
        masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
        softmax = torch.randn(batch, vocab_size, dtype=torch.float32).npu()

        input_grad, weight_grad = torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(
            grad, input_tensor, weight_tensor, target_mask, masked_target,
            label_smoothing=0.0, softmax=softmax
        )

        self.assertEqual(input_grad.shape, torch.Size([batch, hidden]))
        self.assertEqual(weight_grad.shape, torch.Size([vocab_size, hidden]))
        self.assertEqual(input_grad.dtype, torch.bfloat16)
        self.assertEqual(weight_grad.dtype, torch.bfloat16)

    def test_backward_memory_save_mode_fp16(self):
        batch = 128
        hidden = 64
        vocab_size = 256

        grad = torch.randn(batch, dtype=torch.float32).npu()
        input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
        weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
        target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
        masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
        logits_max = torch.randn(batch, dtype=torch.float32).npu()
        sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0

        input_grad, weight_grad = torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(
            grad, input_tensor, weight_tensor, target_mask, masked_target,
            label_smoothing=0.0, logits_max=logits_max, sum_exp_logits=sum_exp_logits, softmax=None
        )

        self.assertEqual(input_grad.shape, torch.Size([batch, hidden]))
        self.assertEqual(weight_grad.shape, torch.Size([vocab_size, hidden]))
        self.assertEqual(input_grad.dtype, torch.float16)
        self.assertEqual(weight_grad.dtype, torch.float16)

    def test_backward_int64_target(self):
        batch = 64
        hidden = 32
        vocab_size = 128

        grad = torch.randn(batch, dtype=torch.float32).npu()
        input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
        weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
        target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
        masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int64).npu()
        softmax = torch.randn(batch, vocab_size, dtype=torch.float32).npu()

        input_grad, weight_grad = torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(
            grad, input_tensor, weight_tensor, target_mask, masked_target,
            label_smoothing=0.0, softmax=softmax
        )

        self.assertEqual(input_grad.shape, torch.Size([batch, hidden]))
        self.assertEqual(weight_grad.shape, torch.Size([vocab_size, hidden]))


if __name__ == "__main__":
    run_tests()
