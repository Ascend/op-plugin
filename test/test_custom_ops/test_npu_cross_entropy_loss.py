import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUCrossEntropyLoss(TestCase):

    # pylint:disable = huawei-too-many-arguments
    def cross_entropy_loss_golden(self, N, C, predictions, targets, weight, ignore_index, label_smoothing, reduction):
        if weight is None:
            weight = torch.ones((C,)).npu()
        predictions_cast = predictions.to(torch.float32)
        predictions_max = torch.max(predictions_cast, dim=1)[0].unsqueeze(-1)
        log_softmax_probs = (predictions_cast - predictions_max -
                             torch.log(torch.sum(torch.exp(predictions_cast - predictions_max), -1, keepdim=True)))
        nll_loss = torch.gather(log_softmax_probs, 1, targets.reshape(-1, 1)).reshape(-1)
        weight_yn = torch.gather(weight, 0, targets)
        loss_out = -nll_loss * weight_yn
        if ignore_index >= 0:
            ignore_mask = (targets - ignore_index).bool().float().npu()
        else:
            ignore_mask = torch.ones((N,)).npu()
        loss_out = loss_out * ignore_mask
        
        smooth_loss = -torch.sum(log_softmax_probs * weight.unsqueeze(0), -1, keepdim=False)
        if ignore_index >= 0:
            smooth_loss = smooth_loss * ignore_mask
        
        if reduction == "mean":
            weight_after_mask = weight_yn * ignore_mask
            mean_out = torch.sum(loss_out, -1, keepdim=False) / torch.sum(weight_after_mask, -1, keepdim=False)
            ret = ((1 - label_smoothing) * mean_out + torch.sum(smooth_loss, -1, keepdim=False)
                   / torch.sum(weight_after_mask, -1, keepdim=False) * label_smoothing / C).reshape(1)
        elif reduction == "sum":
            sum_out = torch.sum(loss_out, -1, keepdim=False)
            ret = (1 - label_smoothing) * sum_out + torch.sum(smooth_loss, -1, keepdim=False) * label_smoothing / C
            ret = ret.reshape(1)
        else:
            none_out = loss_out
            ret = (1 - label_smoothing) * none_out + smooth_loss * label_smoothing / C
        ret = ret.to(predictions.dtype)
        log_softmax_probs = log_softmax_probs.to(predictions.dtype)
        return ret, log_softmax_probs

    # pylint:disable = huawei-too-many-arguments
    def cross_entropy_loss_custom(self, inputs, target, weight, reduction, ignore_index=-100, label_smoothing=0.0):
        loss, log_probs, _, _ = torch_npu.npu_cross_entropy_loss(inputs, target, weight,
                                                                 reduction=reduction, ignore_index=ignore_index,
                                                                 label_smoothing=label_smoothing)
        return loss, log_probs

    @unittest.skip("Skipping test_npu_cross_entropy_loss for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_cross_entropy_loss(self):
        shape_list = [[4096, 8080], [4096, 8192]]
        dtype_list = [torch.float32, torch.float16, torch.bfloat16]
        reduction_list = ["mean", "sum", "none"]

        generate_list = []
        for shape in shape_list:
            for dtype in dtype_list:
                for reduction in reduction_list:
                    generate_list.append([shape, dtype, reduction])
        for item in generate_list:
            N = item[0][0]
            C = item[0][1]
            input_npu = torch.randn(N, C).to(item[1]).npu()
            target_npu = torch.arange(0, N).npu()
            input_golden = input_npu.clone()
            target_golden = target_npu.clone()
            
            golden0, golden1 = self.cross_entropy_loss_golden(N, C, input_golden,
                                                              target_golden, weight=None, ignore_index=-100,
                                                              label_smoothing=0.0, reduction=item[2])
            out0, out1 = self.cross_entropy_loss_custom(input_npu, target_npu, None, reduction=item[2])
            if item[1] == torch.bfloat16:
                self.assertRtolEqual(golden0, out0, prec16=2**(-6))
                self.assertRtolEqual(golden1, out1, prec16=2**(-6))
            elif item[1] == torch.float16:
                self.assertRtolEqual(golden0, out0, 2**(-7))
                self.assertRtolEqual(golden1, out1, 2**(-7))
            else:
                self.assertRtolEqual(golden0, out0, 2**(-10))
                self.assertRtolEqual(golden1, out1, 2**(-10))
    
    @unittest.skip("Skipping test_npu_cross_entropy_loss_backward for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_cross_entropy_loss_backward(self):
        shape_list = [[4096, 8080], [4096, 8192]]
        dtype_list = [torch.float32, torch.float16, torch.bfloat16]
        reduction_list = ["mean", "none"]

        generate_list = []
        for shape in shape_list:
            for dtype in dtype_list:
                for reduction in reduction_list:
                    generate_list.append([shape, dtype, reduction])
        for item in generate_list:
            N = item[0][0]
            C = item[0][1]
            input_npu = torch.randn(N, C).to(item[1]).npu()
            target_npu = torch.arange(0, N).npu()
            input_golden = input_npu.clone()
            target_golden = target_npu.clone()
            
            input_golden.requires_grad_()
            input_npu.requires_grad_()
            
            golden0, _ = self.cross_entropy_loss_golden(N, C, input_golden,
                                                              target_golden, weight=None, ignore_index=-100,
                                                              label_smoothing=0.0, reduction=item[2])
            out0, _ = self.cross_entropy_loss_custom(input_npu, target_npu, None, reduction=item[2])
            golden0.sum().backward()
            out0.sum().backward()
            golden_grad = input_golden.grad.detach()
            input_grad = input_npu.grad.detach()
            if item[1] == torch.bfloat16:
                self.assertRtolEqual(golden_grad, input_grad, prec16=2**(-6))
            elif item[1] == torch.float16:
                self.assertRtolEqual(golden_grad, input_grad, 2**(-7))
            else:
                self.assertRtolEqual(golden_grad, input_grad, 2**(-10))


if __name__ == "__main__":
    run_tests()
