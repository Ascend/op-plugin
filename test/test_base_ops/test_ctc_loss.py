import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestCtcLoss(TestCase):
    def generate_data(self, item):
        T = item[0][0]
        C = item[0][1]
        N = item[0][2]
        S = item[0][3]
        S_min = item[0][4]
        dtype = item[1]

        log_probs = np.random.uniform(-10, 10, (T, N, C)).astype(dtype)
        targets = torch.randint(1, C, (N, S), dtype=torch.long)
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(S_min, S, (N,), dtype=torch.long)

        # modify from numpy.ndarray to torch.tensor
        log_probs = torch.from_numpy(log_probs)

        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction=item[2], blank=item[3])

        # pylint:disable=too-many-return-values
        return ctc_loss, log_probs, targets, input_lengths, target_lengths

    def generate_data_targets_1d(self, item):
        T = item[0][0]
        C = item[0][1]
        N = item[0][2]
        S = item[0][3]
        S_min = item[0][4]
        dtype = item[1]

        log_probs = np.random.uniform(-5, 0, (T, N, C)).astype(dtype)

        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(S_min, S, (N,), dtype=torch.long)

        target_lengths_sum = torch.sum(target_lengths)
        target_length = target_lengths_sum.item()
        targets = torch.randint(1, C, (target_length,), dtype=torch.long)

        log_probs = torch.from_numpy(log_probs)
        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction=item[2], blank=item[3])

        # pylint:disable=too-many-return-values
        return ctc_loss, log_probs, targets, input_lengths, target_lengths
    
    def generate_data_log_probs_2d(self, item):
        T = item[0][0]
        C = item[0][1]
        N = 1
        S = item[0][3]
        S_min = item[0][4]
        dtype = item[1]

        log_probs = np.random.uniform(-5, 0, (T, C)).astype(dtype)
        targets = torch.randint(1, C, (N, S), dtype=torch.long)
        input_lengths = torch.tensor(T).to(torch.long)
        target_lengths = torch.tensor(S).to(torch.long)

        # modify from numpy.ndarray to torch.tensor
        log_probs = torch.from_numpy(log_probs)

        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction=item[2], blank=item[3])

        # pylint:disable=too-many-return-values
        return ctc_loss, log_probs, targets, input_lengths, target_lengths

    def cpu_op_exec(self, ctc_loss, log_probs, targets, input_lengths, target_lengths):
        if log_probs.dtype == torch.float16:
            log_probs = log_probs.to(torch.float32)

        neg_log_likelihood = ctc_loss(log_probs.log_softmax(2), targets, input_lengths, target_lengths)

        neg_log_likelihood = neg_log_likelihood.numpy()

        return neg_log_likelihood

    def npu_op_exec(self, ctc_loss, log_probs, targets, input_lengths, target_lengths):
        log_probs = log_probs.npu()
        targets = targets.npu()
        input_lengths = input_lengths.npu()
        target_lengths = target_lengths.npu()

        neg_log_likelihood = ctc_loss(log_probs.log_softmax(2), targets, input_lengths, target_lengths)

        if neg_log_likelihood.dtype == torch.float16:
            neg_log_likelihood = neg_log_likelihood.to(torch.float32)

        neg_log_likelihood = neg_log_likelihood.cpu().numpy()

        return neg_log_likelihood
    
    def cpu_op_exec_2d(self, ctc_loss, log_probs, targets, input_lengths, target_lengths):
        if log_probs.dtype == torch.float16:
            log_probs = log_probs.to(torch.float32)

        neg_log_likelihood = ctc_loss(log_probs.log_softmax(1), targets, input_lengths, target_lengths)

        neg_log_likelihood = neg_log_likelihood.numpy()

        return neg_log_likelihood

    def npu_op_exec_2d(self, ctc_loss, log_probs, targets, input_lengths, target_lengths):
        log_probs = log_probs.npu()
        targets = targets.npu()
        input_lengths = input_lengths.npu()
        target_lengths = target_lengths.npu()

        neg_log_likelihood = ctc_loss(log_probs.log_softmax(1), targets, input_lengths, target_lengths)

        if neg_log_likelihood.dtype == torch.float16:
            neg_log_likelihood = neg_log_likelihood.to(torch.float32)

        neg_log_likelihood = neg_log_likelihood.cpu().numpy()

        return neg_log_likelihood

    def test_ctc_loss(self):
        sizes_list = [[50, 20, 16, 30, 10], [26, 37, 16, 18, 10]]
        para_reduction = ["sum", "mean", "none"]
        dtype_list = [np.float32]
        blank_list = [0, 9]
        # pylint:disable=complicate-comprehension
        shape_format = [
            [sizes, dtype, para, blank] for sizes in sizes_list for dtype in dtype_list for para in para_reduction for blank in blank_list
        ]

        for item in shape_format:
            ctc_loss, log_probs, targets, input_lengths, target_lengths = self.generate_data(item)

            neg_log_likelihood_cpu = self.cpu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)
            neg_log_likelihood_npu = self.npu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)

            self.assertRtolEqual(neg_log_likelihood_cpu, neg_log_likelihood_npu, 1e-3)

    def test_ctc_loss_targets_1d(self):
        sizes_list = [[50, 20, 16, 30, 10], [26, 37, 16, 18, 10]]
        para_reduction = ["sum", "mean", "none"]
        dtype_list = [np.float32]
        blank_list = [0, 9]
        # pylint:disable=complicate-comprehension
        shape_format = [
            [sizes, dtype, para, blank] for sizes in sizes_list for dtype in dtype_list for para in para_reduction for blank in blank_list
        ]

        for item in shape_format:
            ctc_loss, log_probs, targets, input_lengths, target_lengths = self.generate_data_targets_1d(item)

            neg_log_likelihood_cpu = self.cpu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)
            neg_log_likelihood_npu = self.npu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)

            self.assertRtolEqual(neg_log_likelihood_cpu, neg_log_likelihood_npu, 1e-3)

    def test_ctc_loss_log_probs_2d(self):
        sizes_list = [[50, 20, 16, 30, 10], [26, 37, 16, 18, 10]]
        para_reduction = ["sum", "mean", "none"]
        dtype_list = [np.float32]
        blank_list = [0, 9]
        # pylint:disable=complicate-comprehension
        shape_format = [
            [sizes, dtype, para, blank] for sizes in sizes_list for dtype in dtype_list for para in para_reduction for blank in blank_list
        ]

        for item in shape_format:
            ctc_loss, log_probs, targets, input_lengths, target_lengths = self.generate_data_log_probs_2d(item)

            neg_log_likelihood_cpu = self.cpu_op_exec_2d(ctc_loss, log_probs, targets, input_lengths, target_lengths)
            neg_log_likelihood_npu = self.npu_op_exec_2d(ctc_loss, log_probs, targets, input_lengths, target_lengths)

            self.assertRtolEqual(neg_log_likelihood_cpu, neg_log_likelihood_npu, 1e-3)

    def test_ctc_loss_tensor_no_device_check(self):
        batch_size, time_steps, num_classes, target_length = 4, 50, 20, 30
        log_probs = torch.randn(time_steps, batch_size, num_classes).npu()
        log_probs = torch.nn.functional.log_softmax(log_probs, dim=2)
        targets = torch.randint(1, num_classes, (batch_size, target_length),
                                dtype=torch.long).npu()

        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        target_lengths = torch.full((batch_size,), target_length, dtype=torch.long)

        ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
        loss_npu = ctc_loss(log_probs, targets, input_lengths, target_lengths).cpu().numpy()
        loss_cpu = ctc_loss(log_probs.cpu(), targets.cpu(), input_lengths, target_lengths).numpy()
        self.assertRtolEqual(loss_cpu, loss_npu)


if __name__ == "__main__":
    run_tests()
