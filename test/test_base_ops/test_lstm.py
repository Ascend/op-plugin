import copy
import torch
import numpy as np
import torch_npu
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch_npu.testing.testcase import TestCase, run_tests


class TestLstm(TestCase):
    device = "npu"

    def _build_lstm(self, dtype=torch.float32, num_layers=2):
        model = torch.nn.LSTM(
            input_size=4,
            hidden_size=6,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
        ).to(self.device, dtype=dtype)
        model.eval()
        return model

    def _make_dense_inputs(self, dtype):
        x = torch.randn(5, 3, 4, device=self.device, dtype=dtype)
        h0 = torch.randn(2, 3, 6, device=self.device, dtype=dtype)
        c0 = torch.randn(2, 3, 6, device=self.device, dtype=dtype)
        return x, h0, c0

    def _make_packed_inputs(self, dtype, num_layers=2):
        lengths = [5, 3, 2]
        padded = torch.randn(5, 3, 4, device=self.device, dtype=dtype)
        packed = pack_padded_sequence(padded, lengths, enforce_sorted=True)
        h0 = torch.randn(num_layers, 3, 6, device=self.device, dtype=dtype)
        c0 = torch.randn(num_layers, 3, 6, device=self.device, dtype=dtype)
        return packed, h0, c0

    def _run_raw_lstm(self, model, x, hx):
        model._update_flat_weights()
        params = tuple(model._flat_weights)
        return torch._VF.lstm(
            x, hx, params,
            model.bias,
            model.num_layers,
            float(model.dropout),
            model.training,
            model.bidirectional,
            model.batch_first,
        )

    def _run_raw_packed_lstm(self, model, packed, hx):
        model._update_flat_weights()
        params = tuple(model._flat_weights)
        return torch._VF.lstm(
            packed.data,
            packed.batch_sizes,
            hx,
            params,
            model.bias,
            model.num_layers,
            float(model.dropout),
            model.training,
            model.bidirectional,
        )

    def test_lstm_dense_fallback(self):
        model = self._build_lstm(dtype=torch.float32)
        x, h0, c0 = self._make_dense_inputs(torch.float16)
        y, h, c = self._run_raw_lstm(model, x, (h0, c0))

        self.assertEqual(y.shape, (5, 3, 6))
        self.assertEqual(h.shape, (2, 3, 6))
        self.assertEqual(c.shape, (2, 3, 6))

    def _assert_packed_fallback(self, model_dtype, input_dtype, num_layers=1):
        model = self._build_lstm(dtype=model_dtype, num_layers=num_layers)
        packed, h0, c0 = self._make_packed_inputs(input_dtype, num_layers=num_layers)
        y, h, c = self._run_raw_packed_lstm(model, packed, (h0, c0))

        self.assertEqual(y.shape, (packed.data.size(0), 6))
        self.assertEqual(h.shape, (num_layers, 3, 6))
        self.assertEqual(c.shape, (num_layers, 3, 6))

    def test_lstm_packed_mixed_dtype_fallback(self):
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            self._assert_packed_fallback(torch.float32, torch.float16)

    def test_lstm_packed_autocast_fallback(self):
        jit_compile = not torch_npu.npu.is_jit_compile_false()
        try:
            for jit in (False, True):
                torch_npu.npu.set_compile_mode(jit_compile=jit)
                for device_type in (self.device, "cpu"):
                    with self.subTest(jit=jit, autocast=device_type), torch.autocast(device_type=device_type):
                        model = self._build_lstm(num_layers=1)
                        packed, h0, c0 = self._make_packed_inputs(torch.float16, num_layers=1)
                        output, (h, c) = model(packed, (h0, c0))
                        self.assertEqual(output.data.shape, (packed.data.size(0), 6))
                        self.assertEqual(h.shape, h0.shape)
                        self.assertEqual(c.shape, c0.shape)
                        self.assertTrue(torch.isfinite(output.data).all().item())
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=jit_compile)

    def test_lstm_packed_bf16_fallback(self):
        self._assert_packed_fallback(torch.bfloat16, torch.bfloat16)

    def test_lstm_packed_params_size_fallback(self):
        self._assert_packed_fallback(torch.float32, torch.float32, num_layers=2)

    def test_lstm_packed_aclop_output_shape(self):
        jit_compile = not torch_npu.npu.is_jit_compile_false()
        torch_npu.npu.set_compile_mode(jit_compile=True)
        try:
            model = self._build_lstm(dtype=torch.float32, num_layers=1)
            packed, h0, c0 = self._make_packed_inputs(torch.float32, num_layers=1)
            y, h, c = self._run_raw_packed_lstm(model, packed, (h0, c0))

            self.assertEqual(y.shape, (packed.data.size(0), 6))
            self.assertEqual(h.shape, (1, 3, 6))
            self.assertEqual(c.shape, (1, 3, 6))

            output = PackedSequence(y, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
            padded, _ = pad_packed_sequence(output)
            self.assertEqual(padded.shape, (5, 3, 6))
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=jit_compile)

    def test_lstm_params_size_fallback(self):
        model = torch.nn.LSTM(
            input_size=2,
            hidden_size=3,
            num_layers=3,
            bias=False,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
        ).to(self.device)
        model.eval()
        x = torch.randn(4, 1, 2, device=self.device)
        h0 = torch.randn(3, 1, 3, device=self.device)
        c0 = torch.randn(3, 1, 3, device=self.device)
        y, h, c = self._run_raw_lstm(model, x, (h0, c0))

        self.assertEqual(len(tuple(model._flat_weights)), 6)
        self.assertEqual(y.shape, (4, 1, 3))
        self.assertEqual(h.shape, (3, 1, 3))
        self.assertEqual(c.shape, (3, 1, 3))

    def test_lstm_single_direction(self):
        # shape_format:[[dtype, (num_step, batch_size, input_size)],
        # num_layers, input_size, hidden_size, is_training, batch_first]
        shape_format = [
            [[np.float32, (5, 32, 64)], 1, 64, 64, True, False],
            [[np.float32, (5, 32, 64)], 1, 64, 64, False, False],
            [[np.float32, (26, 2560, 512)], 1, 512, 256, False, True],
            [[np.float32, (10, 33, 128)], 1, 128, 64, False, False],
            [[np.float32, (5, 32, 64)], 2, 64, 64, True, False],
            [[np.float32, (5, 32, 64)], 2, 64, 64, False, False],
            [[np.float32, (26, 2560, 512)], 2, 512, 256, False, True],
            [[np.float32, (10, 33, 128)], 2, 128, 64, False, False],
        ]

        for item in shape_format:
            cpu_lstm = torch.nn.LSTM(input_size=item[2], hidden_size=item[3], batch_first=item[5],
                                     num_layers=item[1], bidirectional=False, bias=False)
            cpu_lstm.training = item[4]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float16).astype(np.float32)
            cpu_input1 = torch.from_numpy(input1)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(),
                                 npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(),
                                 npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(),
                                 npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-3)

    def test_lstm_bidirection(self):
        # shape_format:[[dtype, (num_step, batch_size, input_size)],
        # num_layers, input_size, hidden_size, is_training]
        shape_format = [
            [[np.float32, (5, 32, 64)], 1, 64, 64, True], [[np.float32, (5, 32, 64)], 1, 64, 64, False],
            [[np.float32, (26, 2560, 512)], 1, 512, 256, False], [[np.float32, (10, 33, 128)], 1, 128, 64, False],
            [[np.float32, (5, 32, 64)], 2, 64, 64, True], [[np.float32, (5, 32, 64)], 2, 64, 64, False],
            [[np.float32, (26, 2560, 512)], 2, 512, 256, False], [[np.float32, (10, 33, 128)], 2, 128, 64, False],
        ]

        for item in shape_format:
            cpu_lstm = torch.nn.LSTM(input_size=item[2], hidden_size=item[3], batch_first=True,
                                     num_layers=item[1], bidirectional=True, bias=False)
            cpu_lstm.training = item[4]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float16).astype(np.float32)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(torch.from_numpy(input1))
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(torch.from_numpy(input1.astype(item[0][0])).npu())

            self.assertRtolEqual(cpu_output_y.detach().numpy(),
                                 npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(),
                                 npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(),
                                 npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-3)

    def test_lstm_sequence(self):
        max_len = 6
        embedding_size = 2
        hidden_size = 16
        vocab_size = 20
        input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
        lengths = [5, 3, 6]

        # embedding
        embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        rnn = torch.nn.LSTM(embedding_size, hidden_size)
        rnn_npu = copy.deepcopy(rnn).npu()

        # Sorting from Large to Small
        input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
        lengths = sorted(lengths, key=lambda tp: tp, reverse=True)
        '''
        outputs:
        input_seq: [[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2], [4, 11, 14]]
        lengths : [6, 5, 3]
        '''

        def pad_seq(seq, seq_len, max_length):
            # The padding subscript is 0
            pad_token = 0
            seq += [pad_token for _ in range(max_length - seq_len)]
            return seq

        # Data after padding
        pad_seqs = []
        for i, j in zip(input_seq, lengths):
            pad_seqs.append(pad_seq(i, j, max_len))

        lengths = [6, 5, 3]
        pad_seqs = torch.tensor(pad_seqs)
        embeded = embedding(pad_seqs)
        embeded = embeded.reshape(6, 3, 2)
        embeded = embeded.to(torch.float16).to(torch.float32)

        # cacl cpu
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=False)
        pade_outputs, (hn, cn) = rnn(pack)
        pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

        # cacl npu
        embeded_npu = embeded.npu()
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded_npu, lengths, batch_first=False)
        pade_outputs_npu, (hn_n, cn_n) = rnn_npu(pack)
        pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

        self.assertRtolEqual(pade_outputs.detach().numpy(),
                             pade_outputs_npu.cpu().to(torch.float).detach().numpy(), prec=1.e-4)

    def test_lstm_sequence_bidirection(self):
        max_len = 6
        embedding_size = 2
        hidden_size = 16
        vocab_size = 20
        input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
        lengths = [5, 3, 6]

        # embedding
        embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers=1, bidirectional=True, bias=False)
        rnn_npu = copy.deepcopy(rnn).npu()

        # Sorting from Large to Small
        input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
        lengths = sorted(lengths, key=lambda tp: tp, reverse=True)
        '''
        outputs:
        input_seq: [[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2], [4, 11, 14]]
        lengths : [6, 5, 3]
        '''

        def pad_seq(seq, seq_len, max_length):
            # The padding subscript is 0
            pad_token = 0
            seq += [pad_token for _ in range(max_length - seq_len)]
            return seq

        # Data after padding
        pad_seqs = []
        for i, j in zip(input_seq, lengths):
            pad_seqs.append(pad_seq(i, j, max_len))

        lengths = [6, 5, 3]
        pad_seqs = torch.tensor(pad_seqs)
        embeded = embedding(pad_seqs)
        embeded = embeded.reshape(6, 3, 2)
        embeded = embeded.to(torch.float16).to(torch.float32)

        # cacl cpu
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=False)
        pade_outputs, (hn, cn) = rnn(pack)
        pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

        # cacl npu
        embeded_npu = embeded.npu()
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded_npu, lengths, batch_first=False)
        pade_outputs_npu, (hn_n, cn_n) = rnn_npu(pack)
        pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

        self.assertRtolEqual(pade_outputs.detach().numpy(),
                             pade_outputs_npu.cpu().detach().numpy(), prec=1.e-4)

    def test_lstm_sequence_double_layer(self):
        for item in [True, False]:
            max_len, embedding_size, hidden_size, vocab_size = 6, 2, 16, 20
            input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
            lengths = [5, 3, 6]

            embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers=2, bidirectional=item, bias=False)
            rnn_npu = copy.deepcopy(rnn).npu()

            # Sorting from Large to Small
            input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
            lengths = sorted(lengths, key=lambda tp: tp, reverse=True)

            # The padding subscript is 0
            pad_token = 0

            def pad_seq(seq, seq_len, max_length):
                # The padding subscript is 0
                pad_token = 0
                seq += [pad_token for _ in range(max_length - seq_len)]
                return seq

            # Data after padding
            pad_seqs = [pad_seq(i, j, max_len) for i, j in zip(input_seq, lengths)]

            lengths = [6, 5, 3]
            pad_seqs = torch.tensor(pad_seqs)
            embeded = embedding(pad_seqs)
            embeded = embeded.reshape(6, 3, 2)
            embeded = embeded.to(torch.float16).to(torch.float32)

            # cacl cpu
            pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=False)
            pade_outputs, (hn, cn) = rnn(pack)
            pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

            # cacl npu
            pack = torch.nn.utils.rnn.pack_padded_sequence(embeded.npu(), lengths, batch_first=False)
            pade_outputs_npu, (hn_n, cn_n) = rnn_npu(pack)
            pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

            self.assertRtolEqual(pade_outputs.detach().numpy(), pade_outputs_npu.cpu().detach().numpy(), prec=1.e-4)


if __name__ == "__main__":
    run_tests()
