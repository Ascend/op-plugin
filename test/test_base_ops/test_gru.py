import copy
import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGru(TestCase):
    @unittest.skip("skip test_gru_fp32: aclnnGRU not in CANN yet. Remove this skip after CANN update.")
    def test_gru_fp32(self):
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            # single layer, unidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, True, False],
            # single layer, unidirectional, with bias, batch_first=True
            [[np.float32, (2, 3, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, True, True],
            # single layer, bidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, True, False],
            # two layers, unidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (2, 2, 3)], 4, 3, 2, False, True, False],
            # two layers, bidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (4, 2, 3)], 4, 3, 2, True, True, False],
            # single layer, unidirectional, without bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, False, False],
            # single layer, bidirectional, without bias, batch_first=True
            [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, False, True],
            # two layers, unidirectional, without bias, batch_first=True
            [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 2, False, False, True],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1)
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            cpu_h0 = torch.from_numpy(h0)
            npu_h0 = torch.from_numpy(h0).npu()

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
            npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            # Ascend fp32 isn't enough precision, relaxation of precision requirement temporary
            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy())
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy())

    @unittest.skip("skip test_gru_fp16: aclnnGRU not in CANN yet. Remove this skip after CANN update.")
    def test_gru_fp16(self):
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            [[np.float16, (3, 2, 4)], [np.float16, (1, 2, 3)], 4, 3, 1, False, True, False],
            [[np.float16, (2, 3, 4)], [np.float16, (1, 2, 3)], 4, 3, 1, False, True, True],
            [[np.float16, (3, 2, 4)], [np.float16, (2, 2, 3)], 4, 3, 1, True, True, False],
            [[np.float16, (3, 2, 4)], [np.float16, (2, 2, 3)], 4, 3, 2, False, True, False],
            [[np.float16, (3, 2, 4)], [np.float16, (4, 2, 3)], 4, 3, 2, True, True, False],
            [[np.float16, (3, 2, 4)], [np.float16, (1, 2, 3)], 4, 3, 1, False, False, False],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1.astype(np.float32))
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            cpu_h0 = torch.from_numpy(h0.astype(np.float32))
            npu_h0 = torch.from_numpy(h0).npu()

            # CPU keeps fp32 for higher precision reference; NPU uses fp16
            npu_gru = npu_gru.to(torch.float16)
            npu_input1 = npu_input1.to(torch.float16)
            npu_h0 = npu_h0.to(torch.float16)

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
            npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            self.assertRtolEqual(cpu_output_y.detach().numpy().astype(np.float16),
                                 npu_output_y.cpu().detach().numpy())
            self.assertRtolEqual(cpu_output_h.detach().numpy().astype(np.float16),
                                 npu_output_h.cpu().detach().numpy())

    @unittest.skip("skip test_gru_eval_mode: aclnnGRU not in CANN yet. Remove this skip after CANN update.")
    def test_gru_eval_mode(self):
        # Test inference mode (train=False)
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, True, False],
            [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, True, True],
            [[np.float32, (3, 2, 4)], [np.float32, (4, 2, 3)], 4, 3, 2, True, True, False],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            cpu_gru.eval()
            npu_gru.eval()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1)
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            cpu_h0 = torch.from_numpy(h0)
            npu_h0 = torch.from_numpy(h0).npu()

            with torch.no_grad():
                cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
                npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy())
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy())

    @unittest.skip("skip test_gru_sequence: aclnnGRU does not support packed sequence (batch_sizes) yet")
    def test_gru_sequence(self):
        max_len = 6
        embedding_size = 2
        hidden_size = 16
        vocab_size = 20
        input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
        lengths = [5, 3, 6]

        # embedding
        embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        rnn = torch.nn.GRU(embedding_size, hidden_size)
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
        pade_outputs, hn = rnn(pack)
        pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

        # cacl npu
        embeded_npu = embeded.npu()
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded_npu, lengths, batch_first=False)
        pade_outputs_npu, hn_n = rnn_npu(pack)
        pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

        self.assertRtolEqual(pade_outputs.detach().numpy(),
                             pade_outputs_npu.cpu().to(torch.float).detach().numpy(), prec=1.e-4)

    @unittest.skip("skip test_gru_sequence_bidirection: aclnnGRU does not support packed sequence (batch_sizes) yet")
    def test_gru_sequence_bidirection(self):
        max_len = 6
        embedding_size = 2
        hidden_size = 16
        vocab_size = 20
        input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
        lengths = [5, 3, 6]

        # embedding
        embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        rnn = torch.nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, bias=False)
        rnn_npu = copy.deepcopy(rnn).npu()

        # Sorting from Large to Small
        input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
        lengths = sorted(lengths, key=lambda tp: tp, reverse=True)

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
        pade_outputs, hn = rnn(pack)
        pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

        # cacl npu
        embeded_npu = embeded.npu()
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded_npu, lengths, batch_first=False)
        pade_outputs_npu, hn_n = rnn_npu(pack)
        pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

        self.assertRtolEqual(pade_outputs.detach().numpy(),
                             pade_outputs_npu.cpu().detach().numpy(), prec=1.e-4)

    @unittest.skip("skip test_gru_sequence_double_layer: aclnnGRU does not support packed sequence (batch_sizes) yet")
    def test_gru_sequence_double_layer(self):
        for item in [True, False]:
            max_len, embedding_size, hidden_size, vocab_size = 6, 2, 16, 20
            input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
            lengths = [5, 3, 6]

            embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            rnn = torch.nn.GRU(embedding_size, hidden_size, num_layers=2, bidirectional=item, bias=False)
            rnn_npu = copy.deepcopy(rnn).npu()

            # Sorting from Large to Small
            input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
            lengths = sorted(lengths, key=lambda tp: tp, reverse=True)

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
            pade_outputs, hn = rnn(pack)
            pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

            # cacl npu
            pack = torch.nn.utils.rnn.pack_padded_sequence(embeded.npu(), lengths, batch_first=False)
            pade_outputs_npu, hn_n = rnn_npu(pack)
            pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

            self.assertRtolEqual(pade_outputs.detach().numpy(),
                                 pade_outputs_npu.cpu().detach().numpy(), prec=1.e-4)


if __name__ == "__main__":
    run_tests()
