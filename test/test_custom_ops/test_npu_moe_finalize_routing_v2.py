from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


@dataclass
class MoeFinalizeRoutingData:
    expanded_permuted_rows: np.array
    skip1: np.array
    skip2_optional: np.array
    bias: np.array
    scales: np.array
    expanded_src_to_dst_row: np.array
    expert_for_source_row: np.array
    drop_pad_mode: np.int64


class TestMoeFinalizeRouting(TestCase):
   
    def moe_finalize_routing_np(self, data_struct):
        num_rows = data_struct.expert_for_source_row.shape[0]
        K = data_struct.expanded_src_to_dst_row.shape[0] // num_rows
        H = data_struct.expanded_permuted_rows.shape[1]
        if (data_struct.skip1 is not None) and (data_struct.skip2_optional is not None):
            out = data_struct.skip1 + data_struct.skip2_optional
        elif (data_struct.skip2_optional is not None) and (data_struct.skip1 is None):
            out = deepcopy(data_struct.skip2_optional)
        elif (data_struct.skip2_optional is None) and (data_struct.skip1 is not None):
            out = deepcopy(data_struct.skip1)
        else:
            out = np.zeros([num_rows, H])
        for i in range(num_rows):
            for k in range(K):
                value = data_struct.expanded_src_to_dst_row[k * num_rows + i]
                if value == -1:
                    dst_row = 0
                else:
                    dst_row = data_struct.expanded_permuted_rows[value, :]
                expert_id = data_struct.expert_for_source_row[i, k]
                if data_struct.bias is not None:
                    out[i, :] += data_struct.scales[i, k] * (dst_row + data_struct.bias[expert_id, :])
                else:
                    out[i, :] += data_struct.scales[i, k] * dst_row
        return out
   
    def custom_op_exec(self, data_struct):
        return torch_npu.npu_moe_finalize_routing(torch.tensor(data_struct.expanded_permuted_rows).npu(),
                                                  torch.tensor(data_struct.skip1).npu(),
                                                  torch.tensor(data_struct.skip2_optional).npu(),
                                                  torch.tensor(data_struct.bias).npu(),
                                                  torch.tensor(data_struct.scales).npu(),
                                                  torch.tensor(data_struct.expanded_src_to_dst_row).npu(),
                                                  torch.tensor(data_struct.expert_for_source_row).reshape(5, 4).npu(),
                                                  0)
    
    def generate_input_data(self, expert_num=16, token_len=10, top_k=4, num_rows=50):
        expanded_permuted_rows = np.random.randn(num_rows * top_k, token_len).astype(np.float32)
        skip1 = np.random.randn(num_rows, token_len).astype(np.float32)
        skip2_optional = np.random.randn(num_rows, token_len).astype(np.float32)
        bias = np.random.randn(expert_num, token_len).astype(np.float32)
        scales = np.random.randn(num_rows, top_k).astype(np.float32)
        expanded_src_to_dst_row = np.arange(num_rows * top_k).astype(np.int32)
        np.random.shuffle(expanded_src_to_dst_row)
        expert_for_source_row = np.random.randint(low=0, high=expert_num, size=(num_rows, top_k)).astype(np.int32)
        data_struct = MoeFinalizeRoutingData(expanded_permuted_rows, skip1, skip2_optional, bias, scales,
                                             expanded_src_to_dst_row, expert_for_source_row, 0)
        return data_struct
    
    @SupportedDevices(['Ascend910B'])
    def test_moe_finalize_routing(self, device="npu"):
        data_struct = self.generate_input_data(expert_num=16, token_len=5, top_k=4, num_rows=5)

        expected_output = self.moe_finalize_routing_np(data_struct)
        custom_output = self.custom_op_exec(data_struct)
        self.assertRtolEqual(expected_output, custom_output.cpu().numpy(), 0.0001)

if __name__ == "__main__":
    run_tests()        