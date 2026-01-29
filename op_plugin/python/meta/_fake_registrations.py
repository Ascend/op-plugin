import torch
import torch_npu
from torch_npu.utils._error_code import ErrCode, ops_error


@torch.library.register_fake("npu::npu_fusion_attention_v3")
def npu_fusion_attention_forward_v3(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
                                inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                                 gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
    B = query.size(0)
    S1 = query.size(2)
    T = query.size(0)
    N_L = query.size(1)
    aten_score_shape = query.shape

    if input_layout == "BSH":
        S1 = query.size(1)
        H = query.size(2)
        D = H / head_num
        D2 = 0 if D == 0 or key.size(2) == 0 else value.size(2) / (key.size(2) / D)
        aten_score_shape = [B, S1, head_num * D2]
    elif input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        H = query.size(2)
        D = H / head_num
        D2 = 0 if D == 0 or key.size(2) == 0 else value.size(2) / (key.size(2) / D)
        aten_score_shape = [S1, B, head_num * D2]
    elif input_layout == "BNSD":
        D2 = value.size(3)
        aten_score_shape = [B, N_L, S1, D2]
    elif input_layout == "BSND":
        S1 = query.size(1)
        N_L = query.size(2)
        D2 = value.size(3)
        aten_score_shape = [B, S1, N_L, D2]
    elif input_layout == "TND":
        D2 = value.size(2)
        aten_score_shape = [T, N_L, D2]

    if input_layout == "TND":
        softmax_shape = [T, N_L, 8]
    else:
        softmax_shape = [B, head_num, S1, 8]

    seed = torch.empty([1], dtype=torch.long, device='cpu')
    offset = torch.empty([1], dtype=torch.long, device='cpu')
    attention_score = query.new_empty(aten_score_shape, dtype=query.dtype, device=query.device)
    softmax_max = torch.empty(softmax_shape, dtype=torch.float32, device=query.device)
    softmax_sum = torch.empty(softmax_shape, dtype=torch.float32, device=query.device)
    softmax_out = torch.empty([0], dtype=query.dtype, device=query.device)
    return (attention_score, softmax_max, softmax_sum, softmax_out, seed, offset)


@torch.library.register_fake("npu::npu_fusion_attention_grad_v3")
def npu_fusion_attention_backward_v3(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                                  softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, scale_value=1.0,
                                  keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, seed=None, offset=None,
                                  prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                                  gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
    dq = query.new_empty(query.shape, dtype=query.dtype, device=query.device)
    dk = key.new_empty(key.shape, dtype=query.dtype, device=query.device)
    dv = value.new_empty(value.shape, dtype=query.dtype, device=query.device)
    dpse = torch.empty([0], dtype=query.dtype, device=query.device)
    dsink = torch.empty([], device=query.device) if sink is None else torch.empty(sink.shape, dtype=sink.dtype, device=query.device)
    return (dq, dk, dv, dpse, dsink)
