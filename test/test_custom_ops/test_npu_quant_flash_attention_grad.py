import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

PER_BLOCK_S1_SIZE = 512
PER_BLOCK_S1_SIZE_FOR_SCALE_DS = PER_BLOCK_S1_SIZE
PER_BLOCK_S2_SIZE = 512
gtype = torch.float32

def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    softmax_res = y.div(x_sum)
    return softmax_res, x_max, x_sum

def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res

def chunked_matmul_with_dv_quant(left: torch.Tensor, right: torch.Tensor, scale_left, scale_right = torch.Tensor) -> torch.Tensor:
    B, N2, G1, S2, S1 = left.shape
    B_r, N2_r, G2, S1_r, D = right.shape
    G_out = max(G1, G2)
    result = torch.zeros((B, N2, G_out, S2, D), 
                        dtype=left.dtype,
                        device=left.device)
    s1_tail = PER_BLOCK_S1_SIZE_FOR_SCALE_DS if S1 % PER_BLOCK_S1_SIZE_FOR_SCALE_DS == 0 else S1 % PER_BLOCK_S1_SIZE_FOR_SCALE_DS
    for b in range(B):
        for n in range(N2):
            for g in range(G_out):
                g_left = 0 if G1 == 1 else g
                g_right = 0 if G2 == 1 else g
                for s2 in range(0, S2, PER_BLOCK_S2_SIZE):   
                    s2_start = s2
                    s2_end = s2_start + PER_BLOCK_S2_SIZE
                    s2_end = min(s2_end, S2)
                    scale_g_idx = 0
                    for s1 in range(0, S1 - s1_tail, PER_BLOCK_S1_SIZE_FOR_SCALE_DS):
                        s1_start = s1
                        s1_end = s1_start + PER_BLOCK_S1_SIZE_FOR_SCALE_DS
                        s1_end = min(s1_end, S1)
                        
                        left_block = left[b, n, g_left, s2_start:s2_end, s1_start:s1_end]
                        scale_left_value = scale_left
                        right_block = right[b, n, g_right, s1_start:s1_end, :]
                        scale_right_value = scale_right[b, n, scale_g_idx, s1_start // PER_BLOCK_S1_SIZE, 0]
                        result[b, n, g, s2_start:s2_end] += torch.matmul(left_block, right_block) * scale_left_value * scale_right_value
                    s1_start = S1 - s1_tail
                    s1_end = S1
                    left_block = left[b, n, g_left, s2_start:s2_end, s1_start:s1_end]
                    scale_left_value = scale_left
                    right_block = right[b, n, g_right, s1_start:s1_end, :]
                    scale_right_value = scale_right[b, n, scale_g_idx, s1_start // PER_BLOCK_S1_SIZE, 0]
                    result[b, n, g, s2_start:s2_end] += torch.matmul(left_block, right_block) * scale_left_value * scale_right_value
    return result

def chunked_matmul_with_quant(left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    B, N2, G1, S1, S2 = left.shape
    B_r, N2_r, G2, S2_r, D = right.shape

    G_out = max(G1, G2)
    result = torch.zeros((B, N2, G_out, S1, D), 
                        dtype=left.dtype,
                        device=left.device)
    for b in range(B):
        for n in range(N2):
            for g in range(G_out):
                g_left = 0 if G1 == 1 else g
                g_right = 0 if G2 == 1 else g

                for start in range(0, S2, PER_BLOCK_S2_SIZE):
                    end = start + PER_BLOCK_S2_SIZE
                    end = min(end, S2)
                    current_chunk = min(PER_BLOCK_S2_SIZE, S2 - start)
                    
                    left_block = left[b, n, g_left, :, start:end]
                    right_block = right[b, n, g_right, start:end, :]
                    
                    scale_g_idx = 0 if scale.shape[2] == 1 else g
                    result[b, n, g] += torch.matmul(left_block, right_block) * scale[b, n, scale_g_idx, start // PER_BLOCK_S2_SIZE, 0]
    return result

def chunked_matmul_with_dq_quant(left: torch.Tensor, right: torch.Tensor, scale_left, scale_right = torch.Tensor) -> torch.Tensor:
    B, N2, G1, S1, S2 = left.shape
    B_r, N2_r, G2, S2_r, D = right.shape
    G_out = max(G1, G2)
    result = torch.zeros((B, N2, G_out, S1, D), 
                        dtype=left.dtype,
                        device=left.device)
    s1_tail = PER_BLOCK_S1_SIZE_FOR_SCALE_DS if S1 % PER_BLOCK_S1_SIZE_FOR_SCALE_DS == 0 else S1 % PER_BLOCK_S1_SIZE_FOR_SCALE_DS
    for b in range(B):
        for n in range(N2):
            for g in range(G_out):
                g_left = 0 if G1 == 1 else g
                g_right = 0 if G2 == 1 else g
                for s2 in range(0, S2, PER_BLOCK_S2_SIZE):   
                    s2_start = s2
                    s2_end = s2_start + PER_BLOCK_S2_SIZE
                    s2_end = min(s2_end, S2)
                    scale_right_value = scale_right[b, n, 0, s2_start // PER_BLOCK_S2_SIZE, 0]
                    right_block = right[b, n, g_right, s2_start:s2_end, :]
                    for s1 in range(0, S1 - s1_tail, PER_BLOCK_S1_SIZE_FOR_SCALE_DS):
                        s1_start = s1
                        s1_end = s1_start + PER_BLOCK_S1_SIZE_FOR_SCALE_DS
                        s1_end = min(s1_end, S1)
                        left_block = left[b, n, g_left, s1_start:s1_end, s2_start:s2_end]
                        scale_left_value = scale_left
                        result[b, n, g, s1_start:s1_end, :] += torch.matmul(left_block, right_block) * scale_left_value * scale_right_value
                    # 处理尾块
                    s1_start = S1 - s1_tail
                    s1_end = S1
                    left_block = left[b, n, g_left, s1_start:s1_end, s2_start:s2_end]
                    scale_left_value = scale_left
                    result[b, n, g, s1_start:s1_end, :] += torch.matmul(left_block, right_block) * scale_left_value * scale_right_value
    return result

def chunked_matmul_with_dk_quant(left: torch.Tensor, right: torch.Tensor, scale_left, scale_right = torch.Tensor) -> torch.Tensor:
    B, N2, G1, S2, S1 = left.shape
    B_r, N2_r, G2, S1_r, D = right.shape
    G_out = max(G1, G2)
    result = torch.zeros((B, N2, G_out, S2, D), 
                        dtype=left.dtype,
                        device=left.device)
    s1_tail = PER_BLOCK_S1_SIZE_FOR_SCALE_DS if S1 % PER_BLOCK_S1_SIZE_FOR_SCALE_DS == 0 else S1 % PER_BLOCK_S1_SIZE_FOR_SCALE_DS
    for b in range(B):
        for n in range(N2):
            for g in range(G_out):
                g_left = 0 if G1 == 1 else g
                g_right = 0 if G2 == 1 else g
                for s2 in range(0, S2, PER_BLOCK_S2_SIZE):   
                    s2_start = s2
                    s2_end = s2_start + PER_BLOCK_S2_SIZE
                    s2_end = min(s2_end, S2)
                    scale_g_idx = 0
                    for s1 in range(0, S1 - s1_tail, PER_BLOCK_S1_SIZE_FOR_SCALE_DS):
                        s1_start = s1
                        s1_end = s1_start + PER_BLOCK_S1_SIZE_FOR_SCALE_DS
                        s1_end = min(s1_end, S1)
                        
                        left_block = left[b, n, g_left, s2_start:s2_end, s1_start:s1_end]
                        scale_left_value = scale_left
                        right_block = right[b, n, g_right, s1_start:s1_end, :]
                        scale_right_value = scale_right[b, n, scale_g_idx, s1_start // PER_BLOCK_S1_SIZE, 0]
                        result[b, n, g, s2_start:s2_end] += torch.matmul(left_block, right_block) * scale_left_value * scale_right_value
                    s1_start = S1 - s1_tail
                    s1_end = S1
                    left_block = left[b, n, g_left, s2_start:s2_end, s1_start:s1_end]
                    scale_left_value = scale_left
                    right_block = right[b, n, g_right, s1_start:s1_end, :]
                    scale_right_value = scale_right[b, n, scale_g_idx, s1_start // PER_BLOCK_S1_SIZE, 0]
                    result[b, n, g, s2_start:s2_end] += torch.matmul(left_block, right_block) * scale_left_value * scale_right_value
    return result

def block_quant_or_dequant_single(tensor, scale, block_size, is_scale_ds=False):
    dim1 = tensor.shape[0]
    dim2 = tensor.shape[1]
    dim3 = tensor.shape[2]
    dim4 = tensor.shape[3]
    dim5 = tensor.shape[4]
    quanted_tensor = torch.zeros([dim1, dim2, dim3, dim4, dim5])
    if not is_scale_ds:
        for b in range (dim1):
            for n in range (dim2):
                for g in range (dim3):
                    for s in range (0, dim4, block_size):
                        s_start = s // block_size * block_size
                        s_end = min(s // block_size * block_size + block_size, dim4)
                        scale_g_idx = 0 if scale.shape[2] == 1 else g
                        quanted_tensor[b, n, g, s_start : s_end, :] = tensor[b, n, g, s_start : s_end, :] * scale[b, n, scale_g_idx, s // block_size, 0]
    else :
        quanted_tensor = tensor * scale
    return quanted_tensor

def block_dequant(res, dequant_matrix_s1, dequant_matrix_s2):
    B, Nkv, G, S1, S2 = res.shape
    result = np.zeros_like(res)
    for b in range(B):
        for n in range(Nkv):
            for g in range(G):
                for i in range((S1 + PER_BLOCK_S1_SIZE - 1) // PER_BLOCK_S1_SIZE):
                    start_s1 = i * PER_BLOCK_S1_SIZE
                    end_s1 = min(start_s1 + PER_BLOCK_S1_SIZE, S1)
                    for j in range((S2 + PER_BLOCK_S2_SIZE - 1) // PER_BLOCK_S2_SIZE):
                        start_s2 = j * PER_BLOCK_S2_SIZE
                        end_s2 = min(start_s2 + PER_BLOCK_S2_SIZE, S2)
                        block = res[b, n, g, start_s1:end_s1, start_s2:end_s2]
                        s1_g_idx = 0 if dequant_matrix_s1.shape[2] == 1 else g
                        s2_g_idx = 0 if dequant_matrix_s2.shape[2] == 1 else g
                        dequant_value_s1 = dequant_matrix_s1[b, n, s1_g_idx, i, 0]
                        dequant_value_s2 = dequant_matrix_s2[b, n, s2_g_idx, j, 0]
                        dequant_kv = dequant_value_s1 * dequant_value_s2
                        weighted_block = block * dequant_kv
                        result[b, n, g, start_s1:end_s1, start_s2:end_s2] = weighted_block
    return torch.from_numpy(result)

def tforward(q, k, v, scale, dscale_q, dscale_k, dscale_v):
    qk = None
    scale_q = 1 / dscale_q
    scale_k = 1 / dscale_k
    scale_v = 1 / dscale_v

    q = block_quant_or_dequant_single(q, scale_q, PER_BLOCK_S1_SIZE)
    k = block_quant_or_dequant_single(k, scale_k, PER_BLOCK_S2_SIZE)
    v = block_quant_or_dequant_single(v, scale_v, PER_BLOCK_S2_SIZE)
    q = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(q.cpu().numpy())))
    k = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(k.cpu().numpy())))
    v = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(v.cpu().numpy())))
    qkk = torch.matmul(q, k.permute(0, 1, 2, 4, 3))
    qk = block_dequant(qkk, dscale_q, dscale_k).mul(scale)
    softmax_res, x_max, x_sum = tsoftmax(qk)
    drop_res = softmax_res
    y = chunked_matmul_with_quant(drop_res, v, dscale_v)
    return y, softmax_res, x_max, x_sum

def tbackward(dx, q, k, v, softmax_res, y, scale, dscale_q, dscale_k, dscale_v, dscale_dx, ds_scale, p_scale):
    scale_q = 1 / dscale_q
    scale_k = 1 / dscale_k
    scale_v = 1 / dscale_v
    scale_dx = 1 / dscale_dx

    dscale_p = 1 / p_scale
    dscale_ds = 1 / ds_scale
    q = block_quant_or_dequant_single(q, scale_q, PER_BLOCK_S1_SIZE)
    k = block_quant_or_dequant_single(k, scale_k, PER_BLOCK_S2_SIZE)
    v = block_quant_or_dequant_single(v, scale_v, PER_BLOCK_S2_SIZE)
    dx = block_quant_or_dequant_single(dx, scale_dx, PER_BLOCK_S1_SIZE)
    softmax_res_quant = block_quant_or_dequant_single(softmax_res, p_scale, PER_BLOCK_S2_SIZE, True)

    q = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(q.cpu().numpy())))
    k = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(k.cpu().numpy())))
    v = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(v.cpu().numpy())))

    dx = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(dx.cpu().numpy())))
    softmax_res_quant = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(softmax_res_quant.cpu().numpy())))
    
    drop_res = softmax_res_quant
    dv = chunked_matmul_with_dv_quant(drop_res.permute(0, 1, 2, 4, 3), dx, dscale_p, dscale_dx)

    dpp = torch.matmul(dx, v.permute(0, 1, 2, 4, 3))
    dp = block_dequant(dpp, dscale_dx, dscale_v)
    dp_drop = dp
    softmax_grad_res = (y * block_quant_or_dequant_single(dx, dscale_dx, PER_BLOCK_S1_SIZE)).sum(dim=-1, keepdim=True)

    softmax_grad_res = (dp_drop - softmax_grad_res) * softmax_res

    q_scale_ds = ds_scale
    softmax_grad_res_quanted = block_quant_or_dequant_single(softmax_grad_res, q_scale_ds, PER_BLOCK_S2_SIZE, True)
    softmax_grad_res_fp8 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(trans_np_float_tensor_to_hifuint8(softmax_grad_res_quanted.cpu().numpy())))

    deq_scale_ds = dscale_ds
    dq = chunked_matmul_with_dq_quant(softmax_grad_res_fp8, k, deq_scale_ds, dscale_k) * scale
    dk = chunked_matmul_with_dk_quant(softmax_grad_res_fp8.permute(0, 1, 2, 4, 3), q, deq_scale_ds, dscale_q) * scale
    dk = torch.sum(dk, dim=2, keepdim=True)
    dv = torch.sum(dv, dim=2, keepdim=True)
    return dq, dk, dv

def trans_np_float_tensor_to_hifuint8(in_tensor):
    import en_dtypes
    from en_dtypes import hifloat8
    if isinstance(in_tensor, torch.Tensor):
        in_tensor = in_tensor.cpu().numpy()

    out_tensor = in_tensor.astype(hifloat8).view(np.uint8)
    return out_tensor

def trans_np_hifuint8_tensor_to_float32(in_tensor):
    import en_dtypes
    from en_dtypes import hifloat8
    if isinstance(in_tensor, torch.Tensor):
        in_tensor = in_tensor.cpu().numpy()
    
    if in_tensor.dtype == np.uint8:
        in_tensor = in_tensor.view(hifloat8)

    out_tensor = in_tensor.astype(np.float32)
    return out_tensor

def npu_block_quant(tensor, scale = 1,PER_BLOCK_SIZE = 512):
    dim1 = tensor.shape[0]
    dim2 = tensor.shape[1]
    dim3 = tensor.shape[2]
    dim4 = tensor.shape[3]
    quanted_tensor = torch.zeros([dim1, dim2, dim3, dim4]).to(torch.float32)
    for b in range(dim1):
        for n in range(dim3):
            for s in range (0,dim2,PER_BLOCK_SIZE):
                s_start = s // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                s_end = min (s // PER_BLOCK_SIZE * PER_BLOCK_SIZE + PER_BLOCK_SIZE, dim2)
                quanted_tensor[b, s_start:s_end, n, :] = tensor[b, s_start:s_end, n, :] * scale
    return quanted_tensor

class TestNPUQuantFlashAttentionV2(TestCase):
    def supported_op_exec(self, q, k, v, dx, scale_q, scale_k, scale_v, scale_dx, ds_scale, p_scale):
        scale = 0.08838
        out, softmax_res, x_max, x_sum = tforward(q.cpu().to(gtype), k.cpu().to(gtype), v.cpu().to(gtype), \
                                                    scale, scale_q, scale_k, scale_v)

        dq_golden, dk_golden, dv_golden = tbackward(dx.cpu().to(gtype), q.cpu().to(gtype), k.cpu().to(gtype), \
                                                    v.cpu().to(gtype),softmax_res.cpu().to(gtype), out.cpu().to(gtype), \
                                                    scale, scale_q, scale_k, scale_v, scale_dx, ds_scale, p_scale)
        return dq_golden, dk_golden, dv_golden, out, x_max, x_sum

    @SupportedDevices(['Ascend950'])
    def test_npu_flash_attention_v2(self, device="npu"):
        query = torch.randn(1, 7200, 40, 128, dtype=torch.float32)
        key = torch.randn(1, 512, 40, 128, dtype=torch.float32)
        value = torch.randn(1, 512, 40, 128, dtype=torch.float32)
        dy = torch.randn(1, 7200, 40, 128, dtype=torch.float32)
        x_max = torch.randn(1, 40, 7200, 1, dtype=torch.float32)
        x_sum = torch.randn(1, 40, 7200, 1, dtype=torch.float32)
        attention_in = torch.randn(1, 7200, 40, 128, dtype=torch.bfloat16)
        dscale_q = torch.ones(1, 40, 15, 1, dtype=torch.float32)
        dscale_k = torch.ones(1, 40, 1, 1, dtype=torch.float32)
        dscale_v = torch.ones(1, 40, 1, 1, dtype=torch.float32)
        dscale_dy = torch.ones(1, 40, 15, 1, dtype=torch.float32)
        query_dtype=torch_npu.hifloat8
        head_num = 40
        scale = 0.08838
        cpu_q = query.unsqueeze(0)
        cpu_k = key.unsqueeze(0)
        cpu_v = value.unsqueeze(0)
        cpu_dy = dy.unsqueeze(0)
        cpu_dscale_q = dscale_q.unsqueeze(0)
        cpu_dscale_k = dscale_k.unsqueeze(0)
        cpu_dscale_v = dscale_v.unsqueeze(0)
        cpu_dscale_dy = dscale_dy.unsqueeze(0)

        q = cpu_q.permute(0,1,3,2,4)
        k = cpu_k.permute(0,1,3,2,4)
        v = cpu_v.permute(0,1,3,2,4)
        dx = cpu_dy.permute(0,1,3,2,4)

        query_hifp8 = npu_block_quant(query)
        key_hifp8 = npu_block_quant(key)
        value_hifp8 = npu_block_quant(value)
        dy_hifp8 = npu_block_quant(dy)

        query_hifp8 = trans_np_float_tensor_to_hifuint8(query_hifp8.cpu().numpy())
        key_hifp8 = trans_np_float_tensor_to_hifuint8(key_hifp8.cpu().numpy())
        value_hifp8 = trans_np_float_tensor_to_hifuint8(value_hifp8.cpu().numpy())
        dy_hifp8 = trans_np_float_tensor_to_hifuint8(dy_hifp8.cpu().numpy())
        
        query_hifp8 = torch.from_numpy(query_hifp8)
        key_hifp8 = torch.from_numpy(key_hifp8)
        value_hifp8 = torch.from_numpy(value_hifp8)
        dy_hifp8 = torch.from_numpy(dy_hifp8)

        p_scale = torch.tensor([15]).float()
        ds_scale = torch.tensor([3.5]).float()
        cpu_result = self.supported_op_exec(q, k, v, dx, cpu_dscale_q, cpu_dscale_k, cpu_dscale_v, cpu_dscale_dy, p_scale, ds_scale)
        attention_in = cpu_result[3]
        x_max = cpu_result[4]
        x_sum = cpu_result[5]
        attention_in = attention_in.permute(0,1,3,2,4).squeeze(0).to(torch.bfloat16)
        x_max = x_max.squeeze(0)
        x_sum = x_sum.squeeze(0)

        npu_result = torch_npu.npu_quant_fusion_attention_backward(query_hifp8.npu(), key_hifp8.npu(), value_hifp8.npu(), dy_hifp8.npu(), head_num,
                                                                "BSND",
                                                                dscale_q.npu(),
                                                                dscale_k.npu(),
                                                                dscale_v.npu(),
                                                                dscale_dy.npu(),
                                                                p_scale=p_scale.npu(),
                                                                ds_scale=ds_scale.npu(),
                                                                softmax_max=x_max.npu(),
                                                                softmax_sum=x_sum.npu(),
                                                                attention_in=attention_in.npu(),
                                                                scale_value=scale,
                                                                query_dtype=query_dtype)
        dq_cpu = cpu_result[0]
        dq = npu_result[0]
        dq_cpu = dq_cpu.permute(0,1,3,2,4).squeeze(0)
        self.assertRtolEqual(dq_cpu, dq.to(torch.float32), prec=0.01, prec16=0.01)
if __name__ == "__main__":
    run_tests()

