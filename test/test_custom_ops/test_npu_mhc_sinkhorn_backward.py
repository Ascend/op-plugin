import torch
import torch_npu
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class SplitInfo:
    def __init__(self, t_main, t_tail, used_core_num, ubfactor):
        self.t_main = t_main
        self.t_tail = t_tail
        self.used_core_num = used_core_num
        self.ubfactor = ubfactor


class TestNpuMhcSinkhornBackward(TestCase):
    DATA_BLOCK_NUM = 8
    SIZEOF_FLOAT = 4
    VF_T_FACTOR = 8
    DOUBLE_BUFFER = 2
    MASK_BUFFER = 64
    MAX_BUFFER_FORWARD = 256 * DOUBLE_BUFFER
    MAX_BUFFER_BACKWARD = 256 * DOUBLE_BUFFER

    def do_op_tiling(self, T, n, n_align):
        def ceil_align(val):
            return (val + self.DATA_BLOCK_NUM - 1) // self.DATA_BLOCK_NUM * self.DATA_BLOCK_NUM
        def backward_cal_occupy_size(ubfactor):
            inque_size = ceil_align(ubfactor * n * n)
            outque_size = ceil_align(ubfactor * n * n)
            norm_que_size = ceil_align(2 * ubfactor * n * n_align)
            sum_que_size = ceil_align(2 * ubfactor * n_align)
            return inque_size + outque_size + norm_que_size + sum_que_size

        def forward_cal_occupy_size(ubfactor):
            inque_size = ceil_align(ubfactor * n * n)
            outque_size = ceil_align(ubfactor * n * n)
            norm_que_size = ceil_align(n * ubfactor * n_align)
            sum_col_size = ceil_align(ubfactor * n_align)
            sum_row_size = ceil_align(ubfactor * n_align)
            return inque_size + outque_size + 2 * norm_que_size + sum_col_size + sum_row_size
        
        def backward_split_core(ub_size, total_core, T):
            aviable_ub_size = (ub_size - self.MASK_BUFFER - self.MAX_BUFFER_BACKWARD) // self.DOUBLE_BUFFER
            t_main = (T + total_core - 1) // total_core
            used_core_num = (T + t_main - 1) // t_main
            t_tail = T - t_main * (used_core_num - 1)
            ubfactor = t_main
            occupy_size = backward_cal_occupy_size(ubfactor) * self.SIZEOF_FLOAT
            if occupy_size > aviable_ub_size:
                one_pice_size = backward_cal_occupy_size(1) * self.SIZEOF_FLOAT
                ubfactor = aviable_ub_size // one_pice_size
                ubfactor = ubfactor if ubfactor // self.VF_T_FACTOR == 0 else ubfactor // self.VF_T_FACTOR * self.VF_T_FACTOR
            return SplitInfo(t_main, t_tail, used_core_num, ubfactor)

        def forward_split_core(ub_size, total_core, T):
            aviable_ub_size = (ub_size - self.MASK_BUFFER - self.MAX_BUFFER_FORWARD) // self.DOUBLE_BUFFER
            t_main = (T + total_core - 1) // total_core
            used_core_num = (T + t_main - 1) // t_main
            t_tail = T - t_main * (used_core_num - 1)
            ubfactor = t_main
            occupy_size = forward_cal_occupy_size(ubfactor) * self.SIZEOF_FLOAT
            if occupy_size > aviable_ub_size:
                one_pice_size = forward_cal_occupy_size(1) * self.SIZEOF_FLOAT
                ubfactor = aviable_ub_size // one_pice_size
            return SplitInfo(t_main, t_tail, used_core_num, ubfactor)
        
        ub_size = 248 * 1024
        total_core = 64
        backward_split_info = backward_split_core(ub_size, total_core, T)
        forward_split_info = forward_split_core(ub_size, total_core, T)
        output_split_info = forward_split_info if forward_split_info.ubfactor <= backward_split_info.ubfactor else backward_split_info
        t_main = output_split_info.t_main
        t_tail = output_split_info.t_tail
        used_core_num = output_split_info.used_core_num
        ubfactor = output_split_info.ubfactor
        return t_main, t_tail, used_core_num, ubfactor, n, n_align

    def proc_norm_single_core(self, norm_cache, blockIdx, tiling_info):
        t_main, t_tail, used_core_num, ubfactor, *rest = tiling_info
        t_num = t_tail if blockIdx == (used_core_num - 1) else t_main

        # 取出每个核要处理的内存
        norm_cache_single_core = []
        for iter_norm in norm_cache:
            start = blockIdx * t_main
            norm_cache_single_core.append(iter_norm[start:start + t_num])

        loop = (t_num + ubfactor - 1) // ubfactor
        main_num = ubfactor * (loop - 1)
        tail_num = t_num - main_num
        
        iter_main_cache = []
        iter_tail_cache = []
        for norm in norm_cache_single_core:
            if loop > 1:
                norm_main = norm[:main_num]
                main_reshape = norm_main.view(loop - 1, ubfactor, *list(norm.shape[-2:]))
                main_loop = main_reshape.permute(0,2,1,3) # [loop - 1, 4, ubfactor, 8]
                iter_main_cache.append(main_loop)

            norm_tail = norm[main_num:]
            tail_reshape = norm_tail.view(1, tail_num, *list(norm.shape[-2:]))
            tail_loop = tail_reshape.permute(0,2,1,3) # [1, 4, ubfactor, 8]
            iter_tail_cache.append(tail_loop)

        if loop > 1:
            main_layout = torch.cat(iter_main_cache, dim=1)
            tail_layout = torch.cat(iter_tail_cache, dim=1)
            main_layout = main_layout.reshape(-1, main_layout.shape[-1])
            tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
            norm_out_single_core = torch.cat((main_layout, tail_layout), dim=0)
        else:
            tail_layout = torch.cat(iter_tail_cache, dim=1)
            tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
            norm_out_single_core = tail_layout

        return norm_out_single_core

    def proc_sum_single_core(self, sum_col_cache, sum_row_cache, blockIdx, tiling_info):
        t_main, t_tail, used_core_num, ubfactor, n_, n_align = tiling_info
        t_num = t_tail if blockIdx == (used_core_num - 1) else t_main

        # 取出每个核要处理的内存
        sum_col_cache_single_core = []
        sum_row_cache_single_core = []
        for iter_sum in sum_col_cache:
            start = blockIdx * t_main
            sum_col_cache_single_core.append(iter_sum[start:start + t_num]) # col尾轴已经按8对齐pad
        for iter_sum in sum_row_cache:
            start = blockIdx * t_main
            sum_row_cache_single_core.append(iter_sum[start:start + t_num])

        loop = (t_num + ubfactor - 1) // ubfactor
        main_num = ubfactor * (loop - 1)
        tail_num = t_num - main_num
        
        iter_main_cache = []
        iter_tail_cache = []
        for i in range(len(sum_col_cache_single_core)):
            sum_col = sum_col_cache_single_core[i]
            sum_row = sum_row_cache_single_core[i]

            if loop > 1:
                # mainloop处理
                sum_col_main = sum_col[:main_num]
                sum_row_main = sum_row[:main_num]
                col_main_reshape = sum_col_main.view(loop - 1, ubfactor, n_align) # [1, 2, 8]
                row_main_reshape = sum_row_main.view(loop - 1, ubfactor, n_).permute(0,2,1) # [1, 4, 2]
                # 在1轴上n0*t_对齐为n0_align*t_
                row_main_reshape = F.pad(row_main_reshape, (0, 0, 0, n_align - n_), mode='constant', value=1) # [1, 8, 2]
                # reshape之后cat
                col_ = col_main_reshape.reshape(loop - 1, -1) # [2, 2 * 8]
                row_ = row_main_reshape.reshape(loop - 1, -1) # [2, 8 * 2]
                main_loop = torch.cat((row_, col_), dim=1) # [2, 32]
                iter_main_cache.append(main_loop)
                
            # tailloop处理
            sum_col_tail = sum_col[main_num:]
            sum_row_tail = sum_row[main_num:]
            col_tail_reshape = sum_col_tail.view(1, tail_num, n_align) # [1, 1, 8]
            row_tail_reshape = sum_row_tail.view(1, tail_num, n_).permute(0,2,1) # [1, 4, 1]
            row_tail_reshape = F.pad(row_tail_reshape, (0, 0, 0, n_align - n_), mode='constant', value=1) # [1, 8, 1]
            # reshape之后cat
            col_ = col_tail_reshape.reshape(1, -1) # [1, 1 * 8]
            row_ = row_tail_reshape.reshape(1, -1) # [1, 8 * 1]
            tail_loop = torch.cat((row_, col_), dim=1) # [2, 8]
            iter_tail_cache.append(tail_loop)
        
        if loop > 1:
            main_layout = torch.cat(iter_main_cache, dim=1) # [40, 16]
            tail_layout = torch.cat(iter_tail_cache, dim=1) # [40, 8]
            main_layout = main_layout.reshape(-1)
            tail_layout = tail_layout.reshape(-1)
            sum_out = torch.cat((main_layout, tail_layout), dim=0)
        else:
            tail_layout = torch.cat(iter_tail_cache, dim=1)
            tail_layout = tail_layout.reshape(-1, tail_layout.shape[-1])
            sum_out = tail_layout

        return sum_out.reshape(-1)

    def proc_norm(self, norm_cache, sum_col_cache, sum_row_cache, tiling_info):
        t_main, t_tail, used_core_num, ubfactor, *rest = tiling_info

        norm_out_cache = []
        sum_out_cache = []

        for blockIdx in range(used_core_num):
            norm_on_core = self.proc_norm_single_core(norm_cache, blockIdx, tiling_info)
            sum_on_core = self.proc_sum_single_core(sum_col_cache, sum_row_cache, blockIdx, tiling_info)

            norm_out_cache.append(norm_on_core)
            sum_out_cache.append(sum_on_core)

        norm_out = torch.cat(norm_out_cache, dim=0)
        sum_out = torch.cat(sum_out_cache, dim=0)

        return norm_out, sum_out

    def softmax(self, x, dim=-1):
        """计算softmax"""
        return torch.softmax(x, dim=dim)

    def sinkhorn_forward_simple(self, x, num_iters, eps):
        """
        Sinkhorn前向传播（简化版）

        参数:
            x: 输入张量 [T, n, n]
            num_iters: 迭代次数
            eps: 小常数防止除零

        返回:
            output: 输出 [T, n, n]
            norm_list: 归一化输出列表
            sum_list: 求和结果列表
        """
        norm_list = []
        sum_list = []

        # Step 0: Softmax
        prob = self.softmax(x, dim=-1)
        curr = prob + eps
        norm_list.append(prob.clone())
        sum_list.append(None)

        # Step 1: Initial Col Norm
        col_sum = curr.sum(dim=-2, keepdim=True) + eps
        curr = curr / col_sum
        norm_list.append(curr.clone())
        sum_list.append(col_sum.clone())

        # Step 2: Loop
        for i in range(num_iters - 1):
            # Row Norm
            row_sum = curr.sum(dim=-1, keepdim=True) + eps
            curr = curr / row_sum
            norm_list.append(curr.clone())
            sum_list.append(row_sum.clone())

            # Col Norm
            col_sum = curr.sum(dim=-2, keepdim=True) + eps
            curr = curr / col_sum
            norm_list.append(curr.clone())
            sum_list.append(col_sum.clone())

        return curr, norm_list, sum_list

    def sinkhorn_backward_golden(self, grad_output, norm_out, sum_out, num_iters, bsnn_flag):
        if bsnn_flag == 1:
            B, S, n0, n1 = grad_output.shape
            grad_output = grad_output.reshape(-1, *grad_output.shape[-2:])
        # 将grad_output从[T, n, n]转换为[n, n, T]以匹配norm_out的格式
        grad_curr = grad_output.permute(1, 2, 0)  # [T, n, n] -> [n, n, T]
        # 共进行num_iters次迭代
        for i in range(num_iters - 1, 0, -1):
            # 列归一化的反向传播
            col_idx = 2 * i + 1
            # grad_curr: [n, n, T], norm_out[col_idx]: [n, n, T]
            # 列归一化是在axis=0方向求和（第一个n维度）
            dot_prod = torch.sum(grad_curr * norm_out[col_idx], dim=0, keepdim=True)  # [1, n, T]
            # sum_out[col_idx]: [n, T], 需要扩展为 [1, n, T]
            grad_curr = (grad_curr - dot_prod) / sum_out[col_idx].unsqueeze(0)

            # 行归一化的反向传播
            row_idx = 2 * i
            # grad_curr: [n, n, T], norm_out[row_idx]: [n, n, T]
            # 行归一化是在axis=1方向求和（第二个n维度）
            dot_prod = torch.sum(grad_curr * norm_out[row_idx], dim=1, keepdim=True)  # [n, 1, T]
            grad_curr = (grad_curr - dot_prod) / sum_out[row_idx].unsqueeze(1)


        # 最后一次迭代：列归一化
        dot_prod = torch.sum(grad_curr * norm_out[1], dim=0, keepdim=True)  # [1, n, T]
        grad_curr = (grad_curr - dot_prod) / sum_out[1].unsqueeze(0)
        
        dot_prod = torch.sum(grad_curr * norm_out[0], dim=1, keepdim=True)  # [n, 1, T]
        grad_input = norm_out[0] * (grad_curr - dot_prod)  # [n, n, T]
        # 将结果从[n, n, T]转回[T, n, n]
        grad_input = grad_input.permute(2, 0, 1).contiguous()  # [n, n, T] -> [T, n, n]
        if bsnn_flag == 1:
            grad_input = grad_input.view(B, S, n0, n1)
        return grad_input

    def convert_to_npu_format(self, norm_list, sum_list, num_iters):
        """
        将列表格式的中间结果转换为NPU格式

        参数:
            norm_list: 归一化输出列表，每个元素shape为[T, n, n]
            sum_list: 求和结果列表，每个元素shape为[T, 1, n]或[T, n, 1]
            num_iters: 迭代次数

        返回:
            norm_out: [2*num_iters, n, n, T]
            sum_out: [2*num_iters, n, T]
        """
        T, n, _ = norm_list[0].shape

        # 构建norm_out: [2*num_iters, n, n, T]
        norm_out = torch.zeros(2 * num_iters, n, n, T, dtype=norm_list[0].dtype, device=norm_list[0].device)
        for idx, norm_tensor in enumerate(norm_list):
            # [T, n, n] -> [n, n, T]
            norm_out[idx] = norm_tensor.permute(1, 2, 0)

        # 构建sum_out: [2*num_iters, n, T]
        sum_out = torch.zeros(2 * num_iters, n, T, dtype=norm_list[0].dtype, device=norm_list[0].device)
        for idx, sum_tensor in enumerate(sum_list):
            if sum_tensor is None:
                continue
            # [T, 1, n] or [T, n, 1] -> [n, T]
            if sum_tensor.shape[-2] == 1:  # col_sum: [T, 1, n]
                sum_out[idx] = sum_tensor.squeeze(-2).T
            else:  # row_sum: [T, n, 1]
                sum_out[idx] = sum_tensor.squeeze(-1).T

        return norm_out, sum_out

    # 手动反向传播
    def cpu_op_exec(self, grad_output, norm_list_golden, sum_list_golden, num_iters, bsnn_flag):
        norm_out_golden, sum_out_golden = self.convert_to_npu_format(norm_list_golden, sum_list_golden, num_iters)
        grad_input_golden = self.sinkhorn_backward_golden(grad_output, norm_out_golden, sum_out_golden, num_iters, bsnn_flag)        
        return grad_input_golden

    def npu_op_exec(self, grad_output, norm_out, sum_out):
        return torch_npu.npu_mhc_sinkhorn_backward(
            grad_output.npu(), norm_out.npu(), sum_out.npu()
        )

    def build_input_tensors(self, tensor_x, num_iters=20):
        bsnn_flag = 0
        if tensor_x.dim() == 4:
            B, S, n0, n1 = tensor_x.shape
            bsnn_flag = 1
            tensor_x = tensor_x.reshape(-1, *tensor_x.shape[-2:])
            T = torch.numel(tensor_x) // (n0 * n1)
        else:
            T, n0, n1 = tensor_x.shape
        x = tensor_x.numpy()
        eps = 1e-6
        eps = float(eps)
        out_flag = 1
        n_align = 8
        PAD_VALE = 0
        def pad_norm(norm):
            norm_pad = F.pad(norm, (0, n_align - n1), mode='constant', value=PAD_VALE)
            norm_reshape = norm_pad.reshape(-1, *norm_pad.shape[-2:])
            return norm_reshape

        def pad_sum(sum_col):
            sum_pad = F.pad(sum_col, (0, n_align - n1), mode='constant', value=PAD_VALE)
            sum_reshape = sum_pad.reshape(-1, *sum_pad.shape[-1:])
            return sum_reshape
        
        tiling_info = []
        if out_flag != 0:
            tiling_info = self.do_op_tiling(T, n0, n_align)

        norm_cache = []
        sum_col_cache = []
        sum_row_cache = []
        # Sinkhorn-Knopp算法功能
        prob = torch.softmax(tensor_x, dim=-1)
        h_comb_softmax = prob + eps
        sum_col = h_comb_softmax.sum(dim=-2, keepdim=True) + eps
        h_comb = h_comb_softmax / sum_col
        
        norm_cache.append(pad_norm(prob))
        norm_cache.append(pad_norm(h_comb))
        sum_row_cache.append(torch.ones(T, n0))
        sum_col_cache.append(pad_sum(sum_col))
        
        for i in range(max(num_iters - 1, 0)):
            # Row Norm
            sum_row = (h_comb.sum(dim=-1, keepdim=True) + eps)
            h_comb = h_comb / sum_row
            norm_cache.append(pad_norm(h_comb))
            sum_row_cache.append(sum_row.reshape(-1, *sum_row.shape[-2:]))

            # Col Norm
            sum_col = (h_comb.sum(dim=-2, keepdim=True) + eps)
            h_comb = h_comb / sum_col
            norm_cache.append(pad_norm(h_comb))
            sum_col_cache.append(pad_sum(sum_col))

        if out_flag != 0:                    
            norm_out, sum_out = self.proc_norm(norm_cache, sum_col_cache, sum_row_cache, tiling_info)
            norm_out = norm_out.reshape(-1)
            sum_out = sum_out.reshape(-1)
        
        y, norm_list_golden, sum_list_golden = self.sinkhorn_forward_simple(tensor_x, num_iters, eps)
        # 构造输出梯度（假设loss = sum(y^2)）
        if bsnn_flag == 1:
            grad_y = 2 * (y.view(B, S, n0, n1))
        else:
            grad_y = 2 * y

        return (grad_y, norm_list_golden, sum_list_golden, num_iters, norm_out, sum_out, bsnn_flag)

    def run_and_check(self, tensor_x, num_iters):
        with torch.no_grad():
            (grad_output, norm_list_golden, sum_list_golden, num_iters, norm_out, sum_out, bsnn_flag) = self.build_input_tensors(tensor_x, num_iters)

            expected_output = self.cpu_op_exec(grad_output, norm_list_golden, sum_list_golden, num_iters, bsnn_flag)

            actual_output = self.npu_op_exec(grad_output, norm_out, sum_out)

        self.assertRtolEqual(expected_output.float().numpy(), actual_output.cpu().numpy(), prec=1e-4)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_sinkhorn_backward_with_tnn_case1(self, device="npu"):
        tensor_x = torch.randn(1024, 4, 4, dtype=torch.float32)
        self.run_and_check(tensor_x, num_iters=20)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_sinkhorn_backward_with_tnn_case2(self, device="npu"):
        tensor_x = torch.randn(2048, 6, 6, dtype=torch.float32)
        self.run_and_check(tensor_x, num_iters=20)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_sinkhorn_backward_with_tnn_case3(self, device="npu"):
        tensor_x = torch.randn(4096, 8, 8, dtype=torch.float32)
        self.run_and_check(tensor_x, num_iters=20)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_sinkhorn_backward_with_bsnn(self, device="npu"):
        tensor_x = torch.randn(8, 128, 4, 4, dtype=torch.float32)
        self.run_and_check(tensor_x, num_iters=20)


if __name__ == "__main__":
    run_tests()
