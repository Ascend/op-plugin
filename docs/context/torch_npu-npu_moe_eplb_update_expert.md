# torch\_npu.npu\_moe\_eplb\_update\_expert<a name="ZH-CN_TOPIC_0000002350725100"></a>

## 产品支持情况<a name="zh-cn_topic_0000002366611733_section8593133131718"></a>

<a name="zh-cn_topic_0000002366611733_table1659316316174"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002366611733_row2059343171716"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002366611733_p125930301711"><a name="zh-cn_topic_0000002366611733_p125930301711"></a><a name="zh-cn_topic_0000002366611733_p125930301711"></a><span id="zh-cn_topic_0000002366611733_ph12593183191719"><a name="zh-cn_topic_0000002366611733_ph12593183191719"></a><a name="zh-cn_topic_0000002366611733_ph12593183191719"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002366611733_p18593639173"><a name="zh-cn_topic_0000002366611733_p18593639173"></a><a name="zh-cn_topic_0000002366611733_p18593639173"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002366611733_row294304412306"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002366611733_p49437440302"><a name="zh-cn_topic_0000002366611733_p49437440302"></a><a name="zh-cn_topic_0000002366611733_p49437440302"></a><span id="zh-cn_topic_0000002366611733_ph19280164145411"><a name="zh-cn_topic_0000002366611733_ph19280164145411"></a><a name="zh-cn_topic_0000002366611733_ph19280164145411"></a><term id="zh-cn_topic_0000002366611733_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002366611733_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002366611733_zh-cn_topic_0000001312391781_term1253731311225"></a><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002366611733_p8877121915317"><a name="zh-cn_topic_0000002366611733_p8877121915317"></a><a name="zh-cn_topic_0000002366611733_p8877121915317"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002366611733_section14441124184110"></a>

-   算子功能：为了解决负载不均衡的场景，MoE网络中常用EPLB（Expert Parallelism Load Balancer）算法进行冗余专家部署，一个逻辑专家在多个卡上都有实例部署（即有多个物理专家），在这种场景下，MoeEPLBUpdateExpert算子可以完成每个token的topK个专家逻辑专家号到物理专家实例号的映射。
-   计算公式：

    对于expert\_ids中的第i个值，即第i个token：

    ![](figures/zh-cn_formulaimage_0000002350811180.png)

    -   当eplb\_table\[tableOffset\]=1时

        ![](figures/zh-cn_formulaimage_0000002384541317.png)

    -   当eplb\_table\[tableOffset\]\>1时

        ![](figures/zh-cn_formulaimage_0000002350982146.png)

## 函数原型<a name="zh-cn_topic_0000002366611733_section45077510411"></a>

```
torch_npu.npu_moe_eplb_update_expert(Tensor expert_ids, Tensor eplb_table, int local_rank_id, int world_size, *, int balance_mode=0) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000002366611733_section112637109429"></a>

-   **expert\_ids**（`Tensor`）：必选参数，表示每个token的topK个专家索引，数据类型支持`int32`、`int64`。数据格式支持ND。维度支持2维，shape为\(Bs, K\)，支持非连续的Tensor。
-   **eplb\_table**（`Tensor`）：必选参数，表示逻辑专家到物理专家实例的映射表，请保证输入Tensor的值正确。`world_size`张卡，每张卡部署moe\_expert\_per\_rank个路由专家，一共有`world_size`\*moe\_expert\_per\_rank个实例。`eplb_table`的每行对应一个逻辑moe专家的部署策略，第一列为该逻辑专家部署的实例数count，值需大于等于1；每行第\[1, count\]列为对应的实例编号，取值范围为\[0, `world_size`\*moe\_expert\_per\_rank\)，有效的实例编号不可以重复。数据类型支持`int32`。数据格式支持ND。维度支持2维，shape为\(moe\_expert\_num, F\)，支持非连续的Tensor。
-   **local\_rank\_id**（`int`）：必选参数，表示本卡Id。数据类型支持`int64`。取值范围为\[0, `world_size`\)。同一个通信域中各卡的local\_rank\_id不重复。
-   **world\_size**（`int`）：必选参数，表示通信域Size。取值范围为\[2, 384\]。
-   **balance\_mode**（`int`）：可选参数，**预留参数，暂未使用，使用默认值即可**。

## 返回值说明<a name="zh-cn_topic_0000002366611733_section22231435517"></a>

**balanced\_expert\_ids**（`Tensor`）：表示映射后每个token的topK个专家物理专家的实例编号，要求是一个2D的Tensor，shape为\(Bs, K\)，数据类型、数据格式与`expert_ids`保持一致，不支持非连续的Tensor。

## 约束说明<a name="zh-cn_topic_0000002366611733_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   参数里Shape使用的变量如下：
    -   BS：表示batch sequence size，即本卡最终输出的token数量，取值范围为0<BS≤512。
    -   K：表示选取topK个专家，取值范围为0<K≤16同时满足0<K≤moe\_expert\_num。
    -   moe\_expert\_num：表示MoE专家数，取值范围\(0, 512\]。
    -   F：表示输入映射表的列数，第一列为各行号对应MoE专家部署的实例个数（取值\>0），后F-1列为该MoE专家部署的实例编号，取值范围\[0, world\_size\*moe\_expert\_per\_rank\)。

## 调用示例<a name="zh-cn_topic_0000002366611733_section14459801435"></a>

-   单算子模式调用

    ```python
    import math
    import os
    import time
    import psutil
    import torch
    import numpy as np
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    DATA_PATH='<path_of_data>'
    
    
    class MoeEPLBUpdateExpert():
        def __init__(self, BS:int, K:int, log_ep_size:int, pyh_ep_size:int, F:int, world_size:int, rank_id:int, dtype):
            self.BS = BS
            self.K = K
            self.log_ep_size = log_ep_size
            self.pyh_ep_size = pyh_ep_size
            self.F = F
            self.rank_id = rank_id
            self.dtype = dtype
            self.expert_ids = None
            self.eplb_table = None
            self.expert_ids_tensor = None
            self.eplb_table_tensor = None
            self.world_size = world_size
            self.balance_mode = 0
            self.cpu_golden = None
            self.npu_output = None
            self.gen_input()
            self.exec_cpu()
    
        def get_case_key(self):
            return '{}BS_{}K_{}LogE_{}phyE_{}F_{}r_{}'.format(self.BS, self.K, self.log_ep_size, self.pyh_ep_size, self.F, self.world_size, 'int64' if self.dtype == torch.int64 else 'int32')
    
        def get_data_bin_dir(self):
            return '{}/{}'.format(DATA_PATH, self.get_case_key())
    
        def load_input_cache(self):
            folder_path=self.get_data_bin_dir()
            if not os.path.exists(folder_path):
                return False
            bin_path='{}/input_{}r.bin.npz'.format(folder_path, self.rank_id)
            if not os.path.exists(bin_path):
                return False
            print("start load input in rank{}".format(self.rank_id))
            input_cache=np.load(bin_path)
            print(input_cache)
            self.expert_ids=input_cache['expert_ids']
            self.eplb_table=input_cache['eplb_table']
            return True
    
        def save_input_cache(self):
            folder_path=self.get_data_bin_dir()
            try:
                if not os.path.exists(folder_path):
                    print("Make dir:", folder_path)
                    os.makedirs(folder_path)
            except OSError:
                pass
            bin_path='{}/input_{}r.bin'.format(folder_path, self.rank_id)
            np.savez(bin_path, expert_ids=self.expert_ids, eplb_table=self.eplb_table)
    
        def save_tensor(self, tensor, tensor_name):
            folder_path=self.get_data_bin_dir()
            try:
                if not os.path.exists(folder_path):
                    print("Make dir:", folder_path)
                    os.makedirs(folder_path)
            except OSError:
                pass
            tensor_path='{}/{}_{}_{}r.bin'.format(folder_path, tensor_name, tensor.dtype, self.rank_id)
            if tensor.dtype == 'bf16':
                np.array(tensor.float().cpu()).tofile(tensor_path)
            else:
                np.array(tensor.cpu()).tofile(tensor_path)
    
        def save_cpu_output(self):
            self.save_tensor(self.cpu_golden['balanced_expert_ids'], "cpu_balanced_expert_ids")
          
        def gen_input(self):
            if not self.load_input_cache():
                print("start gen input in rank{}".format(self.rank_id))
                eplb_table = np.zeros((self.log_ep_size, self.F - 1))
                all_ranks = np.arange(self.pyh_ep_size)
                count_column = np.random.randint(1, self.F, size=(self.log_ep_size, 1))
                for i in range(self.log_ep_size):
                    np.random.shuffle(all_ranks)
                    for j in range(count_column[i][0]):
                        eplb_table[i][j] = all_ranks[j]
                self.eplb_table = np.hstack((count_column, eplb_table))
                self.expert_ids = np.random.randint(low=0, high=self.log_ep_size, size=(self.BS, self.K))
                self.save_input_cache()
            self.eplb_table_tensor = torch.from_numpy(self.eplb_table).to(torch.int32)
            self.expert_ids_tensor=torch.from_numpy(self.expert_ids).to(self.dtype)
    
        def exec_cpu(self):
            balanced_expert_ids = np.zeros((self.BS, self.K))
            for i in range(self.BS):
                for j in range(self.K):
                    log_ep_id = self.expert_ids_tensor[i][j]
                    mod_val=math.ceil(self.world_size / self.eplb_table_tensor[log_ep_id][0])
                    phy_ep_id=self.eplb_table_tensor[log_ep_id][(self.rank_id // mod_val) + 1]
                    balanced_expert_ids[i][j] = phy_ep_id
            self.cpu_golden = {'balanced_expert_ids': torch.from_numpy(balanced_expert_ids).to(self.dtype)}
            self.save_cpu_output()
    
        def exec_npu(self):
            output = torch_npu.npu_moe_eplb_update_expert(
                expert_ids = self.expert_ids_tensor.npu(),
                eplb_table = self.eplb_table_tensor.npu(),
                local_rank_id = self.rank_id,
                world_size = self.world_size,
                balance_mode = 0
            )
            self.npu_output = {'balanced_expert_ids': output}
            print(self.npu_output)
    
    def run_npu_moe_eplb_update_expert(rank_id, BS:int, K:int, log_ep_size:int, pyh_ep_size:int, F:int, world_size:int, dtype):
        if world_size < F - 1:
            raise Exception('world_size:{} must bigger than F:{} + 1'.format(world_size, F))
        torch_npu.npu.set_device(rank_id)
        init_method = 'tcp://' + "127.0.0.1" + ':' + '50001'
        dist.init_process_group(backend="hccl", rank=rank_id, world_size=world_size, init_method=init_method)
        mee = MoeEPLBUpdateExpert(BS, K, log_ep_size, pyh_ep_size, F, world_size, rank_id, dtype)
        mee.exec_npu()
        print("End of MoeEPLBUpdateExpert.")
        
    if __name__ == "__main__":
        mp.spawn(run_npu_moe_eplb_update_expert, args=(128, 8, 256, 128, 8, 16, torch.int64), nprocs=16)
    ```

-   图模式调用

    ```python
    import math
    import os
    import time
    import psutil
    import torch
    import numpy as np
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    DATA_PATH='/home/mc2/graph_data'
    
    
    class MoeEPLBUpdateExpertModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, expert_ids, eplb_table, local_rank_id, world_size, balance_mode):
            res = torch_npu.npu_moe_eplb_update_expert(expert_ids=expert_ids,
                                                        eplb_table=eplb_table,
                                                        local_rank_id=local_rank_id,
                                                        world_size=world_size,
                                                        balance_mode=balance_mode)
            print(res)
            return res
    
    
    class MoeEPLBUpdateExpert():
        def __init__(self, BS:int, K:int, log_ep_size:int, pyh_ep_size:int, F:int, world_size:int, rank_id:int, dtype):
            self.graph_types=[1,2]
            self.BS = BS
            self.K = K
            self.log_ep_size = log_ep_size
            self.pyh_ep_size = pyh_ep_size
            self.F = F
            self.rank_id = rank_id
            self.dtype = dtype
            self.expert_ids = None
            self.eplb_table = None
            self.expert_ids_tensor = None
            self.eplb_table_tensor = None
            self.world_size = world_size
            self.balance_mode = 0
            self.cpu_golden = None
            self.graph_outputs = []
            self.gen_input()
            self.exec_cpu()
    
        def get_case_key(self):
            return '{}BS_{}K_{}LogE_{}phyE_{}F_{}r_{}'.format(self.BS, self.K, self.log_ep_size, self.pyh_ep_size, self.F, self.world_size, 'int64' if self.dtype == torch.int64 else 'int32')
    
        def get_data_bin_dir(self):
            return '{}/{}'.format(DATA_PATH, self.get_case_key())
    
        def load_input_cache(self):
            folder_path=self.get_data_bin_dir()
            if not os.path.exists(folder_path):
                return False
            bin_path='{}/input_{}r.bin.npz'.format(folder_path, self.rank_id)
            if not os.path.exists(bin_path):
                return False
            print("start load input in rank{}".format(self.rank_id))
            input_cache=np.load(bin_path)
            print(input_cache)
            self.expert_ids=input_cache['expert_ids']
            self.eplb_table=input_cache['eplb_table']
            return True
    
        def save_input_cache(self):
            folder_path=self.get_data_bin_dir()
            try:
                if not os.path.exists(folder_path):
                    print("Make dir:", folder_path)
                    os.makedirs(folder_path)
            except OSError:
                pass
            bin_path='{}/input_{}r.bin'.format(folder_path, self.rank_id)
            np.savez(bin_path, expert_ids=self.expert_ids, eplb_table=self.eplb_table)
    
        def save_tensor(self, tensor, tensor_name):
            folder_path=self.get_data_bin_dir()
            try:
                if not os.path.exists(folder_path):
                    print("Make dir:", folder_path)
                    os.makedirs(folder_path)
            except OSError:
                pass
            tensor_path='{}/{}_{}_{}r.bin'.format(folder_path, tensor_name, tensor.dtype, self.rank_id)
            if tensor.dtype == 'bf16':
                np.array(tensor.float().cpu()).tofile(tensor_path)
            else:
                np.array(tensor.cpu()).tofile(tensor_path)
    
        def save_cpu_output(self):
            self.save_tensor(self.cpu_golden['balanced_expert_ids'], "cpu_balanced_expert_ids")
    
        def gen_input(self):
            if not self.load_input_cache():
                print("start gen input in rank{}".format(self.rank_id))
                eplb_table = np.zeros((self.log_ep_size, self.F - 1))
                all_ranks = np.arange(self.pyh_ep_size)
                count_column = np.random.randint(1, self.F, size=(self.log_ep_size, 1))
                for i in range(self.log_ep_size):
                    np.random.shuffle(all_ranks)
                    for j in range(count_column[i][0]):
                        eplb_table[i][j] = all_ranks[j]
                self.eplb_table = np.hstack((count_column, eplb_table))
                self.expert_ids = np.random.randint(low=0, high=self.log_ep_size, size=(self.BS, self.K))
                self.save_input_cache()
            self.eplb_table_tensor = torch.from_numpy(self.eplb_table).to(torch.int32)
            self.expert_ids_tensor=torch.from_numpy(self.expert_ids).to(self.dtype)
    
        def exec_cpu(self):
            balanced_expert_ids = np.zeros((self.BS, self.K))
            for i in range(self.BS):
                for j in range(self.K):
                    log_ep_id = self.expert_ids_tensor[i][j]
                    mod_val=math.ceil(self.world_size / self.eplb_table_tensor[log_ep_id][0])
                    phy_ep_id=self.eplb_table_tensor[log_ep_id][(self.rank_id // mod_val) + 1]
                    balanced_expert_ids[i][j] = phy_ep_id
            self.cpu_golden = {'balanced_expert_ids': torch.from_numpy(balanced_expert_ids).to(self.dtype)}
            self.save_cpu_output()
    
        def define_model(self, model, graph_type):
            print("------------------graph_type", graph_type)
            import torchair
            if graph_type == 1:          # 传统入图模式，静态shape+在线编译场景
                print('graph_type=1，测试场景：传统入图模式，静态shape+二进制场景')
                npu_backend = torchair.get_npu_backend(compiler_config=None)
                model = torch.compile(model, backend=npu_backend, dynamic=False)
            elif graph_type == 2:        # ACLNN入图模式，动态shape+二进制
                print('graph_type=2，测试场景：ACLNN入图模式，动态shape+二进制')
                npu_backend = torchair.get_npu_backend(compiler_config=None)
                model = torch.compile(model, backend=npu_backend, dynamic=True)
            else:
                print('用例未设置graph_type，测试场景：单算子模式')
            return model
    
        def exec_graph(self):
            model = MoeEPLBUpdateExpertModel()
            for graph_type_ in self.graph_types:
                model = self.define_model(model, graph_type_)
                balanced_expert_ids=None
                model = self.define_model(model, graph_type_)
                balanced_expert_ids = model(self.expert_ids_tensor, self.eplb_table_tensor, self.rank_id, self.world_size, self.balance_mode)
                self.graph_outputs.append({'balanced_expert_ids': balanced_expert_ids,'name':'graph_{}'.format('static' if graph_type_ == 1 else 'dynamic')})
    
    def run_npu_moe_eplb_update_expert(rank_id, BS:int, K:int, log_ep_size:int, pyh_ep_size:int, F:int, world_size:int, dtype):
        if world_size < F - 1:
            raise Exception('world_size:{} must bigger than F:{} + 1'.format(world_size, F))
        torch_npu.npu.set_device(rank_id)
        init_method = 'tcp://' + "127.0.0.1" + ':' + '50001'
        dist.init_process_group(backend="hccl", rank=rank_id, world_size=world_size, init_method=init_method)
        mee = MoeEPLBUpdateExpert(BS, K, log_ep_size, pyh_ep_size, F, world_size, rank_id, dtype)
        mee.exec_graph()
    
    
    if __name__ == "__main__":
        mp.spawn(run_npu_moe_eplb_update_expert, args=(128, 8, 256, 128, 8, 16, torch.int64), nprocs=16)
    ```

