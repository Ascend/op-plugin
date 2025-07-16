# Distributed接口列表

本章节包含适配后的分布式接口，提供并行计算能力。

**表1** Distributed API

<a name="table2069619331171"></a>
<table><thead align="left"><tr id="row86962336177"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p2696143312179"><a name="p2696143312179"></a><a name="p2696143312179"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p669613318178"><a name="p669613318178"></a><a name="p669613318178"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row669611337171"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p46966336179"><a name="p46966336179"></a><a name="p46966336179"></a><a href="（beta）torch-distributed-is_hccl_available.md">（beta）torch.distributed.is_hccl_available</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p176961133151712"><a name="p176961133151712"></a><a name="p176961133151712"></a>判断HCCL通信后端是否可用，与torch.distributed.is_nccl_available类似。</p>
</td>
</tr>
<tr id="row186961833171719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1669719330179"><a name="p1669719330179"></a><a name="p1669719330179"></a><a href="torch-distributed-distributed_c10d.md">torch.distributed.distributed_c10d._world.default_pg._get_backend(torch.device("npu")).get_hccl_comm_name</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p5697183351712"><a name="p5697183351712"></a><a name="p5697183351712"></a>从初始化完成的集合通信域中获取集合通信域名字。</p>
</td>
</tr>
<tr id="row174513178290"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p204514170294"><a name="p204514170294"></a><a name="p204514170294"></a><a href="（beta）torch-distributed-ProcessGroupHCCL.md">（beta）torch.distributed.ProcessGroupHCCL</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001788457948_zh-cn_topic_0000001718962912_p173841155158"><a name="zh-cn_topic_0000001788457948_zh-cn_topic_0000001718962912_p173841155158"></a><a name="zh-cn_topic_0000001788457948_zh-cn_topic_0000001718962912_p173841155158"></a>创建一个ProcessGroupHCCL对象并返回。</p>
</td>
</tr>
<tr id="row193021022143117"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1530342217311"><a name="p1530342217311"></a><a name="p1530342217311"></a><a href="（beta）torch_npu-distributed-reinit_process_group.md">（beta）torch_npu.distributed.reinit_process_group</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p03031822133115"><a name="p03031822133115"></a><a name="p03031822133115"></a>重新构建processgroup集合通信域。</p>
</td>
</tr>
<tr id="row6513134685111"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p451334611515"><a name="p451334611515"></a><a name="p451334611515"></a><a href="torch_npu-distributed-reduce_scatter_tensor_uneven.md">（beta）torch_npu.distributed.reduce_scatter_tensor_uneven</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p651315465516"><a name="p651315465516"></a><a name="p651315465516"></a>参考原生接口torch.distributed.reduce_scatter_tensor功能，torch_npu.distributed.reduce_scatter_tensor_uneven接口新增支持零拷贝和非等长切分功能。</p>
</td>
</tr>
<tr id="row125161049165116"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p105171249165115"><a name="p105171249165115"></a><a name="p105171249165115"></a><a href="（beta）torch_npu-distributed-all_gather_into_tensor_uneven.md">（beta）torch_npu.distributed.all_gather_into_tensor_uneven</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2517104919514"><a name="p2517104919514"></a><a name="p2517104919514"></a>参考原生接口torch.distributed.all_gather_into_tensor功能，torch_npu.distributed.all_gather_into_tensor_uneven接口新增支持零拷贝和非等长切分功能。</p>
</td>
</tr>
</tbody>
</table>

