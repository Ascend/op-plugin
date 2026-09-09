# Distributed APIs

This section describes the adapted distributed APIs that provide parallel computing capabilities.

**Table 1** Distributed APIs

<a name="table2069619331171"></a>
<table><thead align="left"><tr id="row86962336177"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p2696143312179"><a name="p2696143312179"></a><a name="p2696143312179"></a>API</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p669613318178"><a name="p669613318178"></a><a name="p669613318178"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row669611337171"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p46966336179"><a name="p46966336179"></a><a name="p46966336179"></a><a href="(beta)torch-distributed-is_hccl_available.md">(beta) torch.distributed.is_hccl_available</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p176961133151712"><a name="p176961133151712"></a><a name="p176961133151712"></a>Determines whether the <code>HCCL</code> communication backend is available, similar to <code>torch.distributed.is_nccl_available</code>.</p>
</td>
</tr>
<tr id="row186961833171719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1669719330179"><a name="p1669719330179"></a><a name="p1669719330179"></a><a href="torch-distributed-distributed_c10d.md">torch.distributed.distributed_c10d._world.default_pg._get_backend(torch.device("npu")).get_hccl_comm_name</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p5697183351712"><a name="p5697183351712"></a><a name="p5697183351712"></a>Obtains the name of the collective communication domain from the initialized domain.</p>
</td>
</tr>
<tr id="row174513178290"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p204514170294"><a name="p204514170294"></a><a name="p204514170294"></a><a href="(beta)torch-distributed-ProcessGroupHCCL.md">(beta) torch.distributed.ProcessGroupHCCL</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001788457948_en-us_topic_0000001718962912_p173841155158"><a name="en-us_topic_0000001788457948_en-us_topic_0000001718962912_p173841155158"></a><a name="en-us_topic_0000001788457948_en-us_topic_0000001718962912_p173841155158"></a>Creates and returns a <code>ProcessGroupHCCL</code> object.</p>
</td>
</tr>
<tr id="row193021022143117"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1530342217311"><a name="p1530342217311"></a><a name="p1530342217311"></a><a href="(beta)torch_npu-distributed-reinit_process_group.md">(beta) torch_npu.distributed.reinit_process_group</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p03031822133115"><a name="p03031822133115"></a><a name="p03031822133115"></a>Rebuilds the <code>ProcessGroup</code> for collective communication.</p>
</td>
</tr>
<tr id="row6513134685111"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p451334611515"><a name="p451334611515"></a><a name="p451334611515"></a><a href="torch_npu-distributed-reduce_scatter_tensor_uneven.md">(beta) torch_npu.distributed.reduce_scatter_tensor_uneven</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p651315465516"><a name="p651315465516"></a><a name="p651315465516"></a>Extends the native <code>torch.distributed.reduce_scatter_tensor</code> API by supporting zero-copy and uneven tensor splitting in <code>torch_npu.distributed.reduce_scatter_tensor_uneven</code>.</p>
</td>
</tr>
<tr id="row125161049165116"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p105171249165115"><a name="p105171249165115"></a><a name="p105171249165115"></a><a href="(beta)torch_npu-distributed-all_gather_into_tensor_uneven.md">(beta) torch_npu.distributed.all_gather_into_tensor_uneven</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2517104919514"><a name="p2517104919514"></a><a name="p2517104919514"></a>Extends the native <code>torch.distributed.all_gather_into_tensor</code> API by supporting zero-copy and uneven tensor splitting in <code>torch_npu.distributed.all_gather_into_tensor_uneven</code>.</p>
</td>
</tr>
</tbody>
</table>
