# （beta）NPU Tensor

Torch_npu提供NPU tensor相关的部分接口使用与Cuda类似。

**表1** NPU Tensor API(beta)

<a name="zh-cn_topic_0000001655204269_table896052319157"></a>
<table><thead align="left"><tr id="row74720376177"><th class="cellrowborder" valign="top" width="26.75%" id="mcps1.2.5.1.1"><p id="p18161141918211"><a name="p18161141918211"></a><a name="p18161141918211"></a>PyTorch原生API名称</p>
</th>
<th class="cellrowborder" valign="top" width="30.349999999999998%" id="mcps1.2.5.1.2"><p id="p3164023235"><a name="p3164023235"></a><a name="p3164023235"></a>NPU形式名称</p>
</th>
<th class="cellrowborder" valign="top" width="6.22%" id="mcps1.2.5.1.3"><p id="p216112194214"><a name="p216112194214"></a><a name="p216112194214"></a>是否支持</p>
</th>
<th class="cellrowborder" valign="top" width="36.68%" id="mcps1.2.5.1.4"><p id="p151413620217"><a name="p151413620217"></a><a name="p151413620217"></a>参考链接</p>
</th>
</tr>
</thead>
<tbody><tr id="row16728461210"><td class="cellrowborder" valign="top" width="26.75%" headers="mcps1.2.5.1.1 "><p id="p137218461814"><a name="p137218461814"></a><a name="p137218461814"></a>torch.cuda.DoubleTensor</p>
</td>
<td class="cellrowborder" valign="top" width="30.349999999999998%" headers="mcps1.2.5.1.2 "><p id="p1996044910114"><a name="p1996044910114"></a><a name="p1996044910114"></a>torch_npu.npu.DoubleTensor</p>
</td>
<td class="cellrowborder" valign="top" width="6.22%" headers="mcps1.2.5.1.3 "><p id="p27215465118"><a name="p27215465118"></a><a name="p27215465118"></a>是</p>
</td>
<td class="cellrowborder" rowspan="9" valign="top" width="36.68%" headers="mcps1.2.5.1.4 "><p id="p225252418262"><a name="p225252418262"></a><a name="p225252418262"></a><a href="https://pytorch.org/docs/stable/tensors.html#data-types" target="_blank" rel="noopener noreferrer">https://pytorch.org/docs/stable/tensors.html#data-types</a></p>
</td>
</tr>
<tr id="row47210461517"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p4725461314"><a name="p4725461314"></a><a name="p4725461314"></a>torch.cuda.ShortTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p49600493114"><a name="p49600493114"></a><a name="p49600493114"></a>torch_npu.npu.ShortTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p772146719"><a name="p772146719"></a><a name="p772146719"></a>是</p>
</td>
</tr>
<tr id="row19728461116"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p19728461012"><a name="p19728461012"></a><a name="p19728461012"></a>torch.cuda.CharTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p109609491714"><a name="p109609491714"></a><a name="p109609491714"></a>torch_npu.npu.CharTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p8724460120"><a name="p8724460120"></a><a name="p8724460120"></a>是</p>
</td>
</tr>
<tr id="row47317466110"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1873154618118"><a name="p1873154618118"></a><a name="p1873154618118"></a>torch.cuda.ByteTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1896013491711"><a name="p1896013491711"></a><a name="p1896013491711"></a>torch_npu.npu.ByteTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p11739461719"><a name="p11739461719"></a><a name="p11739461719"></a>是</p>
</td>
</tr>
<tr id="row1509152312215"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p5471124912115"><a name="p5471124912115"></a><a name="p5471124912115"></a>torch.cuda.FloatTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p195091923112118"><a name="p195091923112118"></a><a name="p195091923112118"></a>torch_npu.npu.FloatTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p155091223162115"><a name="p155091223162115"></a><a name="p155091223162115"></a>是</p>
</td>
</tr>
<tr id="row16510192372120"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p8471144914219"><a name="p8471144914219"></a><a name="p8471144914219"></a>torch.cuda.HalfTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p155102231215"><a name="p155102231215"></a><a name="p155102231215"></a>torch_npu.npu.HalfTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p951012235213"><a name="p951012235213"></a><a name="p951012235213"></a>是</p>
</td>
</tr>
<tr id="row17861230162113"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p9584195616219"><a name="p9584195616219"></a><a name="p9584195616219"></a>torch.cuda.IntTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p158615305219"><a name="p158615305219"></a><a name="p158615305219"></a>torch_npu.npu.IntTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p5861530152110"><a name="p5861530152110"></a><a name="p5861530152110"></a>是</p>
</td>
</tr>
<tr id="row8752162662111"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p95841156132111"><a name="p95841156132111"></a><a name="p95841156132111"></a>torch.cuda.BoolTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1575212662118"><a name="p1575212662118"></a><a name="p1575212662118"></a>torch_npu.npu.BoolTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p13752162642116"><a name="p13752162642116"></a><a name="p13752162642116"></a>是</p>
</td>
</tr>
<tr id="row97533261212"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p6753182613210"><a name="p6753182613210"></a><a name="p6753182613210"></a>torch.cuda.LongTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p775314269219"><a name="p775314269219"></a><a name="p775314269219"></a>torch_npu.npu.LongTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p975372632115"><a name="p975372632115"></a><a name="p975372632115"></a>是</p>
</td>
</tr>
</tbody>
</table>

