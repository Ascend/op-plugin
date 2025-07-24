# torch_npu.npu.mstx.range_end

## 函数原型

```
torch_npu.npu.mstx.range_end(range_id: int, domain: str='default') -> int:
```

## 功能说明

标识打点结束。

## 参数说明

**表1** 参数说明

<a name="table827101275518"></a>
<table><thead align="left"><tr id="row429121265517"><th class="cellrowborder" valign="top" width="28.65286528652865%" id="mcps1.2.4.1.1"><p id="p1329121214558"><a name="p1329121214558"></a><a name="p1329121214558"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="26.782678267826782%" id="mcps1.2.4.1.2"><p id="p10230141454318"><a name="p10230141454318"></a><a name="p10230141454318"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="44.56445644564456%" id="mcps1.2.4.1.3"><p id="p83121275519"><a name="p83121275519"></a><a name="p83121275519"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1131131265511"><td class="cellrowborder" valign="top" width="28.65286528652865%" headers="mcps1.2.4.1.1 "><p id="p7669321185110"><a name="p7669321185110"></a><a name="p7669321185110"></a>range_id</p>
</td>
<td class="cellrowborder" valign="top" width="26.782678267826782%" headers="mcps1.2.4.1.2 "><p id="p723015144436"><a name="p723015144436"></a><a name="p723015144436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.56445644564456%" headers="mcps1.2.4.1.3 "><p id="p131994242276"><a name="p131994242276"></a><a name="p131994242276"></a>通过torch_npu.npu.mstx.range_start接口返回的id。</p>
</td>
</tr></tr>
<tr id="row18212822311"><td class="cellrowborder" valign="top" width="28.65286528652865%" headers="mcps1.2.4.1.1 "><p id="p211549516">domain</p>
</td>
<td class="cellrowborder" valign="top" width="26.782678267826782%" headers="mcps1.2.4.1.2 "><p id="p1920117129516">输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.56445644564456%" headers="mcps1.2.4.1.3 "><p id="p175061539114317">指定的domain名称，表示在指定的domain内，标识时间段事件的结束。需要与torch_npu.npu.mstx.range_start接口的domain配置一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值

无。

## 支持的型号

- <term> Atlas 训练系列产品</term> 
- <term> Atlas A2 训练系列产品</term> 
- <term> Atlas A3 训练系列产品</term>

