# torch_npu.npu.mstx.mstx_range

## 函数原型

```
torch_npu.npu.mstx.mstx_range(msg: str, stream=None, domain: str='default')
```

## 功能说明

range装饰器，用来采集被装饰函数的range执行耗时。

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
<tbody><tr id="row1131131265511"><td class="cellrowborder" valign="top" width="28.65286528652865%" headers="mcps1.2.4.1.1 "><p id="p8272551931"><a name="p8272551931"></a><a name="p8272551931"></a>msg</p>
</td>
<td class="cellrowborder" valign="top" width="26.782678267826782%" headers="mcps1.2.4.1.2 "><p id="p19272251234"><a name="p19272251234"></a><a name="p19272251234"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.56445644564456%" headers="mcps1.2.4.1.3 "><p id="p131994242276"><a name="p131994242276"></a><a name="p131994242276"></a>打点携带信息字符串指针。</p>
</td>
</tr>
<tr id="row18212822311"><td class="cellrowborder" valign="top" width="28.65286528652865%" headers="mcps1.2.4.1.1 "><p id="p211549516"><a name="p211549516"></a><a name="p211549516"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="26.782678267826782%" headers="mcps1.2.4.1.2 "><p id="p1920117129516"><a name="p1920117129516"></a><a name="p1920117129516"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.56445644564456%" headers="mcps1.2.4.1.3 "><p id="p175061539114317"><a name="p175061539114317"></a><a name="p175061539114317"></a>用于执行打点任务的stream。</p>
<a name="ul9122114715433"></a><a name="ul9122114715433"></a><ul id="ul9122114715433"><li>配置为None或不配置时，只标记Host侧的瞬时事件。</li><li>配置为有效的stream时，标识Host侧和对应Device侧的瞬时事件。</li></ul>
</td>
</tr>
<tr id="row18212822311"><td class="cellrowborder" valign="top" width="28.65286528652865%" headers="mcps1.2.4.1.1 "><p id="p211549516">domain</p>
</td>
<td class="cellrowborder" valign="top" width="26.782678267826782%" headers="mcps1.2.4.1.2 "><p id="p1920117129516">输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.56445644564456%" headers="mcps1.2.4.1.3 "><p id="p175061539114317">指定的domain名称，表示在指定的domain内，标识时间段事件。默认为default，表示默认domain，不设置也为默认domain。</p>
</td>
</tr>
</tbody>
</table>

## 返回值

无

## 支持的型号

- <term> Atlas 训练系列产品</term> 
- <term> Atlas A2 训练系列产品</term> 
- <term> Atlas A3 训练系列产品</term>

