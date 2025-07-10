# torch_npu.utils接口列表

本章节包含辅助性接口，如异步保存等。

**表1** torch_npu.utils API

<a name="table77751852132012"></a>
<table><thead align="left"><tr id="row1377512526202"><th class="cellrowborder" valign="top" width="39.14%" id="mcps1.2.3.1.1"><p id="p177685262020"><a name="p177685262020"></a><a name="p177685262020"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="60.86%" id="mcps1.2.3.1.2"><p id="p1177685213201"><a name="p1177685213201"></a><a name="p1177685213201"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row27761152102014"><td class="cellrowborder" valign="top" width="39.14%" headers="mcps1.2.3.1.1 "><p id="p12776752192011"><a name="p12776752192011"></a><a name="p12776752192011"></a><a href="（beta）torch_npu-utils-save_async.md">（beta）torch_npu.utils.save_async</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.86%" headers="mcps1.2.3.1.2 "><p id="p193131582206"><a name="p193131582206"></a><a name="p193131582206"></a>异步保存一个对象到一个硬盘文件上。</p>
</td>
</tr>
<tr id="row848935494011"><td class="cellrowborder" valign="top" width="39.14%" headers="mcps1.2.3.1.1 "><p id="p64891454164016"><a name="p64891454164016"></a><a name="p64891454164016"></a><a href="（beta）torch_npu-utils-npu_combine_tensors.md">（beta）torch_npu.utils.npu_combine_tensors</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.86%" headers="mcps1.2.3.1.2 "><p id="p2049005414407"><a name="p2049005414407"></a><a name="p2049005414407"></a>应用基于NPU的Tensor融合操作，将NPU上的多个Tensor融合为内存连续的一个新Tensor，访问原Tensor时实际访问新融合Tensor的对应偏移地址。</p>
</td>
</tr>
<tr id="row14789258164020"><td class="cellrowborder" valign="top" width="39.14%" headers="mcps1.2.3.1.1 "><p id="p3790558174012"><a name="p3790558174012"></a><a name="p3790558174012"></a><a href="（beta）torch_npu-utils-get_part_combined_tensor.md">（beta）torch_npu.utils.get_part_combined_tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.86%" headers="mcps1.2.3.1.2 "><p id="p299109481"><a name="p299109481"></a><a name="p299109481"></a>根据地址偏移及内存大小，从经过torch_npu.utils.npu_combine_tensors融合后的融合Tensor中获取局部Tensor。</p>
</td>
</tr>
<tr id="row54301722410"><td class="cellrowborder" valign="top" width="39.14%" headers="mcps1.2.3.1.1 "><p id="p443019224112"><a name="p443019224112"></a><a name="p443019224112"></a><a href="（beta）torch_npu-utils-is_combined_tensor_valid.md">（beta）torch_npu.utils.is_combined_tensor_valid</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.86%" headers="mcps1.2.3.1.2 "><p id="p731884115495"><a name="p731884115495"></a><a name="p731884115495"></a>校验Tensor列表中的Tensor是否全部属于一个经过torch_npu.utils.npu_combine_tensors融合后的新融合Tensor。</p>
</td>
</tr>
<tr id="row13911539133113"><td class="cellrowborder" valign="top" width="39.14%" headers="mcps1.2.3.1.1 "><p id="p491133910317"><a name="p491133910317"></a><a name="p491133910317"></a><a href="（beta）torch_npu-utils-FlopsCounter.md">（beta）torch_npu.utils.FlopsCounter</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.86%" headers="mcps1.2.3.1.2 "><p id="p1991153923112"><a name="p1991153923112"></a><a name="p1991153923112"></a>Flops统计类，用于统计各个常见cube类算子的浮点计算Flops，采用单例模式。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-utils.set_thread_affinity.md">torch_npu.utils.set_thread_affinity</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>设置当前线程的绑核区间。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-utils.reset_thread_affinity.md">torch_npu.utils.reset_thread_affinity</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>恢复当前线程的绑核区间为主线程区间。</p>
</td>
</tr>
</tbody>
</table>

