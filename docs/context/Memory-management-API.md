# Memory management API

**表1** Memory management API

<a name="table683971665519"></a>
<table><thead align="left"><tr id="row1357083844218"><th class="cellrowborder" valign="top" width="63.61%" id="mcps1.2.3.1.1"><p id="p1757053818424"><a name="p1757053818424"></a><a name="p1757053818424"></a>API接口</p>
</th>
<th class="cellrowborder" valign="top" width="36.39%" id="mcps1.2.3.1.2"><p id="p2057018381424"><a name="p2057018381424"></a><a name="p2057018381424"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row12860101605513"><td class="cellrowborder" valign="top" width="63.61%" headers="mcps1.2.3.1.1 "><p id="p486019163553"><a name="p486019163553"></a><a name="p486019163553"></a>（<span id="ph168289415199"><a name="ph168289415199"></a><a name="ph168289415199"></a>beta</span>）torch_npu.npu.caching_allocator_alloc</p>
</td>
<td class="cellrowborder" rowspan="18" valign="top" width="36.39%" headers="mcps1.2.3.1.2 "><p id="p6146131084320"><a name="p6146131084320"></a><a name="p6146131084320"></a>Torch_npu提供内存管理相关的部分接口，具体可参考<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/PyTorchNativeapi/ptaoplist_000158.html">torch.cuda</a>。</p>
</td>
</tr>
<tr id="row0860131635519"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p19860111610554"><a name="p19860111610554"></a><a name="p19860111610554"></a>（<span id="ph1545853943420"><a name="ph1545853943420"></a><a name="ph1545853943420"></a>beta</span>）torch_npu.npu.caching_allocator_delete</p>
</td>
</tr>
<tr id="row1486071645519"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1886031615519"><a name="p1886031615519"></a><a name="p1886031615519"></a>（<span id="ph62311441113420"><a name="ph62311441113420"></a><a name="ph62311441113420"></a>beta</span>）torch_npu.npu.set_per_process_memory_fraction</p>
</td>
</tr>
<tr id="row12860161610554"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p4860151635519"><a name="p4860151635519"></a><a name="p4860151635519"></a>（<span id="ph78914314341"><a name="ph78914314341"></a><a name="ph78914314341"></a>beta</span>）torch_npu.npu.empty_cache</p>
</td>
</tr>
<tr id="row38601116185513"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p9860516115513"><a name="p9860516115513"></a><a name="p9860516115513"></a>（<span id="ph11300154553410"><a name="ph11300154553410"></a><a name="ph11300154553410"></a>beta</span>）torch_npu.npu.memory_stats</p>
</td>
</tr>
<tr id="row1186031665513"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p10860316205512"><a name="p10860316205512"></a><a name="p10860316205512"></a>（<span id="ph111981047103411"><a name="ph111981047103411"></a><a name="ph111981047103411"></a>beta</span>）torch_npu.npu.memory_stats_as_nested_dict</p>
</td>
</tr>
<tr id="row16860116135518"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p88601416155512"><a name="p88601416155512"></a><a name="p88601416155512"></a>（<span id="ph12169174953412"><a name="ph12169174953412"></a><a name="ph12169174953412"></a>beta</span>）torch_npu.npu.reset_accumulated_memory_stats</p>
</td>
</tr>
<tr id="row68601916145518"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p4860141675510"><a name="p4860141675510"></a><a name="p4860141675510"></a>（<span id="ph678145115340"><a name="ph678145115340"></a><a name="ph678145115340"></a>beta</span>）torch_npu.npu.reset_peak_memory_stats</p>
</td>
</tr>
<tr id="row3860716175511"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p178604162559"><a name="p178604162559"></a><a name="p178604162559"></a>（<span id="ph59721052153419"><a name="ph59721052153419"></a><a name="ph59721052153419"></a>beta</span>）torch_npu.npu.reset_max_memory_allocated</p>
</td>
</tr>
<tr id="row886012165551"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1786031612559"><a name="p1786031612559"></a><a name="p1786031612559"></a>（<span id="ph1373115517349"><a name="ph1373115517349"></a><a name="ph1373115517349"></a>beta</span>）torch_npu.npu.reset_max_memory_cached</p>
</td>
</tr>
<tr id="row286091645515"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p9860111655513"><a name="p9860111655513"></a><a name="p9860111655513"></a>（<span id="ph0844859143419"><a name="ph0844859143419"></a><a name="ph0844859143419"></a>beta</span>）torch_npu.npu.memory_allocated</p>
</td>
</tr>
<tr id="row286071611551"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p10860101655517"><a name="p10860101655517"></a><a name="p10860101655517"></a>（<span id="ph128776112351"><a name="ph128776112351"></a><a name="ph128776112351"></a>beta</span>）torch_npu.npu.max_memory_allocated</p>
</td>
</tr>
<tr id="row78608164553"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p18603168555"><a name="p18603168555"></a><a name="p18603168555"></a>（<span id="ph2032720410358"><a name="ph2032720410358"></a><a name="ph2032720410358"></a>beta</span>）torch_npu.npu.memory_reserved</p>
</td>
</tr>
<tr id="row1686071675513"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p5860121613559"><a name="p5860121613559"></a><a name="p5860121613559"></a>（<span id="ph3247476355"><a name="ph3247476355"></a><a name="ph3247476355"></a>beta</span>）torch_npu.npu.max_memory_reserved</p>
</td>
</tr>
<tr id="row1486051675515"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p48601516195510"><a name="p48601516195510"></a><a name="p48601516195510"></a>（<span id="ph6190189103516"><a name="ph6190189103516"></a><a name="ph6190189103516"></a>beta</span>）torch_npu.npu.memory_cached</p>
</td>
</tr>
<tr id="row0860416105516"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p15860151685516"><a name="p15860151685516"></a><a name="p15860151685516"></a>（<span id="ph188631121350"><a name="ph188631121350"></a><a name="ph188631121350"></a>beta</span>）torch_npu.npu.max_memory_cached</p>
</td>
</tr>
<tr id="row186031675517"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p486041618550"><a name="p486041618550"></a><a name="p486041618550"></a>（<span id="ph844815154352"><a name="ph844815154352"></a><a name="ph844815154352"></a>beta</span>）torch_npu.npu.memory_snapshot</p>
</td>
</tr>
<tr id="row10860101635512"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1486012165555"><a name="p1486012165555"></a><a name="p1486012165555"></a>（<span id="ph34464179352"><a name="ph34464179352"></a><a name="ph34464179352"></a>beta</span>）torch_npu.npu.memory_summary</p>
</td>
</tr>
<tr id="row1933424017243"><td class="cellrowborder" valign="top" width="63.61%" headers="mcps1.2.3.1.1 "><p id="p7819165411224"><a name="p7819165411224"></a><a name="p7819165411224"></a>torch_npu.npu.NPUPluggableAllocator</p>
</td>
<td class="cellrowborder" valign="top" width="36.39%" headers="mcps1.2.3.1.2 "><p id="p172891387259"><a name="p172891387259"></a><a name="p172891387259"></a>该接口涉及高危操作，使用请参考<a href="torch-npu-npu-NPUPluggableAllocator.md">torch_npu.npu.NPUPluggableAllocator</a>。</p>
</td>
</tr>
<tr id="row12478193618244"><td class="cellrowborder" valign="top" width="63.61%" headers="mcps1.2.3.1.1 "><p id="p681913541224"><a name="p681913541224"></a><a name="p681913541224"></a>torch_npu.npu.change_current_allocator</p>
</td>
<td class="cellrowborder" valign="top" width="36.39%" headers="mcps1.2.3.1.2 "><p id="p75031810165517"><a name="p75031810165517"></a><a name="p75031810165517"></a>该接口涉及高危操作，使用请参考<a href="torch-npu-npu-change_current_allocator.md">torch_npu.npu.change_current_allocator</a>。</p>
</td>
</tr>
</tbody>
</table>

