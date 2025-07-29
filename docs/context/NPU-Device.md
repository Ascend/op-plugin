# NPU Device

**表1** NPU device API

<a name="table13377193418547"></a>
<table><thead align="left"><tr id="row1888615477417"><th class="cellrowborder" valign="top" width="54.559999999999995%" id="mcps1.2.3.1.1"><p id="p188634764114"><a name="p188634764114"></a><a name="p188634764114"></a>API接口</p>
</th>
<th class="cellrowborder" valign="top" width="45.440000000000005%" id="mcps1.2.3.1.2"><p id="p188862474415"><a name="p188862474415"></a><a name="p188862474415"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row939163412547"><td class="cellrowborder" valign="top" width="54.559999999999995%" headers="mcps1.2.3.1.1 "><p id="p13919348541"><a name="p13919348541"></a><a name="p13919348541"></a>（<span id="ph168289415199"><a name="ph168289415199"></a><a name="ph168289415199"></a>beta</span>）torch_npu.npu.is_initialized</p>
</td>
<td class="cellrowborder" rowspan="11" valign="top" width="45.440000000000005%" headers="mcps1.2.3.1.2 "><p id="p20972816174218"><a name="p20972816174218"></a><a name="p20972816174218"></a>Torch_npu提供设备相关的部分接口，具体可参考<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/PyTorchNativeapi/ptaoplist_000158.html">torch.cuda</a>。</p>
</td>
</tr>
<tr id="row13391173465410"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p173915341547"><a name="p173915341547"></a><a name="p173915341547"></a>（<span id="ph1943317523417"><a name="ph1943317523417"></a><a name="ph1943317523417"></a>beta</span>）torch_npu.npu.init</p>
</td>
</tr>
<tr id="row17391173465415"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p0391173495420"><a name="p0391173495420"></a><a name="p0391173495420"></a>（<span id="ph351513733413"><a name="ph351513733413"></a><a name="ph351513733413"></a>beta</span>）torch_npu.npu.get_device_name</p>
</td>
</tr>
<tr id="row939118343543"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p15391534165418"><a name="p15391534165418"></a><a name="p15391534165418"></a>（<span id="ph718317101345"><a name="ph718317101345"></a><a name="ph718317101345"></a>beta</span>）torch_npu.npu.can_device_access_peer</p>
</td>
</tr>
<tr id="row539173445416"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p23911634165410"><a name="p23911634165410"></a><a name="p23911634165410"></a>（<span id="ph9238212103412"><a name="ph9238212103412"></a><a name="ph9238212103412"></a>beta</span>）torch_npu.npu.get_device_properties</p>
</td>
</tr>
<tr id="row1139193417545"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p133913348548"><a name="p133913348548"></a><a name="p133913348548"></a>（<span id="ph310411493415"><a name="ph310411493415"></a><a name="ph310411493415"></a>beta</span>）torch_npu.npu.device_of</p>
</td>
</tr>
<tr id="row19391103416545"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p20391153411547"><a name="p20391153411547"></a><a name="p20391153411547"></a>（<span id="ph171181816173418"><a name="ph171181816173418"></a><a name="ph171181816173418"></a>beta</span>）torch_npu.npu.current_blas_handle</p>
</td>
</tr>
<tr id="row1739173414545"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p239173495417"><a name="p239173495417"></a><a name="p239173495417"></a>（<span id="ph1024311813418"><a name="ph1024311813418"></a><a name="ph1024311813418"></a>beta</span>）torch_npu.npu.set_stream</p>
</td>
</tr>
<tr id="row13391163465411"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p33911343541"><a name="p33911343541"></a><a name="p33911343541"></a>（<span id="ph43843205344"><a name="ph43843205344"></a><a name="ph43843205344"></a>beta</span>）torch_npu.npu.set_sync_debug_mode</p>
</td>
</tr>
<tr id="row73912034125411"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p173918344543"><a name="p173918344543"></a><a name="p173918344543"></a>（<span id="ph15373102263415"><a name="ph15373102263415"></a><a name="ph15373102263415"></a>beta</span>）torch_npu.npu.get_sync_debug_mode</p>
</td>
</tr>
<tr id="row19886154310548"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1688617434546"><a name="p1688617434546"></a><a name="p1688617434546"></a>（<span id="ph94981924173410"><a name="ph94981924173410"></a><a name="ph94981924173410"></a>beta</span>）torch_npu.npu.utilization</p>
</td>
</tr>
<tr id="row16363129161417"><td class="cellrowborder" valign="top" width="54.559999999999995%" headers="mcps1.2.3.1.1 "><p id="p83919341546"><a name="p83919341546"></a><a name="p83919341546"></a>（<span id="ph871092653413"><a name="ph871092653413"></a><a name="ph871092653413"></a>beta</span>）torch_npu.npu.get_device_capability</p>
</td>
<td class="cellrowborder" valign="top" width="45.440000000000005%" headers="mcps1.2.3.1.2 "><p id="p1036412951413"><a name="p1036412951413"></a><a name="p1036412951413"></a>预留接口，暂不支持，接口默认返回None。</p>
</td>
</tr>
</tbody>
</table>

