# torch_npu.optim APIs

This section describes various fusion optimizers that provide better performance than common optimizers.

**Table 1** torch_npu.optim APIs

<a name="table16908451436"></a>
<table><thead align="left"><tr id="row11690114510437"><th class="cellrowborder" valign="top" width="38.34%" id="mcps1.2.3.1.1"><p id="p12690164511431"><a name="p12690164511431"></a><a name="p12690164511431"></a>API Name</p>
</th>
<th class="cellrowborder" valign="top" width="61.660000000000004%" id="mcps1.2.3.1.2"><p id="p20690945114315"><a name="p20690945114315"></a><a name="p20690945114315"></a> Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row7690645184319"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p2069013457432"><a name="p2069013457432"></a><a name="p2069013457432"></a><a href="torch_npu-optim-NpuFusedOptimizerBase.md">torch_npu.optim.NpuFusedOptimizerBase</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p76905453432"><a name="p76905453432"></a><a name="p76905453432"></a>Provides the base class for tensor fusion optimizers. This class implements basic optimizer functions such as gradient zeroing and gradient updating. Users can inherit from this class to implement custom fused optimizers.</p>
</td>
</tr>
<tr id="row6690174515436"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p1169064524314"><a name="p1169064524314"></a><a name="p1169064524314"></a><a href="torch_npu-optim-NpuFusedSGD.md">torch_npu.optim.NpuFusedSGD</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p1969064524310"><a name="p1969064524310"></a><a name="p1969064524310"></a>Provides a high-performance SGD optimizer implemented through tensor fusion. The core functionality is compatible with <code>torch.optim.SGD</code>.</p>
</td>
</tr>
<tr id="row106904453433"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p19690194515432"><a name="p19690194515432"></a><a name="p19690194515432"></a><a href="torch_npu-optim-NpuFusedAdadelta.md">torch_npu.optim.NpuFusedAdadelta</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p18312105614159"><a name="p18312105614159"></a><a name="p18312105614159"></a>Provides a high-performance Adadelta optimizer implemented through tensor fusion. The core functionality is compatible with <code>torch.optim.Adadelta</code>.</p>
</td>
</tr>
<tr id="row229851194319"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p1629185120434"><a name="p1629185120434"></a><a name="p1629185120434"></a><a href="torch_npu-optim-NpuFusedLamb.md">torch_npu.optim.NpuFusedLamb</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p1084119451352"><a name="p1084119451352"></a><a name="p1084119451352"></a>Provides a high-performance Lamb optimizer implemented through tensor fusion.</p>
</td>
</tr>
<tr id="row132911515430"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p62965115438"><a name="p62965115438"></a><a name="p62965115438"></a><a href="torch_npu-optim-NpuFusedAdam.md">torch_npu.optim.NpuFusedAdam</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p13308175419438"><a name="p13308175419438"></a><a name="p13308175419438"></a>Provides a high-performance Adam optimizer implemented through tensor fusion. The core functionality is compatible with <code>torch.optim.Adam</code>.</p>
</td>
</tr>
<tr id="row172915513437"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p32915118430"><a name="p32915118430"></a><a name="p32915118430"></a><a href="torch_npu-optim-NpuFusedAdamW.md">torch_npu.optim.NpuFusedAdamW</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p47811347134517"><a name="p47811347134517"></a><a name="p47811347134517"></a>Provides a high-performance Adam optimizer implemented through tensor fusion. The core functionality is compatible with <code>torch.optim.Adam</code>.</p>
</td>
</tr>
<tr id="row14291551134320"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p4295511435"><a name="p4295511435"></a><a name="p4295511435"></a><a href="torch_npu-optim-NpuFusedAdamP.md">torch_npu.optim.NpuFusedAdamP</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p519124035115"><a name="p519124035115"></a><a name="p519124035115"></a>Provides a high-performance AdamP optimizer implemented through tensor fusion.</p>
</td>
</tr>
<tr id="row19691194544319"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p1769113458436"><a name="p1769113458436"></a><a name="p1769113458436"></a><a href="torch_npu-optim-NpuFusedBertAdam.md">torch_npu.optim.NpuFusedBertAdam</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p20345174819546"><a name="p20345174819546"></a><a name="p20345174819546"></a>Provides a high-performance BertAdam optimizer implemented through tensor fusion.</p>
</td>
</tr>
<tr id="row136911045164315"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p196911145184314"><a name="p196911145184314"></a><a name="p196911145184314"></a><a href="torch_npu-optim-NpuFusedRMSprop.md">torch_npu.optim.NpuFusedRMSprop</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p142403585617"><a name="p142403585617"></a><a name="p142403585617"></a>Provides a high-performance RMSprop optimizer implemented through tensor fusion. The core functionality is compatible with <code>torch.optim.RMSprop</code>.</p>
</td>
</tr>
<tr id="row7691204524312"><td class="cellrowborder" valign="top" width="38.34%" headers="mcps1.2.3.1.1 "><p id="p106921945194318"><a name="p106921945194318"></a><a name="p106921945194318"></a><a href="torch_npu-optim-NpuFusedRMSpropTF.md">torch_npu.optim.NpuFusedRMSpropTF</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.660000000000004%" headers="mcps1.2.3.1.2 "><p id="p1969217454431"><a name="p1969217454431"></a><a name="p1969217454431"></a>Provides a high-performance RMSpropTF optimizer implemented through tensor fusion. The core functionality is compatible with <code>torch.optim.RMSprop</code>.</p>
</td>
</tr>
</tbody>
</table>
