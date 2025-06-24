# 废弃API列表

**表1** 废弃API列表

<a name="table5311174145516"></a>
<table><thead align="left"><tr id="row2311541105517"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p1631184120559"><a name="p1631184120559"></a><a name="p1631184120559"></a>废弃API名称</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p15311104185510"><a name="p15311104185510"></a><a name="p15311104185510"></a>替换说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17311114115553"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p153112414559"><a name="p153112414559"></a><a name="p153112414559"></a><a href="（beta）torch_npu-copy_memory_.md">torch_npu.copy_memory_</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p186351481316"><a name="p186351481316"></a><a name="p186351481316"></a>该接口计划废弃，可以使用torch.Tensor.copy_接口进行替换。</p>
</td>
</tr>
<tr id="row19311164145515"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p8311124175514"><a name="p8311124175514"></a><a name="p8311124175514"></a><a href="（beta）torch_npu-empty_with_format.md">torch_npu.empty_with_format</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p129949526516"><a name="p129949526516"></a><a name="p129949526516"></a>该接口计划废弃，可以使用torch.empty接口进行替换。</p>
</td>
</tr>
<tr id="row18312341155517"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p173125413551"><a name="p173125413551"></a><a name="p173125413551"></a><a href="（beta）torch_npu-npu_apply_adam.md">torch_npu.npu_apply_adam</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1899612582515"><a name="p1899612582515"></a><a name="p1899612582515"></a>该接口计划废弃，可以使用torch.optim.Adam或torch.optim.adam接口进行替换。</p>
</td>
</tr>
<tr id="row7674950125010"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p367510503501"><a name="p367510503501"></a><a name="p367510503501"></a><a href="（beta）torch_npu-npu_broadcast.md">torch_npu.npu_broadcast</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p121121665216"><a name="p121121665216"></a><a name="p121121665216"></a>该接口计划废弃，可以使用torch.broadcast_to接口进行替换。</p>
</td>
</tr>
<tr id="row183121141175518"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2031274114556"><a name="p2031274114556"></a><a name="p2031274114556"></a><a href="（beta）torch_npu-npu_conv_transpose2d.md">torch_npu.npu_conv_transpose2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p8312194165516"><a name="p8312194165516"></a><a name="p8312194165516"></a>该接口计划废弃，可以使用torch.nn.functional.conv_transpose2d接口进行替换。</p>
</td>
</tr>
<tr id="row123123412554"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9312104111558"><a name="p9312104111558"></a><a name="p9312104111558"></a><a href="（beta）torch_npu-npu_conv2d.md">torch_npu.npu_conv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1242918202521"><a name="p1242918202521"></a><a name="p1242918202521"></a>该接口计划废弃，可以使用torch.nn.functional.conv2d接口进行替换。</p>
</td>
</tr>
<tr id="row12312741155519"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p83121841105519"><a name="p83121841105519"></a><a name="p83121841105519"></a><a href="（beta）torch_npu-npu_convolution.md">torch_npu.npu_convolution</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p9312041115515"><a name="p9312041115515"></a><a name="p9312041115515"></a>该接口计划废弃，可以使用torch.nn.functional.conv2d、torch.nn.functional.conv3d或torch._C._nn.slow_conv3d接口进行替换。</p>
</td>
</tr>
<tr id="row78665276418"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p48674279413"><a name="p48674279413"></a><a name="p48674279413"></a><a href="（beta）torch_npu-npu_convolution_transpose.md">torch_npu.npu_convolution_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p146301141145217"><a name="p146301141145217"></a><a name="p146301141145217"></a>该接口计划废弃，可以使用torch.nn.functional.conv_transpose2d或torch.nn.functional.conv_transpose3d接口进行替换。</p>
</td>
</tr>
<tr id="row48671427164110"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p8867627114111"><a name="p8867627114111"></a><a name="p8867627114111"></a><a href="（beta）torch_npu-npu_dtype_cast.md">torch_npu.npu_dtype_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15123104845216"><a name="p15123104845216"></a><a name="p15123104845216"></a>该接口计划废弃，可以使用torch.to接口进行替换。</p>
</td>
</tr>
<tr id="row186752718417"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4868162714410"><a name="p4868162714410"></a><a name="p4868162714410"></a><a href="（beta）torch_npu-npu_gru.md">torch_npu.npu_gru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p45973015310"><a name="p45973015310"></a><a name="p45973015310"></a>该接口计划废弃，可以使用torch.gru接口进行替换。</p>
</td>
</tr>
<tr id="row6868202716414"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p16868727194118"><a name="p16868727194118"></a><a name="p16868727194118"></a><a href="（beta）torch_npu-npu_layer_norm_eval.md">torch_npu.npu_layer_norm_eval</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p186842764114"><a name="p186842764114"></a><a name="p186842764114"></a>该接口计划废弃，可以使用torch.nn.functional.layer_norm接口进行替换。</p>
</td>
</tr>
<tr id="row786882717416"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1586882734118"><a name="p1586882734118"></a><a name="p1586882734118"></a><a href="（beta）torch_npu-npu_min.md">torch_npu.npu_min</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p547813112537"><a name="p547813112537"></a><a name="p547813112537"></a>该接口计划废弃，可以使用torch.min接口进行替换。</p>
</td>
</tr>
<tr id="row9330173494118"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p933013416416"><a name="p933013416416"></a><a name="p933013416416"></a><a href="（beta）torch_npu-npu_mish.md">torch_npu.npu_mish</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p202681337195317"><a name="p202681337195317"></a><a name="p202681337195317"></a>该接口计划废弃，可以使用torch.nn.functional.mish接口进行替换。</p>
</td>
</tr>
<tr id="row0331103454113"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p19331734194116"><a name="p19331734194116"></a><a name="p19331734194116"></a><a href="（beta）torch_npu-npu_nms_rotated.md">torch_npu.npu_nms_rotated</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p208131543205314"><a name="p208131543205314"></a><a name="p208131543205314"></a>该接口计划废弃，可以参考<a href="https://gitee.com/ascend/pytorch/blob/v2.1.0-6.0.rc1/test/network_ops/test_nms_rotated.py" target="_blank" rel="noopener noreferrer">小算子拼接方案</a>进行替换。</p>
</td>
</tr>
<tr id="row43319341418"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p63313347411"><a name="p63313347411"></a><a name="p63313347411"></a><a href="（beta）torch_npu-npu_ptiou.md">torch_npu.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p143099280139"><a name="p143099280139"></a><a name="p143099280139"></a>该接口计划废弃，可以使用torch_npu.npu_iou接口进行替换。</p>
</td>
</tr>
<tr id="row333133464112"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1533111343412"><a name="p1533111343412"></a><a name="p1533111343412"></a><a href="（beta）torch_npu-npu_reshape.md">torch_npu.npu_reshape</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p11331034154114"><a name="p11331034154114"></a><a name="p11331034154114"></a>该接口计划废弃，可以使用torch.reshape接口进行替换。</p>
</td>
</tr>
<tr id="row1433223417411"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p183324349416"><a name="p183324349416"></a><a name="p183324349416"></a><a href="（beta）torch_npu-npu_silu.md">torch_npu.npu_silu</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p11526133155418"><a name="p11526133155418"></a><a name="p11526133155418"></a>该接口计划废弃，可以使用torch.nn.functional.silu接口进行替换。</p>
</td>
</tr>
<tr id="row13332734114113"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p183327342419"><a name="p183327342419"></a><a name="p183327342419"></a><a href="（beta）torch_npu-npu_sort_v2.md">torch_npu.npu_sort_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p128654045419"><a name="p128654045419"></a><a name="p128654045419"></a>该接口计划废弃，可以使用torch.sort接口进行替换。</p>
</td>
</tr>
<tr id="row533243424115"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p183321434184119"><a name="p183321434184119"></a><a name="p183321434184119"></a><a href="（beta）torch_npu-one_.md">torch_npu.one_</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1533283474110"><a name="p1533283474110"></a><a name="p1533283474110"></a>该接口计划废弃，可以使用torch.fill_或torch.ones_like接口进行替换。</p>
</td>
</tr>
<tr id="row14332534124118"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9332173420417"><a name="p9332173420417"></a><a name="p9332173420417"></a><a href="（beta）torch_npu-contrib-DCNv2.md">torch_npu.contrib.DCNv2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16651085550"><a name="p16651085550"></a><a name="p16651085550"></a>该接口计划废弃，可以使用torch_npu.contrib.ModulationDeformCon接口进行替换。</p>
</td>
</tr>
<tr id="row2333123474117"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p733363404114"><a name="p733363404114"></a><a name="p733363404114"></a><a href="（beta）torch_npu-contrib-BiLSTM.md">torch_npu.contrib.BiLSTM</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p191653147552"><a name="p191653147552"></a><a name="p191653147552"></a>该接口计划废弃，可以参考<a href="https://gitee.com/ascend/ModelZoo-PyTorch/blob/732cb7fc5ab59249ae62a905c0d43400a8250da7/PyTorch/contrib/audio/deepspeech/deepspeech_pytorch/bidirectional_lstm.py#L18" target="_blank" rel="noopener noreferrer">小算子拼接方案</a>进行替换。</p>
</td>
</tr>
<tr id="row1846663914412"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15466439164114"><a name="p15466439164114"></a><a name="p15466439164114"></a><a href="（beta）torch_npu-contrib-Swish.md">torch_npu.contrib.Swish</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p74504207559"><a name="p74504207559"></a><a name="p74504207559"></a>该接口计划废弃，可以使用torch_npu.contrib.ModulationDeformCon接口进行替换。</p>
</td>
</tr>
<tr id="row13466103954110"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5467143917412"><a name="p5467143917412"></a><a name="p5467143917412"></a><a href="（beta）torch_npu-contrib-npu_giou.md">torch_npu.contrib.npu_giou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3480527105515"><a name="p3480527105515"></a><a name="p3480527105515"></a>该接口计划废弃，可以使用torch_npu.npu_giou接口进行替换。</p>
</td>
</tr>
<tr id="row746723920415"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p104671339184114"><a name="p104671339184114"></a><a name="p104671339184114"></a><a href="（beta）torch_npu-contrib-npu_ptiou.md">torch_npu.contrib.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p119998332552"><a name="p119998332552"></a><a name="p119998332552"></a>该接口计划废弃，可以使用torch_npu.npu_iou接口进行替换。</p>
</td>
</tr>
<tr id="row9467103913412"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1946712394415"><a name="p1946712394415"></a><a name="p1946712394415"></a><a href="（beta）torch_npu-contrib-npu_iou.md">torch_npu.contrib.npu_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1261323935510"><a name="p1261323935510"></a><a name="p1261323935510"></a>该接口计划废弃，可以使用torch_npu.npu_iou接口进行替换。</p>
</td>
</tr>
<tr id="row174671139164110"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p84677395415"><a name="p84677395415"></a><a name="p84677395415"></a><a href="（beta）torch_npu-contrib-function-npu_diou.md">torch_npu.contrib.function.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3545164812557"><a name="p3545164812557"></a><a name="p3545164812557"></a>该接口计划废弃，可以使用torch_npu.npu_diou接口进行替换。</p>
</td>
</tr>
<tr id="row13467173915415"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p24671739124114"><a name="p24671739124114"></a><a name="p24671739124114"></a><a href="（beta）torch_npu-contrib-function-npu_ciou.md">torch_npu.contrib.function.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1561015645519"><a name="p1561015645519"></a><a name="p1561015645519"></a>该接口计划废弃，可以使用torch_npu.npu_ciou接口进行替换。</p>
</td>
</tr>
<tr id="row046716399413"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p164671839164116"><a name="p164671839164116"></a><a name="p164671839164116"></a><a href="（beta）torch_npu-contrib-module-Mish.md">torch_npu.contrib.module.Mish</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p763677195619"><a name="p763677195619"></a><a name="p763677195619"></a>该接口计划废弃，可以使用torch.nn.Mish接口进行替换。</p>
</td>
</tr>
<tr id="row124676390414"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7467539124114"><a name="p7467539124114"></a><a name="p7467539124114"></a><a href="（beta）torch_npu-contrib-module-SiLU.md">torch_npu.contrib.module.SiLU</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10467103919418"><a name="p10467103919418"></a><a name="p10467103919418"></a>该接口计划废弃，可以使用torch.nn.SiLU接口进行替换。</p>
</td>
</tr>
<tr id="row11468103954112"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p154681639164115"><a name="p154681639164115"></a><a name="p154681639164115"></a><a href="（beta）torch_npu-contrib-module-FusedColorJitter.md">torch_npu.contrib.module.FusedColorJitter</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p918621912569"><a name="p918621912569"></a><a name="p918621912569"></a>该接口计划废弃，可以使用torchvision.transforms.ColorJitter接口进行替换。</p>
</td>
</tr>
<tr id="row44381047182110"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p147111718151717"><a name="p147111718151717"></a><a name="p147111718151717"></a><a href="torch_npu-contrib-module-LinearA8W8Quant.md">torch_npu.contrib.module.LinearA8W8Quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p12865714111912"><a name="p12865714111912"></a><a name="p12865714111912"></a>该接口计划废弃，可以使用torch_npu.contrib.module.LinearQuant接口进行替换。</p>
</td>
</tr>
<tr id="row1597725217179"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="（beta）torch_npu-contrib-npu_fused_attention_with_layernorm.md">torch_npu.contrib.npu_fused_attention_with_layernorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p169771352131719"><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，可以使用torch_npu.npu_fusion_attention与torch.nn.LayerNorm接口进行替换。</p>
</td>
</tr>
</tbody>
</table>

