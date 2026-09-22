# 废弃API列表

**表1** 废弃API列表

<a name="table5311174145516"></a>
<table><thead align="left"><tr id="row2311541105517"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p1631184120559"><a name="p1631184120559"></a><a name="p1631184120559"></a>废弃API名称</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p15311104185510"><a name="p15311104185510"></a><a name="p15311104185510"></a>替换说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row0331103454113"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p19331734194116"><a name="p19331734194116"></a><a name="p19331734194116"></a><a href="./torch_npu/（beta）torch_npu-npu_nms_rotated.md">torch_npu.npu_nms_rotated</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p208131543205314"><a name="p208131543205314"></a><a name="p208131543205314"></a>该接口计划废弃，可以参考<a href="https://gitcode.com/Ascend/op-plugin/blob/master/test/test_base_ops/test_nms_rotated.py" target="_blank" rel="noopener noreferrer">小算子拼接方案</a>进行替换。</p>
</td>
</tr>
<tr id="row43319341418"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p63313347411"><a name="p63313347411"></a><a name="p63313347411"></a><a href="./torch_npu/（beta）torch_npu-npu_ptiou.md">torch_npu.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p143099280139"><a name="p143099280139"></a><a name="p143099280139"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr id="row2333123474117"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p733363404114"><a name="p733363404114"></a><a name="p733363404114"></a><a href="./torch_npu-contrib/（beta）torch_npu-contrib-BiLSTM.md">torch_npu.contrib.BiLSTM</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p191653147552"><a name="p191653147552"></a><a name="p191653147552"></a>该接口计划废弃，可以参考<a href="https://gitee.com/ascend/ModelZoo-PyTorch/blob/732cb7fc5ab59249ae62a905c0d43400a8250da7/PyTorch/contrib/audio/deepspeech/deepspeech_pytorch/bidirectional_lstm.py#L18" target="_blank" rel="noopener noreferrer">小算子拼接方案</a>进行替换。</p>
</td>
</tr>
<tr id="row13466103954110"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5467143917412"><a name="p5467143917412"></a><a name="p5467143917412"></a><a href="./torch_npu-contrib/（beta）torch_npu-contrib-npu_giou.md">torch_npu.contrib.npu_giou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3480527105515"><a name="p3480527105515"></a><a name="p3480527105515"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr id="row746723920415"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p104671339184114"><a name="p104671339184114"></a><a name="p104671339184114"></a><a href="./torch_npu-contrib/（beta）torch_npu-contrib-npu_ptiou.md">torch_npu.contrib.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p119998332552"><a name="p119998332552"></a><a name="p119998332552"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr id="row9467103913412"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1946712394415"><a name="p1946712394415"></a><a name="p1946712394415"></a><a href="./torch_npu-contrib/（beta）torch_npu-contrib-npu_iou.md">torch_npu.contrib.npu_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1261323935510"><a name="p1261323935510"></a><a name="p1261323935510"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr id="row174671139164110"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p84677395415"><a name="p84677395415"></a><a name="p84677395415"></a><a href="./torch_npu-contrib/（beta）torch_npu-contrib-function-npu_diou.md">torch_npu.contrib.function.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3545164812557"><a name="p3545164812557"></a><a name="p3545164812557"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr id="row13467173915415"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p24671739124114"><a name="p24671739124114"></a><a name="p24671739124114"></a><a href="./torch_npu-contrib/（beta）torch_npu-contrib-function-npu_ciou.md">torch_npu.contrib.function.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1561015645519"><a name="p1561015645519"></a><a name="p1561015645519"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_conv3d.md">torch_npu.npu_conv3d</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，可以使用`torch.nn.functional.conv3d`接口进行替换。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_bmmV2.md">torch_npu.npu_bmmV2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，其内部通过`torch.view`/`torch.Tensor.expand`对输入张量进行形状变换（1D扩展、batch维度广播），再调用`torch.bmm`等价操作（BatchMatMul），最后`torch.view`至目标形状。可使用`torch.bmm`和`Tensor.view()`接口进行替换。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_confusion_transpose.md">torch_npu.npu_confusion_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，可以使用`Tensor.view()`和`torch.permute`接口进行替换。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu-npu/torch_npu-npu-ExternalEvent().reset().md">torch_npu-npu-ExternalEvent().reset()</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，torch_npu.npu.ExternalEvent().wait()会自动复位Event，不推荐调用本接口手动复位Event。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_alloc_float_status.md">torch_npu.npu_alloc_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_anchor_response_flags.md">torch_npu.npu_anchor_response_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_batch_nms.md">torch_npu.npu_batch_nms</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_bert_apply_adam.md">torch_npu.npu_bert_apply_adam</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_bounding_box_decode.md">torch_npu.npu_bounding_box_decode</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_bounding_box_encode.md">torch_npu.npu_bounding_box_encode</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_ciou.md">torch_npu.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_clear_float_status.md">torch_npu.npu_clear_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_diou.md">torch_npu.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_get_float_status.md">torch_npu.npu_get_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_giou.md">torch_npu.npu_giou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_grid_assign_positive.md">torch_npu.npu_grid_assign_positive</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_indexing.md">torch_npu.npu_indexing</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_iou.md">torch_npu.npu_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_nms_v4.md">torch_npu.npu_nms_v4</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_nms_with_mask.md">torch_npu.npu_nms_with_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_one_hot.md">torch_npu.npu_one_hot</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_pad.md">torch_npu.npu_pad</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch.nn.functional.pad。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_ps_roi_pooling.md">torch_npu.npu_ps_roi_pooling</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_random_choice_with_mask.md">torch_npu.npu_random_choice_with_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_rotated_iou.md">torch_npu.npu_rotated_iou
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_rotated_overlaps.md">torch_npu.npu_rotated_overlaps
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_sign_bits_pack.md">torch_npu.npu_sign_bits_pack
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_sign_bits_unpack.md">torch_npu.npu_sign_bits_unpack
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_slice.md">torch_npu.npu_slice
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch.slice。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_softmax_cross_entropy_with_logits.md">torch_npu.npu_softmax_cross_entropy_with_logits
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_transpose.md">torch_npu.npu_transpose
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch.permute。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_yolo_boxes_encode.md">torch_npu.npu_yolo_boxes_encode
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_fused_attention_score.md">torch_npu.npu_fused_attention_score
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch_npu.npu_fusion_attention。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_multi_head_attention.md">torch_npu.npu_multi_head_attention
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/（beta）torch_npu-npu_dropout_with_add_softmax.md">torch_npu.npu_dropout_with_add_softmax
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu/torch_npu-npu_mla_prolog.md">torch_npu.npu_mla_prolog
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu-contrib/（beta）torch_npu-contrib-function-dropout_with_byte_mask.md">torch_npu.contrib.function.dropout_with_byte_mask
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11977252111717"><a name="p11977252111717"></a><a name="p11977252111717"></a><a href="./torch_npu-contrib/（beta）torch_npu-contrib-module-npu_modules-DropoutWithByteMask.md">torch_npu.contrib.module.npu_modules.DropoutWithByteMask
</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p><a name="p169771352131719"></a><a name="p169771352131719"></a>该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。</p>
</td>
</tr>
</tbody>
</table>
