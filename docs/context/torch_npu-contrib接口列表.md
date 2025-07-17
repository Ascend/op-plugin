# torch_npu.contrib接口列表

本章节包含常用亲和库接口，提供模型中常用的组合类接口，无需自行完成接口或导入第三方库。

**表1** torch_npu.contrib API

<a name="table154941337155512"></a>
<table><thead align="left"><tr id="row19494173795513"><th class="cellrowborder" valign="top" width="22.84%" id="mcps1.2.4.1.1"><p id="p174941237115510"><a name="p174941237115510"></a><a name="p174941237115510"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="18.38%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p57415312020"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p57415312020"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p57415312020"></a>原生函数/参考链接</p>
</th>
<th class="cellrowborder" valign="top" width="58.78%" id="mcps1.2.4.1.3"><p id="p1749403710555"><a name="p1749403710555"></a><a name="p1749403710555"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row9494183725519"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p3494193718552"><a name="p3494193718552"></a><a name="p3494193718552"></a><a href="（beta）torch_npu-contrib-npu_fused_attention_with_layernorm.md">（beta）torch_npu.contrib.npu_fused_attention_with_layernorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p243410273524"><a name="p243410273524"></a><a name="p243410273524"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p6494173775514"><a name="p6494173775514"></a><a name="p6494173775514"></a>bert自注意力与前层规范的融合实现。</p>
</td>
</tr>
<tr id="row20494037175516"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1349410376553"><a name="p1349410376553"></a><a name="p1349410376553"></a><a href="（beta）torch_npu-contrib-npu_fused_attention.md">（beta）torch_npu.contrib.npu_fused_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p143412710527"><a name="p143412710527"></a><a name="p143412710527"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1549443765519"><a name="p1549443765519"></a><a name="p1549443765519"></a>bert自注意力的融合实现。</p>
</td>
</tr>
<tr id="row52986599558"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p19298185917556"><a name="p19298185917556"></a><a name="p19298185917556"></a><a href="（beta）torch_npu-contrib-Prefetcher.md">（beta）torch_npu.contrib.Prefetcher</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1843432719523"><a name="p1843432719523"></a><a name="p1843432719523"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1829817598554"><a name="p1829817598554"></a><a name="p1829817598554"></a>NPU设备上使用的预取程序。</p>
</td>
</tr>
<tr id="row929810596554"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p14298175911553"><a name="p14298175911553"></a><a name="p14298175911553"></a><a href="（beta）torch_npu-contrib-DCNv2.md">（beta）torch_npu.contrib.DCNv2</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p14344279524"><a name="p14344279524"></a><a name="p14344279524"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p8298559105514"><a name="p8298559105514"></a><a name="p8298559105514"></a>应用基于NPU的调制可变形2D卷积操作。ModulationDeformConv的实现主要是基于mmcv的实现进行设计和重构。</p>
</td>
</tr>
<tr id="row1829825935515"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p829895965516"><a name="p829895965516"></a><a name="p829895965516"></a><a href="（beta）torch_npu-contrib-BiLSTM.md">（beta）torch_npu.contrib.BiLSTM</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p164341274524"><a name="p164341274524"></a><a name="p164341274524"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p72981559175518"><a name="p72981559175518"></a><a name="p72981559175518"></a>将NPU兼容的双向LSTM操作应用于输入序列。</p>
</td>
</tr>
<tr id="row182981759195519"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p72981059205511"><a name="p72981059205511"></a><a name="p72981059205511"></a><a href="（beta）torch_npu-contrib-Swish.md">（beta）torch_npu.contrib.Swish</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1543482795211"><a name="p1543482795211"></a><a name="p1543482795211"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p182983598556"><a name="p182983598556"></a><a name="p182983598556"></a>应用基于NPU的Sigmoid线性单元（SiLU）函数，按元素方向。SiLU函数也称为swish函数。</p>
</td>
</tr>
<tr id="row152981459165519"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p729815915510"><a name="p729815915510"></a><a name="p729815915510"></a><a href="（beta）torch_npu-contrib-NpuFairseqDropout.md">（beta）torch_npu.contrib.NpuFairseqDropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p144345274522"><a name="p144345274522"></a><a name="p144345274522"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p02983590551"><a name="p02983590551"></a><a name="p02983590551"></a>在NPU设备上使用FairseqDropout。</p>
</td>
</tr>
<tr id="row13298115935518"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p20298145925514"><a name="p20298145925514"></a><a name="p20298145925514"></a><a href="（beta）torch_npu-contrib-npu_giou.md">（beta）torch_npu.contrib.npu_giou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1434122714521"><a name="p1434122714521"></a><a name="p1434122714521"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p16299205915558"><a name="p16299205915558"></a><a name="p16299205915558"></a>提供NPU版本的GIoU计算接口。</p>
</td>
</tr>
<tr id="row22998593551"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1129955913553"><a name="p1129955913553"></a><a name="p1129955913553"></a><a href="（beta）torch_npu-contrib-npu_ptiou.md">（beta）torch_npu.contrib.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1843472775216"><a name="p1843472775216"></a><a name="p1843472775216"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p529965975516"><a name="p529965975516"></a><a name="p529965975516"></a>提供NPU版本的PTIoU计算操作。计算时不会为重叠区域添加极小值。</p>
</td>
</tr>
<tr id="row122991859185516"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1299115913551"><a name="p1299115913551"></a><a name="p1299115913551"></a><a href="（beta）torch_npu-contrib-npu_iou.md">（beta）torch_npu.contrib.npu_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1543413272524"><a name="p1543413272524"></a><a name="p1543413272524"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p529945916559"><a name="p529945916559"></a><a name="p529945916559"></a>提供NPU版本的IoU计算操作。计算时会为重叠区域添加极小值，避免除零问题。</p>
</td>
</tr>
<tr id="row1299155912553"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p162992059135511"><a name="p162992059135511"></a><a name="p162992059135511"></a><a href="（beta）torch_npu-contrib-function-fuse_add_softmax_dropout.md">（beta）torch_npu.contrib.function.fuse_add_softmax_dropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p77473132012"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p77473132012"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p77473132012"></a><a href="https://gitee.com/link?target=https://github.com/huggingface/transformers/blob/7999ec125fc31428ed6879bf01bb013483daf704/src/transformers/models/bert/modeling_bert.py#L346" target="_blank" rel="noopener noreferrer">self.dropout()/nn.functional.softmax()/torch.add</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1529917598550"><a name="p1529917598550"></a><a name="p1529917598550"></a>使用NPU自定义算子替换原生写法，以提高性能。</p>
</td>
</tr>
<tr id="row10299659105519"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p0299165913557"><a name="p0299165913557"></a><a name="p0299165913557"></a><a href="（beta）torch_npu-contrib-function-npu_diou.md">（beta）torch_npu.contrib.function.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p137511352014"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p137511352014"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p137511352014"></a><a href="https://gitee.com/link?target=https://arxiv.org/abs/1902.09630" target="_blank" rel="noopener noreferrer">def bboexs_diou()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p10299559135513"><a name="p10299559135513"></a><a name="p10299559135513"></a>应用基于NPU的DIoU操作。考虑到目标之间距离，以及距离和范围的重叠率，不同目标或边界需趋于稳定。</p>
</td>
</tr>
<tr id="row42991659205514"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p229916594553"><a name="p229916594553"></a><a name="p229916594553"></a><a href="（beta）torch_npu-contrib-function-npu_ciou.md">（beta）torch_npu.contrib.function.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p67513316205"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p67513316205"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p67513316205"></a><a href="https://gitee.com/link?target=https://arxiv.org/abs/1902.09630" target="_blank" rel="noopener noreferrer">def bboexs_giou()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1929945945518"><a name="p1929945945518"></a><a name="p1929945945518"></a>应用基于NPU的CIoU操作。在DIoU的基础上增加了penalty item，并propose CIoU。</p>
</td>
</tr>
<tr id="row181123135613"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p381113175618"><a name="p381113175618"></a><a name="p381113175618"></a><a href="（beta）torch_npu-contrib-module-NpuCachedDropout.md">（beta）torch_npu.contrib.module.NpuCachedDropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1434202785218"><a name="p1434202785218"></a><a name="p1434202785218"></a><a href="https://gitee.com/link?target=https://github.com/facebookresearch/fairseq/blob/e0884db9a7ce83670e21af39bf785b616ce5e3e3/fairseq/modules/fairseq_dropout.py#L16" target="_blank" rel="noopener noreferrer">class FairseqDropout()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p118111934564"><a name="p118111934564"></a><a name="p118111934564"></a>在NPU设备上使用FairseqDropout。</p>
</td>
</tr>
<tr id="row1681119316568"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p2811133145615"><a name="p2811133145615"></a><a name="p2811133145615"></a><a href="（beta）torch_npu-contrib-module-MultiheadAttention.md">（beta）torch_npu.contrib.module.MultiheadAttention</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p19759312016"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p19759312016"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p19759312016"></a><a href="https://gitee.com/link?target=https://github.com/facebookresearch/fairseq/blob/e0884db9a7ce83670e21af39bf785b616ce5e3e3/fairseq/modules/multihead_attention.py#L64" target="_blank" rel="noopener noreferrer">class MultiheadAttention()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1981213335616"><a name="p1981213335616"></a><a name="p1981213335616"></a>Multi-head attention。</p>
</td>
</tr>
<tr id="row178121437561"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p18127335610"><a name="p18127335610"></a><a name="p18127335610"></a><a href="（beta）torch_npu-contrib-function-npu_single_level_responsible_flags.md">（beta）torch_npu.contrib.function.npu_single_level_responsible_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p16435027155218"><a name="p16435027155218"></a><a name="p16435027155218"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L831" target="_blank" rel="noopener noreferrer">def single_level_responsible_flags()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p148121365612"><a name="p148121365612"></a><a name="p148121365612"></a>使用NPU OP在单个特征图中生成锚点的responsible flags。</p>
</td>
</tr>
<tr id="row188123355615"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p4812163185612"><a name="p4812163185612"></a><a name="p4812163185612"></a><a href="（beta）torch_npu-contrib-function-npu_bbox_coder_encode_xyxy2xywh.md">（beta）torch_npu.contrib.function.npu_bbox_coder_encode_xyxy2xywh</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p16767332017"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p16767332017"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p16767332017"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/yolo_bbox_coder.py#L27" target="_blank" rel="noopener noreferrer">def encode()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001606524122_p189551963617"><a name="zh-cn_topic_0000001606524122_p189551963617"></a><a name="zh-cn_topic_0000001606524122_p189551963617"></a>应用基于NPU的bbox格式编码操作，将格式从xyxy编码为xywh。</p>
</td>
</tr>
<tr id="row1981223185613"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p98123395618"><a name="p98123395618"></a><a name="p98123395618"></a><a href="（beta）torch_npu-contrib-function-npu_fast_condition_index_put.md">（beta）torch_npu.contrib.function.npu_fast_condition_index_put</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p54351274522"><a name="p54351274522"></a><a name="p54351274522"></a>无原函数，主要功能语句：input1[condition] = value，请查看测试用例。</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p881216375611"><a name="p881216375611"></a><a name="p881216375611"></a>使用NPU亲和写法替换bool型index_put函数中的原生写法。</p>
</td>
</tr>
<tr id="row381214316567"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p188123320561"><a name="p188123320561"></a><a name="p188123320561"></a><a href="（beta）torch_npu-contrib-function-matmul_transpose.md">（beta）torch_npu.contrib.function.matmul_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p12435162714523"><a name="p12435162714523"></a><a name="p12435162714523"></a><a href="https://gitee.com/link?target=https://github.com/huggingface/transformers/blob/d6eeb871706db0d64ab9ffd79f9545d95286b536/src/transformers/models/bert/modeling_bert.py#L331" target="_blank" rel="noopener noreferrer">torch.matmul()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p68129385610"><a name="p68129385610"></a><a name="p68129385610"></a>使用NPU自定义算子替换原生写法，以提高性能。</p>
</td>
</tr>
<tr id="row38121385617"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p108122039566"><a name="p108122039566"></a><a name="p108122039566"></a><a href="（beta）torch_npu-contrib-function-npu_multiclass_nms.md">（beta）torch_npu.contrib.function.npu_multiclass_nms</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p176431206"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p176431206"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p176431206"></a><a href="https://gitee.com/link?target=https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py#L7" target="_blank" rel="noopener noreferrer">def multiclass_nms()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1681263145612"><a name="p1681263145612"></a><a name="p1681263145612"></a>使用NPU API的多类bbox NMS。</p>
</td>
</tr>
<tr id="row381217310561"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p981215315617"><a name="p981215315617"></a><a name="p981215315617"></a><a href="（beta）torch_npu-contrib-function-npu_batched_multiclass_nms.md">（beta）torch_npu.contrib.function.npu_batched_multiclass_nms</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p5777362018"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p5777362018"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p5777362018"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py#L98" target="_blank" rel="noopener noreferrer">def fast_nms()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p181293195617"><a name="p181293195617"></a><a name="p181293195617"></a>使用NPU API的批量多类bbox NMS。</p>
</td>
</tr>
<tr id="row198121835562"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p188125375611"><a name="p188125375611"></a><a name="p188125375611"></a><a href="（beta）torch_npu-contrib-function-roll.md">（beta）torch_npu.contrib.function.roll</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p9771132201"><a name="zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p9771132201"></a><a name="zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p9771132201"></a><a href="https://gitee.com/link?target=https://pytorch.org/docs/stable/generated/torch.roll.html" target="_blank" rel="noopener noreferrer">torch.roll()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p178121939569"><a name="p178121939569"></a><a name="p178121939569"></a>使用NPU亲和写法替换swin-transformer中的原生roll。</p>
</td>
</tr>
<tr id="row481219385615"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1881212345615"><a name="p1881212345615"></a><a name="p1881212345615"></a><a href="（beta）torch_npu-contrib-module-Mish.md">（beta）torch_npu.contrib.module.Mish</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p107710352010"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p107710352010"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p107710352010"></a><a href="https://gitee.com/link?target=https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py" target="_blank" rel="noopener noreferrer">class Mish()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p581212365619"><a name="p581212365619"></a><a name="p581212365619"></a>应用基于NPU的Mish操作。</p>
</td>
</tr>
<tr id="row16812163115611"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1281210314562"><a name="p1281210314562"></a><a name="p1281210314562"></a><a href="（beta）torch_npu-contrib-module-SiLU.md">（beta）torch_npu.contrib.module.SiLU</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p47703172014"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p47703172014"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p47703172014"></a><a href="https://gitee.com/link?target=https://pytorch.org/docs/1.8.1/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU" target="_blank" rel="noopener noreferrer">class SiLu()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001606524122_p196702038144319"><a name="zh-cn_topic_0000001606524122_p196702038144319"></a><a name="zh-cn_topic_0000001606524122_p196702038144319"></a>按元素应用基于NPU的Sigmoid线性单元（SiLU）函数。SiLU函数也称为Swish函数。</p>
</td>
</tr>
<tr id="row1781253125616"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p381217319561"><a name="p381217319561"></a><a name="p381217319561"></a><a href="（beta）torch_npu-contrib-module-ChannelShuffle.md">（beta）torch_npu.contrib.module.ChannelShuffle</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p197893102017"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p197893102017"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p197893102017"></a><a href="https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py#L28" target="_blank" rel="noopener noreferrer">def channel_shuffle()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p2812113175617"><a name="p2812113175617"></a><a name="p2812113175617"></a>应用NPU兼容的通道shuffle操作。为避免NPU上效率不高的连续操作，我们用相同语义重写替换原始操作。以下两个不连续操作已被替换：transpose和chunk。</p>
</td>
</tr>
<tr id="row681318325610"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p13813173125620"><a name="p13813173125620"></a><a name="p13813173125620"></a><a href="（beta）torch_npu-contrib-module-LabelSmoothingCrossEntropy.md">（beta）torch_npu.contrib.module.LabelSmoothingCrossEntropy</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p2435152716524"><a name="p2435152716524"></a><a name="p2435152716524"></a><a href="https://gitee.com/link?target=https://arxiv.org/pdf/1512.00567.pdf" target="_blank" rel="noopener noreferrer">class LabelSmoothingCrossEntropy()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1481373195615"><a name="p1481373195615"></a><a name="p1481373195615"></a>使用NPU API进行LabelSmoothing Cross Entropy。</p>
</td>
</tr>
<tr id="row1681363165615"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p15813735564"><a name="p15813735564"></a><a name="p15813735564"></a><a href="（beta）torch_npu-contrib-module-ModulatedDeformConv.md">（beta）torch_npu.contrib.module.ModulatedDeformConv</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p97983152013"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p97983152013"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p97983152013"></a><a href="https://gitee.com/link?target=https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/modulated_deform_conv.py" target="_blank" rel="noopener noreferrer">class ModulatedDeformConv2dFunciton()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p88131233567"><a name="p88131233567"></a><a name="p88131233567"></a>应用基于NPU的Modulated Deformable 2D卷积操作。</p>
</td>
</tr>
<tr id="row158131836568"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p178131536563"><a name="p178131536563"></a><a name="p178131536563"></a><a href="（beta）torch_npu-contrib-module-NpuDropPath.md">（beta）torch_npu.contrib.module.NpuDropPath</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p27923102016"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p27923102016"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p27923102016"></a><a href="https://gitee.com/link?target=https://github.com/rwightman/pytorch-image-models/blob/e7f0db866412b9ae61332c205270c9fc0ef5083c/timm/models/layers/drop.py#L160" target="_blank" rel="noopener noreferrer">class DropPath()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p38132325610"><a name="p38132325610"></a><a name="p38132325610"></a>使用NPU亲和写法替换swin_transformer.py中的原生Drop路径。丢弃每个样本（应用于residual blocks的主路径）的路径（随机深度）。</p>
</td>
</tr>
<tr id="row148133312561"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p181317318564"><a name="p181317318564"></a><a name="p181317318564"></a><a href="（beta）torch_npu-contrib-module-Focus.md">（beta）torch_npu.contrib.module.Focus</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1843572712527"><a name="p1843572712527"></a><a name="p1843572712527"></a><a href="https://gitee.com/link?target=https://github.com/ultralytics/yolov5/blob/4d05472d2b50108c0fcfe9208d32cb067a6e21b0/models/common.py#L227" target="_blank" rel="noopener noreferrer">class Focus()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p78131334563"><a name="p78131334563"></a><a name="p78131334563"></a>使用NPU亲和写法替换YOLOv5中的原生Focus。</p>
</td>
</tr>
<tr id="row98136312569"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p98131633564"><a name="p98131633564"></a><a name="p98131633564"></a><a href="（beta）torch_npu-contrib-module-PSROIPool.md">（beta）torch_npu.contrib.module.PSROIPool</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p197913362018"><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p197913362018"></a><a name="zh-cn_topic_0000001678826450_zh-cn_topic_0000001606524122_zh-cn_topic_0000001390596206_zh-cn_topic_0000001385999112_p197913362018"></a><a href="https://gitee.com/link?target=https://github.com/RebornL/RFCN-pytorch.1.0/blob/master/lib/model/roi_layers/ps_roi_pool.py" target="_blank" rel="noopener noreferrer">class PSROIPool()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p48133355616"><a name="p48133355616"></a><a name="p48133355616"></a>使用NPU API进行PSROIPool。</p>
</td>
</tr>
<tr id="row92991059205517"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p429945917558"><a name="p429945917558"></a><a name="p429945917558"></a><a href="（beta）torch_npu-contrib-module-ROIAlign.md">（beta）torch_npu.contrib.module.ROIAlign</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1343522765212"><a name="p1343522765212"></a><a name="p1343522765212"></a><a href="https://gitee.com/link?target=https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/roi_align.py#L7" target="_blank" rel="noopener noreferrer">class ROIAlign()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p429975920559"><a name="p429975920559"></a><a name="p429975920559"></a>使用NPU API进行ROIAlign。</p>
</td>
</tr>
<tr id="row029911593559"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p10299759175510"><a name="p10299759175510"></a><a name="p10299759175510"></a><a href="（beta）torch_npu-contrib-module-FusedColorJitter.md">（beta）torch_npu.contrib.module.FusedColorJitter</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001606524122_p8909182712250"><a name="zh-cn_topic_0000001606524122_p8909182712250"></a><a name="zh-cn_topic_0000001606524122_p8909182712250"></a><a href="https://beesbuzz.biz/code/16-hsv-color-transforms" target="_blank" rel="noopener noreferrer">Reference 1</a>或<a href="https://github.com/NVIDIA/DALI/blob/release_v1.15/dali/operators/image/color/color_twist.h#L155" target="_blank" rel="noopener noreferrer">Reference 2</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p102991559115515"><a name="p102991559115515"></a><a name="p102991559115515"></a>随机更改图像的亮度、对比度、饱和度和色调。</p>
</td>
</tr>
<tr id="row4299155965520"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p14299559105517"><a name="p14299559105517"></a><a name="p14299559105517"></a><a href="（beta）torch_npu-contrib-function-npu_bbox_coder_decode_xywh2xyxy.md">（beta）torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001606524122_p13353211123217"><a name="zh-cn_topic_0000001606524122_p13353211123217"></a><a name="zh-cn_topic_0000001606524122_p13353211123217"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L164" target="_blank" rel="noopener noreferrer">def npu_bbox_coder_decode_xywh2xyxy()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p829965916557"><a name="p829965916557"></a><a name="p829965916557"></a>应用基于NPU的bbox格式编码操作，将格式从xywh编码为xyxy。</p>
</td>
</tr>
<tr id="row182991559155515"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p182991959195516"><a name="p182991959195516"></a><a name="p182991959195516"></a><a href="（beta）torch_npu-contrib-function-npu_bbox_coder_encode_yolo.md">（beta）torch_npu.contrib.function.npu_bbox_coder_encode_yolo</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001606524122_p1676145873611"><a name="zh-cn_topic_0000001606524122_p1676145873611"></a><a name="zh-cn_topic_0000001606524122_p1676145873611"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L118" target="_blank" rel="noopener noreferrer">def npu_bbox_coder_encode_yolo()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p62992059175515"><a name="p62992059175515"></a><a name="p62992059175515"></a>使用NPU OP获取将bbox转换为gt_bbox的框回归转换deltas。</p>
</td>
</tr>
<tr id="row1249411377551"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p154944378551"><a name="p154944378551"></a><a name="p154944378551"></a><a href="（beta）torch_npu-contrib-module-npu_modules-DropoutWithByteMask.md">（beta）torch_npu.contrib.module.npu_modules.DropoutWithByteMask</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p543592755219"><a name="p543592755219"></a><a name="p543592755219"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1149463719556"><a name="p1149463719556"></a><a name="p1149463719556"></a>应用NPU兼容的DropoutWithByteMask操作。</p>
</td>
</tr>
<tr id="row2914185425510"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p2091419545552"><a name="p2091419545552"></a><a name="p2091419545552"></a><a href="（beta）torch_npu-contrib-function-dropout_with_byte_mask.md">（beta）torch_npu.contrib.function.dropout_with_byte_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1435227155218"><a name="p1435227155218"></a><a name="p1435227155218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p3914125415512"><a name="p3914125415512"></a><a name="p3914125415512"></a>应用NPU兼容的dropout_with_byte_mask操作，仅支持NPU设备。这个dropout_with_byte_mask方法生成无状态随机uint8掩码，并根据掩码做dropout。</p>
</td>
</tr>
<tr id="row62941778711"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p183268185292"><a name="p183268185292"></a><a name="p183268185292"></a><a href="torch_npu-contrib-module-LinearA8W8Quant.md">torch_npu.contrib.module.LinearA8W8Quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001778938168_p156512056161014"><a name="zh-cn_topic_0000001778938168_p156512056161014"></a><a name="zh-cn_topic_0000001778938168_p156512056161014"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p16295679713"><a name="p16295679713"></a><a name="p16295679713"></a>LinearA8W8Quant是对torch_npu.npu_quant_matmul接口的封装类，完成A8W8量化算子的矩阵乘计算。</p>
</td>
</tr>
<tr id="row4342314612"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p23423113616"><a name="p23423113616"></a><a name="p23423113616"></a><a href="torch_npu-contrib-module-LinearQuant.md">torch_npu.contrib.module.LinearQuant</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p133412311662"><a name="p133412311662"></a><a name="p133412311662"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002045840024_p9744194141510"><a name="zh-cn_topic_0000002045840024_p9744194141510"></a><a name="zh-cn_topic_0000002045840024_p9744194141510"></a>LinearQuant是对torch_npu.npu_quant_matmul接口的封装类，完成A8W8、A4W4量化算子的矩阵乘计算。</p>
</td>
</tr>
<tr id="row3799410478"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p147993102716"><a name="p147993102716"></a><a name="p147993102716"></a><a href="torch_npu-contrib-module-LinearWeightQuant.md">torch_npu.contrib.module.LinearWeightQuant</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p479919101712"><a name="p479919101712"></a><a name="p479919101712"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p197991410478"><a name="p197991410478"></a><a name="p197991410478"></a>LinearWeightQuant是对torch_npu.npu_weight_quant_batchmatmul接口的封装类，完成矩阵乘计算中的weight输入和输出的量化操作，支持pertensor、perchannel、pergroup多场景量化。</p>
</td>
</tr>
<tr id="row5577141319717"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p9734435173615"><a name="p9734435173615"></a><a name="p9734435173615"></a><a href="torch_npu-contrib-module-QuantConv2d.md">torch_npu.contrib.module.QuantConv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p157710131475"><a name="p157710131475"></a><a name="p157710131475"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1057719133712"><a name="p1057719133712"></a><a name="p1057719133712"></a>QuantConv2d是对torch_npu.npu_quant_conv2d接口的封装类，为用户提供Conv2d算子量化相关功能。</p>
</td>
</tr>
</tbody>
</table>

