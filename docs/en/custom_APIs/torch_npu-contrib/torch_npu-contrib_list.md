# torch_npu.contrib APIs

This section describes common affinity library APIs and provides commonly used composite APIs in models. You do not need to implement APIs yourself or import third-party libraries.

**Table 1** torch_npu.contrib APIs

<a name="table154941337155512"></a>
<table><thead align="left"><tr id="row19494173795513"><th class="cellrowborder" valign="top" width="22.84%" id="mcps1.2.4.1.1"><p id="p174941237115510"><a name="p174941237115510"></a><a name="p174941237115510"></a>API</p>
</th>
<th class="cellrowborder" valign="top" width="18.38%" id="mcps1.2.4.1.2"><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p57415312020"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p57415312020"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p57415312020"></a>Native Function/Reference Link</p>
</th>
<th class="cellrowborder" valign="top" width="58.78%" id="mcps1.2.4.1.3"><p id="p1749403710555"><a name="p1749403710555"></a><a name="p1749403710555"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row9494183725519"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p3494193718552"><a name="p3494193718552"></a><a name="p3494193718552"></a><a href="(beta)torch_npu-contrib-npu_fused_attention_with_layernorm.md">(beta) torch_npu.contrib.npu_fused_attention_with_layernorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p243410273524"><a name="p243410273524"></a><a name="p243410273524"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p6494173775514"><a name="p6494173775514"></a><a name="p6494173775514"></a>Fuses BERT self-attention and layer normalization computations.</p>
</td>
</tr>
<tr id="row20494037175516"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1349410376553"><a name="p1349410376553"></a><a name="p1349410376553"></a><a href="(beta)torch_npu-contrib-npu_fused_attention.md">(beta) torch_npu.contrib.npu_fused_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p143412710527"><a name="p143412710527"></a><a name="p143412710527"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1549443765519"><a name="p1549443765519"></a><a name="p1549443765519"></a>Fuses BERT self-attention computations.</p>
</td>
</tr>
<tr id="row52986599558"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p19298185917556"><a name="p19298185917556"></a><a name="p19298185917556"></a><a href="(beta)torch_npu-contrib-Prefetcher.md">(beta) torch_npu.contrib.Prefetcher</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1843432719523"><a name="p1843432719523"></a><a name="p1843432719523"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1829817598554"><a name="p1829817598554"></a><a name="p1829817598554"></a>Provides a data prefetcher on NPU devices.</p>
</td>
</tr>
<tr id="row929810596554"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p14298175911553"><a name="p14298175911553"></a><a name="p14298175911553"></a><a href="(beta)torch_npu-contrib-DCNv2.md">(beta) torch_npu.contrib.DCNv2</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p14344279524"><a name="p14344279524"></a><a name="p14344279524"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p8298559105514"><a name="p8298559105514"></a><a name="p8298559105514"></a>Applies an NPU-based modulated deformable 2D convolution operation. The implementation of <code>ModulationDeformConv</code> is designed and refactored based on MMCV.</p>
</td>
</tr>
<tr id="row1829825935515"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p829895965516"><a name="p829895965516"></a><a name="p829895965516"></a><a href="(beta)torch_npu-contrib-BiLSTM.md">(beta) torch_npu.contrib.BiLSTM</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p164341274524"><a name="p164341274524"></a><a name="p164341274524"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p72981559175518"><a name="p72981559175518"></a><a name="p72981559175518"></a>Applies NPU-compatible bidirectional LSTM operations on input sequences.</p>
</td>
</tr>
<tr id="row182981759195519"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p72981059205511"><a name="p72981059205511"></a><a name="p72981059205511"></a><a href="(beta)torch_npu-contrib-Swish.md">(beta) torch_npu.contrib.Swish</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1543482795211"><a name="p1543482795211"></a><a name="p1543482795211"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p182983598556"><a name="p182983598556"></a><a name="p182983598556"></a>Applies the NPU-based Sigmoid Linear Unit (SiLU) function element-wise. The SiLU function is also known as the Swish function.</p>
</td>
</tr>
<tr id="row152981459165519"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p729815915510"><a name="p729815915510"></a><a name="p729815915510"></a><a href="(beta)torch_npu-contrib-NpuFairseqDropout.md">(beta) torch_npu.contrib.NpuFairseqDropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p144345274522"><a name="p144345274522"></a><a name="p144345274522"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p02983590551"><a name="p02983590551"></a><a name="p02983590551"></a>Executes a <code>FairseqDropout</code> operation on NPU devices.</p>
</td>
</tr>
<tr id="row13298115935518"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p20298145925514"><a name="p20298145925514"></a><a name="p20298145925514"></a><a href="(beta)torch_npu-contrib-npu_giou.md">(beta) torch_npu.contrib.npu_giou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1434122714521"><a name="p1434122714521"></a><a name="p1434122714521"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p16299205915558"><a name="p16299205915558"></a><a name="p16299205915558"></a>Provides an NPU-based Generalized Intersection over Union (GIoU) computation API.</p>
</td>
</tr>
<tr id="row22998593551"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1129955913553"><a name="p1129955913553"></a><a name="p1129955913553"></a><a href="(beta)torch_npu-contrib-npu_ptiou.md">(beta) torch_npu.contrib.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1843472775216"><a name="p1843472775216"></a><a name="p1843472775216"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p529965975516"><a name="p529965975516"></a><a name="p529965975516"></a>Provides the NPU version of PTIoU computation operations. During computation, no minimum value is added to the overlapping area.</p>
</td>
</tr>
<tr id="row122991859185516"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1299115913551"><a name="p1299115913551"></a><a name="p1299115913551"></a><a href="(beta)torch_npu-contrib-npu_iou.md">(beta) torch_npu.contrib.npu_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1543413272524"><a name="p1543413272524"></a><a name="p1543413272524"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p529945916559"><a name="p529945916559"></a><a name="p529945916559"></a>Provides the NPU version of IoU computation operations. During computation, a small value is added to the overlapping area to avoid division-by-zero errors.</p>
</td>
</tr>
<tr id="row1299155912553"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p162992059135511"><a name="p162992059135511"></a><a name="p162992059135511"></a><a href="(beta)torch_npu-contrib-function-fuse_add_softmax_dropout.md">(beta) torch_npu.contrib.function.fuse_add_softmax_dropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p77473132012"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p77473132012"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p77473132012"></a><a href="https://gitee.com/link?target=https://github.com/huggingface/transformers/blob/7999ec125fc31428ed6879bf01bb013483daf704/src/transformers/models/bert/modeling_bert.py#L346" target="_blank" rel="noopener noreferrer">self.dropout()/nn.functional.softmax()/torch.add</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1529917598550"><a name="p1529917598550"></a><a name="p1529917598550"></a>Replaces the native implementation with an NPU custom operator to improve performance.</p>
</td>
</tr>
<tr id="row10299659105519"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p0299165913557"><a name="p0299165913557"></a><a name="p0299165913557"></a><a href="(beta)torch_npu-contrib-function-npu_diou.md">(beta) torch_npu.contrib.function.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p137511352014"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p137511352014"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p137511352014"></a><a href="https://gitee.com/link?target=https://arxiv.org/abs/1902.09630" target="_blank" rel="noopener noreferrer">def bboxes_diou()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p10299559135513"><a name="p10299559135513"></a><a name="p10299559135513"></a>Applies an NPU-based DIoU operation. DIoU considers both the overlap ratio and the distance between target bounding boxes to improve localization stability.</p>
</td>
</tr>
<tr id="row42991659205514"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p229916594553"><a name="p229916594553"></a><a name="p229916594553"></a><a href="(beta)torch_npu-contrib-function-npu_ciou.md">(beta) torch_npu.contrib.function.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p67513316205"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p67513316205"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p67513316205"></a><a href="https://gitee.com/link?target=https://arxiv.org/abs/1902.09630" target="_blank" rel="noopener noreferrer">def bboxes_giou()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1929945945518"><a name="p1929945945518"></a><a name="p1929945945518"></a>Applies an NPU-based CIoU operation. CIoU is formulated by introducing a penalty term to DIoU.</p>
</td>
</tr>
<tr id="row181123135613"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p381113175618"><a name="p381113175618"></a><a name="p381113175618"></a><a href="(beta)torch_npu-contrib-module-NpuCachedDropout.md">(beta) torch_npu.contrib.module.NpuCachedDropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1434202785218"><a name="p1434202785218"></a><a name="p1434202785218"></a><a href="https://gitee.com/link?target=https://github.com/facebookresearch/fairseq/blob/e0884db9a7ce83670e21af39bf785b616ce5e3e3/fairseq/modules/fairseq_dropout.py#L16" target="_blank" rel="noopener noreferrer">class FairseqDropout()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p118111934564"><a name="p118111934564"></a><a name="p118111934564"></a>Executes a <code>FairseqDropout</code> operation on NPU devices.</p>
</td>
</tr>
<tr id="row1681119316568"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p2811133145615"><a name="p2811133145615"></a><a name="p2811133145615"></a><a href="(beta)torch_npu-contrib-module-MultiheadAttention.md">(beta) torch_npu.contrib.module.MultiheadAttention</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p19759312016"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p19759312016"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p19759312016"></a><a href="https://gitee.com/link?target=https://github.com/facebookresearch/fairseq/blob/e0884db9a7ce83670e21af39bf785b616ce5e3e3/fairseq/modules/multihead_attention.py#L64" target="_blank" rel="noopener noreferrer">class MultiheadAttention()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1981213335616"><a name="p1981213335616"></a><a name="p1981213335616"></a>Executes a multi-head attention operation.</p>
</td>
</tr>
<tr id="row178121437561"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p18127335610"><a name="p18127335610"></a><a name="p18127335610"></a><a href="(beta)torch_npu-contrib-function-npu_single_level_responsible_flags.md">(beta) torch_npu.contrib.function.npu_single_level_responsible_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p16435027155218"><a name="p16435027155218"></a><a name="p16435027155218"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L831" target="_blank" rel="noopener noreferrer">def single_level_responsible_flags()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p148121365612"><a name="p148121365612"></a><a name="p148121365612"></a>Uses an NPU operator to generate responsible flags for anchors in a single feature map.</p>
</td>
</tr>
<tr id="row188123355615"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p4812163185612"><a name="p4812163185612"></a><a name="p4812163185612"></a><a href="(beta)torch_npu-contrib-function-npu_bbox_coder_encode_xyxy2xywh.md">(beta) torch_npu.contrib.function.npu_bbox_coder_encode_xyxy2xywh</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p16767332017"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p16767332017"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p16767332017"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/yolo_bbox_coder.py#L27" target="_blank" rel="noopener noreferrer">def encode()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="en-us_topic_0000001606524122_p189551963617"><a name="en-us_topic_0000001606524122_p189551963617"></a><a name="en-us_topic_0000001606524122_p189551963617"></a>Applies an NPU-based bounding box format encoding operation to convert format from <code>xyxy</code> to <code>xywh</code>.</p>
</td>
</tr>
<tr id="row1981223185613"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p98123395618"><a name="p98123395618"></a><a name="p98123395618"></a><a href="(beta)torch_npu-contrib-function-npu_fast_condition_index_put.md">(beta) torch_npu.contrib.function.npu_fast_condition_index_put</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p54351274522"><a name="p54351274522"></a><a name="p54351274522"></a>No native function. The main function statement is <code>input1[condition] = value</code>. For details, see the test case.</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p881216375611"><a name="p881216375611"></a><a name="p881216375611"></a>Replaces the native Boolean <code>index_put</code> implementation with an NPU-optimized implementation.</p>
</td>
</tr>
<tr id="row381214316567"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p188123320561"><a name="p188123320561"></a><a name="p188123320561"></a><a href="(beta)torch_npu-contrib-function-matmul_transpose.md">(beta) torch_npu.contrib.function.matmul_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p12435162714523"><a name="p12435162714523"></a><a name="p12435162714523"></a><a href="https://gitee.com/link?target=https://github.com/huggingface/transformers/blob/d6eeb871706db0d64ab9ffd79f9545d95286b536/src/transformers/models/bert/modeling_bert.py#L331" target="_blank" rel="noopener noreferrer">torch.matmul()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p68129385610"><a name="p68129385610"></a><a name="p68129385610"></a>Replaces the native implementation with an NPU custom operator to improve performance.</p>
</td>
</tr>
<tr id="row38121385617"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p108122039566"><a name="p108122039566"></a><a name="p108122039566"></a><a href="(beta)torch_npu-contrib-function-npu_multiclass_nms.md">(beta) torch_npu.contrib.function.npu_multiclass_nms</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p176431206"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p176431206"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p176431206"></a><a href="https://gitee.com/link?target=https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py#L7" target="_blank" rel="noopener noreferrer">def multiclass_nms()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1681263145612"><a name="p1681263145612"></a><a name="p1681263145612"></a>Uses the NPU API for multi-class bbox NMS.</p>
</td>
</tr>
<tr id="row381217310561"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p981215315617"><a name="p981215315617"></a><a name="p981215315617"></a><a href="(beta)torch_npu-contrib-function-npu_batched_multiclass_nms.md">(beta) torch_npu.contrib.function.npu_batched_multiclass_nms</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p5777362018"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p5777362018"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p5777362018"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py#L98" target="_blank" rel="noopener noreferrer">def fast_nms()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p181293195617"><a name="p181293195617"></a><a name="p181293195617"></a>Uses the NPU API for batch multi-class bbox NMS.</p>
</td>
</tr>
<tr id="row198121835562"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p188125375611"><a name="p188125375611"></a><a name="p188125375611"></a><a href="(beta)torch_npu-contrib-function-roll.md">(beta) torch_npu.contrib.function.roll</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p9771132201"><a name="en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p9771132201"></a><a name="en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p9771132201"></a><a href="https://gitee.com/link?target=https://pytorch.org/docs/stable/generated/torch.roll.html" target="_blank" rel="noopener noreferrer">torch.roll()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p178121939569"><a name="p178121939569"></a><a name="p178121939569"></a>Replaces the native <code>roll</code> operation in Swin Transformer with an NPU-optimized implementation.</p>
</td>
</tr>
<tr id="row481219385615"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1881212345615"><a name="p1881212345615"></a><a name="p1881212345615"></a><a href="(beta)torch_npu-contrib-module-Mish.md">(beta) torch_npu.contrib.module.Mish</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p107710352010"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p107710352010"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p107710352010"></a><a href="https://gitee.com/link?target=https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py" target="_blank" rel="noopener noreferrer">class Mish()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p581212365619"><a name="p581212365619"></a><a name="p581212365619"></a>Applies an NPU-based Mish operation.</p>
</td>
</tr>
<tr id="row16812163115611"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p1281210314562"><a name="p1281210314562"></a><a name="p1281210314562"></a><a href="(beta)torch_npu-contrib-module-SiLU.md">(beta) torch_npu.contrib.module.SiLU</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p47703172014"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p47703172014"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p47703172014"></a><a href="https://gitee.com/link?target=https://pytorch.org/docs/1.8.1/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU" target="_blank" rel="noopener noreferrer">class SiLU()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="en-us_topic_0000001606524122_p196702038144319"><a name="en-us_topic_0000001606524122_p196702038144319"></a><a name="en-us_topic_0000001606524122_p196702038144319"></a>Applies the NPU-based Sigmoid Linear Unit (SiLU) function element-wise. The SiLU function is also known as the Swish function.</p>
</td>
</tr>
<tr id="row1781253125616"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p381217319561"><a name="p381217319561"></a><a name="p381217319561"></a><a href="(beta)torch_npu-contrib-module-ChannelShuffle.md">(beta) torch_npu.contrib.module.ChannelShuffle</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p197893102017"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p197893102017"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p197893102017"></a><a href="https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py#L28" target="_blank" rel="noopener noreferrer">def channel_shuffle()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p2812113175617"><a name="p2812113175617"></a><a name="p2812113175617"></a>Applies an NPU-optimized channel shuffle operation. To avoid inefficient contiguous operations on the NPU, this function rewrites and replaces the original operations with identical semantics. The following two non-contiguous operations are replaced: <code>transpose</code> and <code>chunk</code>.</p>
</td>
</tr>
<tr id="row681318325610"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p13813173125620"><a name="p13813173125620"></a><a name="p13813173125620"></a><a href="(beta)torch_npu-contrib-module-LabelSmoothingCrossEntropy.md">(beta) torch_npu.contrib.module.LabelSmoothingCrossEntropy</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p2435152716524"><a name="p2435152716524"></a><a name="p2435152716524"></a><a href="https://gitee.com/link?target=https://arxiv.org/pdf/1512.00567.pdf" target="_blank" rel="noopener noreferrer">class LabelSmoothingCrossEntropy()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1481373195615"><a name="p1481373195615"></a><a name="p1481373195615"></a>Performs label smoothing cross entropy loss calculation using NPU APIs.</p>
</td>
</tr>
<tr id="row1681363165615"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p15813735564"><a name="p15813735564"></a><a name="p15813735564"></a><a href="(beta)torch_npu-contrib-module-ModulatedDeformConv.md">(beta) torch_npu.contrib.module.ModulatedDeformConv</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p97983152013"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p97983152013"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p97983152013"></a><a href="https://gitee.com/link?target=https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/modulated_deform_conv.py" target="_blank" rel="noopener noreferrer">class ModulatedDeformConv2dFunciton()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p88131233567"><a name="p88131233567"></a><a name="p88131233567"></a>Applies an NPU-based modulated deformable 2D convolution operation.</p>
</td>
</tr>
<tr id="row158131836568"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p178131536563"><a name="p178131536563"></a><a name="p178131536563"></a><a href="(beta)torch_npu-contrib-module-NpuDropPath.md">(beta) torch_npu.contrib.module.NpuDropPath</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p27923102016"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p27923102016"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p27923102016"></a><a href="https://gitee.com/link?target=https://github.com/rwightman/pytorch-image-models/blob/e7f0db866412b9ae61332c205270c9fc0ef5083c/timm/models/layers/drop.py#L160" target="_blank" rel="noopener noreferrer">class DropPath()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p38132325610"><a name="p38132325610"></a><a name="p38132325610"></a>Replaces the native <code>DropPath</code> implementation in <code>swin_transformer.py</code> with an NPU-optimized implementation. It randomly drops the main path (stochastic depth) of each sample within the residual blocks.</p>
</td>
</tr>
<tr id="row148133312561"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p181317318564"><a name="p181317318564"></a><a name="p181317318564"></a><a href="(beta)torch_npu-contrib-module-Focus.md">(beta) torch_npu.contrib.module.Focus</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1843572712527"><a name="p1843572712527"></a><a name="p1843572712527"></a><a href="https://gitee.com/link?target=https://github.com/ultralytics/yolov5/blob/4d05472d2b50108c0fcfe9208d32cb067a6e21b0/models/common.py#L227" target="_blank" rel="noopener noreferrer">class Focus()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p78131334563"><a name="p78131334563"></a><a name="p78131334563"></a>Replaces the native <code>Focus</code> module in YOLOv5 with an NPU-optimized implementation.</p>
</td>
</tr>
<tr id="row98136312569"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p98131633564"><a name="p98131633564"></a><a name="p98131633564"></a><a href="(beta)torch_npu-contrib-module-PSROIPool.md">(beta) torch_npu.contrib.module.PSROIPool</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p197913362018"><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p197913362018"></a><a name="en-us_topic_0000001678826450_en-us_topic_0000001606524122_en-us_topic_0000001390596206_en-us_topic_0000001385999112_p197913362018"></a><a href="https://gitee.com/link?target=https://github.com/RebornL/RFCN-pytorch.1.0/blob/master/lib/model/roi_layers/ps_roi_pool.py" target="_blank" rel="noopener noreferrer">class PSROIPool()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p48133355616"><a name="p48133355616"></a><a name="p48133355616"></a>Performs a position-sensitive region of interest pooling (PSROIPool) operation using the NPU API.</p>
</td>
</tr>
<tr id="row92991059205517"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p429945917558"><a name="p429945917558"></a><a name="p429945917558"></a><a href="(beta)torch_npu-contrib-module-ROIAlign.md">(beta) torch_npu.contrib.module.ROIAlign</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1343522765212"><a name="p1343522765212"></a><a name="p1343522765212"></a><a href="https://gitee.com/link?target=https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/roi_align.py#L7" target="_blank" rel="noopener noreferrer">class ROIAlign()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p429975920559"><a name="p429975920559"></a><a name="p429975920559"></a>Performs a region of interest alignment (ROIAlign) operation using the NPU API.</p>
</td>
</tr>
<tr id="row029911593559"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p10299759175510"><a name="p10299759175510"></a><a name="p10299759175510"></a><a href="(beta)torch_npu-contrib-module-FusedColorJitter.md">(beta) torch_npu.contrib.module.FusedColorJitter</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001606524122_p8909182712250"><a name="en-us_topic_0000001606524122_p8909182712250"></a><a name="en-us_topic_0000001606524122_p8909182712250"></a><a href="https://beesbuzz.biz/code/16-hsv-color-transforms" target="_blank" rel="noopener noreferrer">Reference 1</a> or <a href="https://github.com/NVIDIA/DALI/blob/release_v1.15/dali/operators/image/color/color_twist.h#L155" target="_blank" rel="noopener noreferrer">Reference 2</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p102991559115515"><a name="p102991559115515"></a><a name="p102991559115515"></a>Randomly adjusts the brightness, contrast, saturation, and hue of an image.</p>
</td>
</tr>
<tr id="row4299155965520"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p14299559105517"><a name="p14299559105517"></a><a name="p14299559105517"></a><a href="(beta)torch_npu-contrib-function-npu_bbox_coder_decode_xywh2xyxy.md">(beta) torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001606524122_p13353211123217"><a name="en-us_topic_0000001606524122_p13353211123217"></a><a name="en-us_topic_0000001606524122_p13353211123217"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L164" target="_blank" rel="noopener noreferrer">def npu_bbox_coder_decode_xywh2xyxy()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p829965916557"><a name="p829965916557"></a><a name="p829965916557"></a>Applies an NPU-based bounding box format encoding operation to convert format from <code>xywh</code> to <code>xyxy</code>.</p>
</td>
</tr>
<tr id="row182991559155515"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p182991959195516"><a name="p182991959195516"></a><a name="p182991959195516"></a><a href="(beta)torch_npu-contrib-function-npu_bbox_coder_encode_yolo.md">(beta) torch_npu.contrib.function.npu_bbox_coder_encode_yolo</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001606524122_p1676145873611"><a name="en-us_topic_0000001606524122_p1676145873611"></a><a name="en-us_topic_0000001606524122_p1676145873611"></a><a href="https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L118" target="_blank" rel="noopener noreferrer">def npu_bbox_coder_encode_yolo()</a></p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p62992059175515"><a name="p62992059175515"></a><a name="p62992059175515"></a>Computes YOLO-style bounding box regression transformation deltas from source bounding boxes (<code>bboxes</code>) to target bounding boxes (<code>gt_bboxes</code>) through an NPU operator.</p>
</td>
</tr>
<tr id="row1249411377551"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p154944378551"><a name="p154944378551"></a><a name="p154944378551"></a><a href="(beta)torch_npu-contrib-module-npu_modules-DropoutWithByteMask.md">(beta) torch_npu.contrib.module.npu_modules.DropoutWithByteMask</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p543592755219"><a name="p543592755219"></a><a name="p543592755219"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1149463719556"><a name="p1149463719556"></a><a name="p1149463719556"></a>Applies an NPU-compatible <code>DropoutWithByteMask</code> operation.</p>
</td>
</tr>
<tr id="row2914185425510"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p2091419545552"><a name="p2091419545552"></a><a name="p2091419545552"></a><a href="(beta)torch_npu-contrib-function-dropout_with_byte_mask.md">(beta) torch_npu.contrib.function.dropout_with_byte_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p1435227155218"><a name="p1435227155218"></a><a name="p1435227155218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p3914125415512"><a name="p3914125415512"></a><a name="p3914125415512"></a>Applies an NPU-compatible <code>dropout_with_byte_mask</code> operation. This function is supported exclusively on NPU devices. It generates a stateless random <code>uint8</code> mask and performs dropout based on that mask.</p>
</td>
</tr>
<tr id="row62941778711"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p183268185292"><a name="p183268185292"></a><a name="p183268185292"></a><a href="torch_npu-contrib-module-LinearA8W8Quant.md">torch_npu.contrib.module.LinearA8W8Quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="en-us_topic_0000001778938168_p156512056161014"><a name="en-us_topic_0000001778938168_p156512056161014"></a><a name="en-us_topic_0000001778938168_p156512056161014"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p16295679713"><a name="p16295679713"></a><a name="p16295679713"></a>Encapsulates the <code>torch_npu.npu_quant_matmul</code> API to perform matrix multiplication computations for the A8W8 quantized operator.</p>
</td>
</tr>
<tr id="row4342314612"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p23423113616"><a name="p23423113616"></a><a name="p23423113616"></a><a href="torch_npu-contrib-module-LinearQuant.md">torch_npu.contrib.module.LinearQuant</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p133412311662"><a name="p133412311662"></a><a name="p133412311662"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="en-us_topic_0000002045840024_p9744194141510"><a name="en-us_topic_0000002045840024_p9744194141510"></a><a name="en-us_topic_0000002045840024_p9744194141510"></a>Encapsulates the <code>torch_npu.npu_quant_matmul</code> API to perform matrix multiplication computations for A8W8 and A4W4 quantized operators.</p>
</td>
</tr>
<tr id="row3799410478"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p147993102716"><a name="p147993102716"></a><a name="p147993102716"></a><a href="torch_npu-contrib-module-LinearWeightQuant.md">torch_npu.contrib.module.LinearWeightQuant</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p479919101712"><a name="p479919101712"></a><a name="p479919101712"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p197991410478"><a name="p197991410478"></a><a name="p197991410478"></a>Encapsulates the <code>torch_npu.npu_weight_quant_batchmatmul</code> API to perform quantization on weight inputs and outputs in matrix multiplication computations. This class supports <code>pertensor</code>, <code>perchannel</code>, and <code>pergroup</code> scenarios.</p>
</td>
</tr>
<tr id="row5577141319717"><td class="cellrowborder" valign="top" width="22.84%" headers="mcps1.2.4.1.1 "><p id="p9734435173615"><a name="p9734435173615"></a><a name="p9734435173615"></a><a href="torch_npu-contrib-module-QuantConv2d.md">torch_npu.contrib.module.QuantConv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="18.38%" headers="mcps1.2.4.1.2 "><p id="p157710131475"><a name="p157710131475"></a><a name="p157710131475"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="58.78%" headers="mcps1.2.4.1.3 "><p id="p1057719133712"><a name="p1057719133712"></a><a name="p1057719133712"></a>Encapsulates the <code>torch_npu.npu_quant_conv2d</code> API to provide quantization-related functionality for the <code>Conv2d</code> operator.</p>
</td>
</tr>
</tbody>
</table>
