# torch_npu接口列表

本章节包含常用自定义接口，包括创建tensor及计算类操作。

**表1** torch_npu API

<a name="table1849611717116"></a>
<table><thead align="left"><tr id="row10496101716111"><th class="cellrowborder" valign="top" width="38.61%" id="mcps1.2.3.1.1"><p id="p1649713174119"><a name="p1649713174119"></a><a name="p1649713174119"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="61.39%" id="mcps1.2.3.1.2"><p id="p9497217151115"><a name="p9497217151115"></a><a name="p9497217151115"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1149711715114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13497217191119"><a name="p13497217191119"></a><a name="p13497217191119"></a><a href="（beta）torch_npu-_npu_dropout.md">（beta）torch_npu._npu_dropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p349751701118"><a name="p349751701118"></a><a name="p349751701118"></a>不使用种子（seed）进行dropout结果计数。与torch.dropout相似，优化NPU设备实现。</p>
</td>
</tr>
<tr id="row13497417121111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p11170526121211"><a name="p11170526121211"></a><a name="p11170526121211"></a><a href="（beta）torch_npu-copy_memory_.md">（beta）torch_npu.copy_memory_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p74971174115"><a name="p74971174115"></a><a name="p74971174115"></a>从src拷贝元素到self张量，并返回self。</p>
</td>
</tr>
<tr id="row949751712110"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p164971017121112"><a name="p164971017121112"></a><a name="p164971017121112"></a><a href="（beta）torch_npu-empty_with_format.md">（beta）torch_npu.empty_with_format</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p8497111791114"><a name="p8497111791114"></a><a name="p8497111791114"></a>返回一个填充未初始化数据的张量。</p>
</td>
</tr>
<tr id="row949712178110"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5497817161113"><a name="p5497817161113"></a><a name="p5497817161113"></a><a href="（beta）torch_npu-fast_gelu.md">（beta）torch_npu.fast_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p9497101717113"><a name="p9497101717113"></a><a name="p9497101717113"></a>计算输入张量中fast_gelu的梯度。</p>
</td>
</tr>
<tr id="row17497217181119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16497617191116"><a name="p16497617191116"></a><a name="p16497617191116"></a><a href="（beta）torch_npu-npu_alloc_float_status.md">（beta）torch_npu.npu_alloc_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p6497141711115"><a name="p6497141711115"></a><a name="p6497141711115"></a>为溢出检测模式申请tensor作为入参。</p>
</td>
</tr>
<tr id="row104977172114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1849701710119"><a name="p1849701710119"></a><a name="p1849701710119"></a><a href="（beta）torch_npu-npu_anchor_response_flags.md">（beta）torch_npu.npu_anchor_response_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p69023133817"><a name="zh-cn_topic_0000001655404257_p69023133817"></a><a name="zh-cn_topic_0000001655404257_p69023133817"></a>在单个特征图中生成锚点的责任标志。</p>
</td>
</tr>
<tr id="row1120483231119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p152051325111"><a name="p152051325111"></a><a name="p152051325111"></a><a href="（beta）torch_npu-npu_apply_adam.md">（beta）torch_npu.npu_apply_adam</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18205193215119"><a name="p18205193215119"></a><a name="p18205193215119"></a>获取adam优化器的计算结果。</p>
</td>
</tr>
<tr id="row1920533291116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13205173251112"><a name="p13205173251112"></a><a name="p13205173251112"></a><a href="（beta）torch_npu-npu_batch_nms.md">（beta）torch_npu.npu_batch_nms</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p820523261113"><a name="p820523261113"></a><a name="p820523261113"></a>根据batch分类计算输入框评分，通过评分排序，删除评分高于阈值（iou_threshold）的框，支持多批多类处理。通过NonMaxSuppression（nms）操作可有效删除冗余的输入框，提高检测精度。NonMaxSuppression：抑制不是极大值的元素，搜索局部的极大值，常用于计算机视觉任务中的检测类模型。</p>
</td>
</tr>
<tr id="row162057328112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p11205193211112"><a name="p11205193211112"></a><a name="p11205193211112"></a><a href="（beta）torch_npu-npu_bert_apply_adam.md">（beta）torch_npu.npu_bert_apply_adam</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12061132181116"><a name="p12061132181116"></a><a name="p12061132181116"></a>针对bert模型，获取adam优化器的计算结果。</p>
</td>
</tr>
<tr id="row182061132151115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p52061332161110"><a name="p52061332161110"></a><a name="p52061332161110"></a><a href="（beta）torch_npu-npu_bmmV2.md">（beta）torch_npu.npu_bmmV2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p1582323323616"><a name="zh-cn_topic_0000001655404257_p1582323323616"></a><a name="zh-cn_topic_0000001655404257_p1582323323616"></a>将矩阵“a”乘以矩阵“b”，生成“a*b”。</p>
</td>
</tr>
<tr id="row520615328117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p020653281114"><a name="p020653281114"></a><a name="p020653281114"></a><a href="（beta）torch_npu-npu_bounding_box_decode.md">（beta）torch_npu.npu_bounding_box_decode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p2640134717250"><a name="zh-cn_topic_0000001655404257_p2640134717250"></a><a name="zh-cn_topic_0000001655404257_p2640134717250"></a>根据rois和deltas生成标注框。自定义Faster R-CNN算子。</p>
</td>
</tr>
<tr id="row657717509111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2577105001114"><a name="p2577105001114"></a><a name="p2577105001114"></a><a href="（beta）torch_npu-npu_bounding_box_encode.md">（beta）torch_npu.npu_bounding_box_encode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2577105019119"><a name="p2577105019119"></a><a name="p2577105019119"></a>计算标注框和ground truth真值框之间的坐标变化。自定义Faster R-CNN算子。</p>
</td>
</tr>
<tr id="row989165331120"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p38965316111"><a name="p38965316111"></a><a name="p38965316111"></a><a href="（beta）torch_npu-npu_broadcast.md">（beta）torch_npu.npu_broadcast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p918313310142"><a name="zh-cn_topic_0000001655404257_p918313310142"></a><a name="zh-cn_topic_0000001655404257_p918313310142"></a>返回self张量的新视图，其单维度扩展，结果连续。张量也可以扩展更多维度，新的维度添加在最前面。</p>
</td>
</tr>
<tr id="row589165316115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1489155321111"><a name="p1489155321111"></a><a name="p1489155321111"></a><a href="（beta）torch_npu-npu_ciou.md">（beta）torch_npu.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p749621223015"><a name="zh-cn_topic_0000001655404257_p749621223015"></a><a name="zh-cn_topic_0000001655404257_p749621223015"></a>应用基于NPU的CIoU操作。在DIoU的基础上增加了penalty item，并propose CIoU。</p>
</td>
</tr>
<tr id="row138914536113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1289453191113"><a name="p1289453191113"></a><a name="p1289453191113"></a><a href="（beta）torch_npu-npu_clear_float_status.md">（beta）torch_npu.npu_clear_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p289105341110"><a name="p289105341110"></a><a name="p289105341110"></a>清除溢出检测相关标志位。</p>
</td>
</tr>
<tr id="row158945317117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5901953141111"><a name="p5901953141111"></a><a name="p5901953141111"></a><a href="（beta）torch_npu-npu_confusion_transpose.md">（beta）torch_npu.npu_confusion_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p7908532119"><a name="p7908532119"></a><a name="p7908532119"></a>混淆reshape和transpose运算。</p>
</td>
</tr>
<tr id="row2090953141119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1690125311117"><a name="p1690125311117"></a><a name="p1690125311117"></a><a href="（beta）torch_npu-npu_conv_transpose2d.md">（beta）torch_npu.npu_conv_transpose2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p122041623817"><a name="zh-cn_topic_0000001655404257_p122041623817"></a><a name="zh-cn_topic_0000001655404257_p122041623817"></a>在由多个输入平面组成的输入图像上应用一个2D转置卷积算子，有时这个过程也被称为“反卷积”。</p>
</td>
</tr>
<tr id="row49065381112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p189035310112"><a name="p189035310112"></a><a name="p189035310112"></a><a href="（beta）torch_npu-npu_conv2d.md">（beta）torch_npu.npu_conv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1790753191113"><a name="p1790753191113"></a><a name="p1790753191113"></a>在由多个输入平面组成的输入图像上应用一个2D卷积。</p>
</td>
</tr>
<tr id="row109045310113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p49035381110"><a name="p49035381110"></a><a name="p49035381110"></a><a href="（beta）torch_npu-npu_conv3d.md">（beta）torch_npu.npu_conv3d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18901253111116"><a name="p18901253111116"></a><a name="p18901253111116"></a>在由多个输入平面组成的输入图像上应用一个3D卷积。</p>
</td>
</tr>
<tr id="row0901534111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1090253101111"><a name="p1090253101111"></a><a name="p1090253101111"></a><a href="（beta）torch_npu-npu_convolution.md">（beta）torch_npu.npu_convolution</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p890053121116"><a name="p890053121116"></a><a name="p890053121116"></a>在由多个输入平面组成的输入图像上应用一个2D或3D卷积。</p>
</td>
</tr>
<tr id="row9577205019119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p7577105010119"><a name="p7577105010119"></a><a name="p7577105010119"></a><a href="（beta）torch_npu-npu_convolution_transpose.md">（beta）torch_npu.npu_convolution_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p35776509119"><a name="p35776509119"></a><a name="p35776509119"></a>在由多个输入平面组成的输入图像上应用一个2D或3D转置卷积算子，有时这个过程也被称为“反卷积”。</p>
</td>
</tr>
<tr id="row75771250101115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p15771150151111"><a name="p15771150151111"></a><a name="p15771150151111"></a><a href="（beta）torch_npu-npu_deformable_conv2d.md">（beta）torch_npu.npu_deformable_conv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p1888192943717"><a name="zh-cn_topic_0000001655404257_p1888192943717"></a><a name="zh-cn_topic_0000001655404257_p1888192943717"></a>使用预期输入计算变形卷积输出（deformed convolution output）。</p>
</td>
</tr>
<tr id="row125771550121117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16577135019112"><a name="p16577135019112"></a><a name="p16577135019112"></a><a href="（beta）torch_npu-npu_diou.md">（beta）torch_npu.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p12796833183010"><a name="zh-cn_topic_0000001655404257_p12796833183010"></a><a name="zh-cn_topic_0000001655404257_p12796833183010"></a>应用基于NPU的DIoU操作。考虑到目标之间距离，以及距离和范围的重叠率，不同目标或边界需趋于稳定。</p>
</td>
</tr>
<tr id="row95771950161112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p35771150191112"><a name="p35771150191112"></a><a name="p35771150191112"></a><a href="（beta）torch_npu-npu_dtype_cast.md">（beta）torch_npu.npu_dtype_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p1221456142618"><a name="zh-cn_topic_0000001655404257_p1221456142618"></a><a name="zh-cn_topic_0000001655404257_p1221456142618"></a>执行张量数据类型（dtype）转换。</p>
</td>
</tr>
<tr id="row195782506118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1857885061113"><a name="p1857885061113"></a><a name="p1857885061113"></a><a href="（beta）torch_npu-npu_format_cast.md">（beta）torch_npu.npu_format_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p68591917123"><a name="zh-cn_topic_0000001655404257_p68591917123"></a><a name="zh-cn_topic_0000001655404257_p68591917123"></a>修改NPU张量的格式。</p>
</td>
</tr>
<tr id="row457855012117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p557813506114"><a name="p557813506114"></a><a name="p557813506114"></a><a href="（beta）torch_npu-npu_format_cast_.md">（beta）torch_npu.npu_format_cast_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p125789503117"><a name="p125789503117"></a><a name="p125789503117"></a>原地修改self张量格式，与src格式保持一致。</p>
</td>
</tr>
<tr id="row1757895091111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p17578195010114"><a name="p17578195010114"></a><a name="p17578195010114"></a><a href="（beta）torch_npu-npu_get_float_status.md">（beta）torch_npu.npu_get_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p8144193193519"><a name="zh-cn_topic_0000001655404257_p8144193193519"></a><a name="zh-cn_topic_0000001655404257_p8144193193519"></a>获取溢出检测结果。</p>
</td>
</tr>
<tr id="row3206113212114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p42065329112"><a name="p42065329112"></a><a name="p42065329112"></a><a href="（beta）torch_npu-npu_giou.md">（beta）torch_npu.npu_giou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p0835111286"><a name="zh-cn_topic_0000001655404257_p0835111286"></a><a name="zh-cn_topic_0000001655404257_p0835111286"></a>首先计算两个框的最小封闭面积和IoU，然后计算封闭区域中不属于两个框的封闭面积的比例，最后从IoU中减去这个比例，得到GIoU。</p>
</td>
</tr>
<tr id="row5377163616111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p937723691115"><a name="p937723691115"></a><a name="p937723691115"></a><a href="（beta）torch_npu-npu_grid_assign_positive.md">（beta）torch_npu.npu_grid_assign_positive</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p258994112512"><a name="zh-cn_topic_0000001655404257_p258994112512"></a><a name="zh-cn_topic_0000001655404257_p258994112512"></a><span>执行position-sensitive的候选区域池化梯度计算。</span></p>
</td>
</tr>
<tr id="row14377936121116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1337753621111"><a name="p1337753621111"></a><a name="p1337753621111"></a><a href="（beta）torch_npu-npu_gru.md">（beta）torch_npu.npu_gru</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p798911210261"><a name="zh-cn_topic_0000001655404257_p798911210261"></a><a name="zh-cn_topic_0000001655404257_p798911210261"></a>计算DynamicGRUV2。</p>
</td>
</tr>
<tr id="row557919470115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p157994710118"><a name="p157994710118"></a><a name="p157994710118"></a><a href="（beta）torch_npu-npu_indexing.md">（beta）torch_npu.npu_indexing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p191786291"><a name="zh-cn_topic_0000001655404257_p191786291"></a><a name="zh-cn_topic_0000001655404257_p191786291"></a>使用“begin,end,strides”数组对index结果进行计数。</p>
</td>
</tr>
<tr id="row11579164761115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p195798473119"><a name="p195798473119"></a><a name="p195798473119"></a><a href="（beta）torch_npu-npu_iou.md">（beta）torch_npu.npu_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p10544194202310"><a name="zh-cn_topic_0000001655404257_p10544194202310"></a><a name="zh-cn_topic_0000001655404257_p10544194202310"></a>根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。</p>
</td>
</tr>
<tr id="row10579184741114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p35791247141119"><a name="p35791247141119"></a><a name="p35791247141119"></a><a href="（beta）torch_npu-npu_layer_norm_eval.md">（beta）torch_npu.npu_layer_norm_eval</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p75791547101117"><a name="p75791547101117"></a><a name="p75791547101117"></a>对层归一化结果进行计数。与torch.nn.functional.layer_norm相同，优化NPU设备实现。</p>
</td>
</tr>
<tr id="row11579104721117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p135794473113"><a name="p135794473113"></a><a name="p135794473113"></a><a href="（beta）torch_npu-npu_linear.md">（beta）torch_npu.npu_linear</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p8579144781110"><a name="p8579144781110"></a><a name="p8579144781110"></a>将矩阵“a”乘以矩阵“b”，生成“a*b”。</p>
</td>
</tr>
<tr id="row9579184715119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p85793478116"><a name="p85793478116"></a><a name="p85793478116"></a><a href="（beta）torch_npu-npu_lstm.md">（beta）torch_npu.npu_lstm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p757912479111"><a name="p757912479111"></a><a name="p757912479111"></a>计算DynamicRNN。</p>
</td>
</tr>
<tr id="row3579194701112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16579184717113"><a name="p16579184717113"></a><a name="p16579184717113"></a><a href="（beta）torch_npu-npu_max.md">（beta）torch_npu.npu_max</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1758054719115"><a name="p1758054719115"></a><a name="p1758054719115"></a>使用dim对最大结果进行计数。类似于torch.max，优化NPU设备实现。</p>
</td>
</tr>
<tr id="row2580347161118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p12580124781118"><a name="p12580124781118"></a><a name="p12580124781118"></a><a href="（beta）torch_npu-npu_min.md">（beta）torch_npu.npu_min</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p1171133519333"><a name="zh-cn_topic_0000001655404257_p1171133519333"></a><a name="zh-cn_topic_0000001655404257_p1171133519333"></a>使用dim对最小结果进行计数。类似于torch.min，优化NPU设备实现。</p>
</td>
</tr>
<tr id="row115511438538"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p75511930538"><a name="p75511930538"></a><a name="p75511930538"></a><a href="（beta）torch_npu-npu_mish.md">（beta）torch_npu.npu_mish</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12741142714357"><a name="p12741142714357"></a><a name="p12741142714357"></a>按元素计算self的双曲正切。</p>
</td>
</tr>
<tr id="row558010477110"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p17580147201113"><a name="p17580147201113"></a><a name="p17580147201113"></a><a href="（beta）torch_npu-npu_nms_rotated.md">（beta）torch_npu.npu_nms_rotated</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p2907120192218"><a name="zh-cn_topic_0000001655404257_p2907120192218"></a><a name="zh-cn_topic_0000001655404257_p2907120192218"></a>按分数降序选择旋转标注框的子集。</p>
</td>
</tr>
<tr id="row1158054718119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1858004714112"><a name="p1858004714112"></a><a name="p1858004714112"></a><a href="（beta）torch_npu-npu_nms_v4.md">（beta）torch_npu.npu_nms_v4</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p296295602119"><a name="zh-cn_topic_0000001655404257_p296295602119"></a><a name="zh-cn_topic_0000001655404257_p296295602119"></a>按分数降序选择标注框的子集。</p>
</td>
</tr>
<tr id="row85801547131119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p105802478112"><a name="p105802478112"></a><a name="p105802478112"></a><a href="（beta）torch_npu-npu_nms_with_mask.md">（beta）torch_npu.npu_nms_with_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1358054721113"><a name="p1358054721113"></a><a name="p1358054721113"></a>生成值0或1，用于nms算子确定有效位。</p>
</td>
</tr>
<tr id="row858014718113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p25801047101117"><a name="p25801047101117"></a><a name="p25801047101117"></a><a href="（beta）torch_npu-npu_one_hot.md">（beta）torch_npu.npu_one_hot</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5580547191120"><a name="p5580547191120"></a><a name="p5580547191120"></a>返回一个one-hot张量。input中index表示的位置采用on_value值，而其他所有位置采用off_value的值。</p>
</td>
</tr>
<tr id="row2037715364114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p173771360112"><a name="p173771360112"></a><a name="p173771360112"></a><a href="（beta）torch_npu-npu_pad.md">（beta）torch_npu.npu_pad</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p93771036191114"><a name="p93771036191114"></a><a name="p93771036191114"></a>填充张量。</p>
</td>
</tr>
<tr id="row1537743614119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p737793615117"><a name="p737793615117"></a><a name="p737793615117"></a><a href="（beta）torch_npu-npu_ps_roi_pooling.md">（beta）torch_npu.npu_ps_roi_pooling</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1037763620111"><a name="p1037763620111"></a><a name="p1037763620111"></a>执行Position Sensitive ROI Pooling。</p>
</td>
</tr>
<tr id="row2037743613112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p12377203601118"><a name="p12377203601118"></a><a name="p12377203601118"></a><a href="（beta）torch_npu-npu_ptiou.md">（beta）torch_npu.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1593155184019"><a name="p1593155184019"></a><a name="p1593155184019"></a>根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。</p>
</td>
</tr>
<tr id="row1337793616112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p133781436181115"><a name="p133781436181115"></a><a name="p133781436181115"></a><a href="（beta）torch_npu-npu_random_choice_with_mask.md">（beta）torch_npu.npu_random_choice_with_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001655404257_p219311346269"><a name="zh-cn_topic_0000001655404257_p219311346269"></a><a name="zh-cn_topic_0000001655404257_p219311346269"></a>混洗非零元素的index。</p>
</td>
</tr>
<tr id="row837833671119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1137810369111"><a name="p1137810369111"></a><a name="p1137810369111"></a><a href="（beta）torch_npu-npu_reshape.md">（beta）torch_npu.npu_reshape</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14378123610113"><a name="p14378123610113"></a><a name="p14378123610113"></a>reshape张量。仅更改张量shape，其数据不变。</p>
</td>
</tr>
<tr id="row11378336161116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p83782360112"><a name="p83782360112"></a><a name="p83782360112"></a><a href="（beta）torch_npu-npu_roi_align.md">（beta）torch_npu.npu_roi_align</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p11378133620112"><a name="p11378133620112"></a><a name="p11378133620112"></a>从特征图中获取ROI特征矩阵。自定义Faster R-CNN算子。</p>
</td>
</tr>
<tr id="row168411042171118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p11841184214110"><a name="p11841184214110"></a><a name="p11841184214110"></a><a href="（beta）torch_npu-npu_rotated_iou.md">（beta）torch_npu.npu_rotated_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p384134214117"><a name="p384134214117"></a><a name="p384134214117"></a>计算旋转框的IoU。</p>
</td>
</tr>
<tr id="row1284118427112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p0841184211115"><a name="p0841184211115"></a><a name="p0841184211115"></a><a href="（beta）torch_npu-npu_rotated_overlaps.md">（beta）torch_npu.npu_rotated_overlaps</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p20841164221110"><a name="p20841164221110"></a><a name="p20841164221110"></a>计算旋转框的重叠面积。</p>
</td>
</tr>
<tr id="row68411842121111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1384134221114"><a name="p1384134221114"></a><a name="p1384134221114"></a><a href="（beta）torch_npu-npu_sign_bits_pack.md">（beta）torch_npu.npu_sign_bits_pack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p484114213112"><a name="p484114213112"></a><a name="p484114213112"></a>将float类型1位Adam打包为uint8。</p>
</td>
</tr>
<tr id="row128412042131111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p784116424113"><a name="p784116424113"></a><a name="p784116424113"></a><a href="（beta）torch_npu-npu_sign_bits_unpack.md">（beta）torch_npu.npu_sign_bits_unpack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p58411242191112"><a name="p58411242191112"></a><a name="p58411242191112"></a>将uint8类型1位Adam拆包为float。</p>
</td>
</tr>
<tr id="row17842154219116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1384284211115"><a name="p1384284211115"></a><a name="p1384284211115"></a><a href="（beta）torch_npu-npu_silu.md">（beta）torch_npu.npu_silu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p11842194231120"><a name="p11842194231120"></a><a name="p11842194231120"></a>计算self的Swish。</p>
</td>
</tr>
<tr id="row68428425112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1842204218113"><a name="p1842204218113"></a><a name="p1842204218113"></a><a href="（beta）torch_npu-npu_slice.md">（beta）torch_npu.npu_slice</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p0842124210117"><a name="p0842124210117"></a><a name="p0842124210117"></a>从张量中提取切片。</p>
</td>
</tr>
<tr id="row7842204261115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13842442161112"><a name="p13842442161112"></a><a name="p13842442161112"></a><a href="（beta）torch_npu-npu_softmax_cross_entropy_with_logits.md">（beta）torch_npu.npu_softmax_cross_entropy_with_logits</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1842144291119"><a name="p1842144291119"></a><a name="p1842144291119"></a>计算softmax的交叉熵cost。</p>
</td>
</tr>
<tr id="row1084224211115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p188421442111120"><a name="p188421442111120"></a><a name="p188421442111120"></a><a href="（beta）torch_npu-npu_sort_v2.md">（beta）torch_npu.npu_sort_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5842442111110"><a name="p5842442111110"></a><a name="p5842442111110"></a>沿给定维度，按无index值对输入张量元素进行升序排序。若dim未设置，则选择输入的最后一个维度。如果descending为True，则元素将按值降序排序。</p>
</td>
</tr>
<tr id="row884254219118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1784284271118"><a name="p1784284271118"></a><a name="p1784284271118"></a><a href="（beta）torch_npu-npu_transpose.md">（beta）torch_npu.npu_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1884294261114"><a name="p1884294261114"></a><a name="p1884294261114"></a>返回原始张量视图，其维度已permute，结果连续。</p>
</td>
</tr>
<tr id="row118421942161116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p188423429111"><a name="p188423429111"></a><a name="p188423429111"></a><a href="（beta）torch_npu-npu_yolo_boxes_encode.md">（beta）torch_npu.npu_yolo_boxes_encode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p68421942191116"><a name="p68421942191116"></a><a name="p68421942191116"></a>根据YOLO的锚点框（anchor box）和真值框（ground-truth box）生成标注框。自定义mmdetection算子。</p>
</td>
</tr>
<tr id="row6842164251115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p19842134241119"><a name="p19842134241119"></a><a name="p19842134241119"></a><a href="（beta）torch_npu-npu_fused_attention_score.md">（beta）torch_npu.npu_fused_attention_score</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p98421642101113"><a name="p98421642101113"></a><a name="p98421642101113"></a>实现“Transformer attention score”的融合计算逻辑，主要将matmul、transpose、add、softmax、dropout、batchmatmul、permute等计算进行了融合。</p>
</td>
</tr>
<tr id="row2378193620119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p9378836161118"><a name="p9378836161118"></a><a name="p9378836161118"></a><a href="（beta）torch_npu-npu_multi_head_attention.md">（beta）torch_npu.npu_multi_head_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14378143615111"><a name="p14378143615111"></a><a name="p14378143615111"></a>实现Transformer模块中的MultiHeadAttention计算逻辑。</p>
</td>
</tr>
<tr id="row14545161172716"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p854661182710"><a name="p854661182710"></a><a name="p854661182710"></a><a href="（beta）torch_npu-npu_rms_norm.md">（beta）torch_npu.npu_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p125461318273"><a name="p125461318273"></a><a name="p125461318273"></a>RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。</p>
</td>
</tr>
<tr id="row2791419328"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1990671211278"><a name="p1990671211278"></a><a name="p1990671211278"></a><a href="（beta）torch_npu-npu_dropout_with_add_softmax.md">（beta）torch_npu.npu_dropout_with_add_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p15906141216278"><a name="p15906141216278"></a><a name="p15906141216278"></a>实现axpy_v2、softmax_v2、drop_out_domask_v3功能。</p>
</td>
</tr>
<tr id="row17801219526"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p89062124277"><a name="p89062124277"></a><a name="p89062124277"></a><a href="torch_npu-npu_rotary_mul.md">torch_npu.npu_rotary_mul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p69061812102715"><a name="p69061812102715"></a><a name="p69061812102715"></a>实现RotaryEmbedding旋转位置编码。</p>
</td>
</tr>
<tr id="row58015194219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1590618121275"><a name="p1590618121275"></a><a name="p1590618121275"></a><a href="torch_npu-npu_scaled_masked_softmax.md">torch_npu.npu_scaled_masked_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1290691218270"><a name="p1290691218270"></a><a name="p1290691218270"></a>计算输入张量x缩放并按照mask遮蔽后的Softmax结果。</p>
</td>
</tr>
<tr id="row480181910211"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p137651452352"><a name="p137651452352"></a><a name="p137651452352"></a><a href="（beta）torch_npu-npu_swiglu.md">（beta）torch_npu.npu_swiglu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14325153814812"><a name="p14325153814812"></a><a name="p14325153814812"></a>提供swiglu的激活函数。</p>
</td>
</tr>
<tr id="row178010191727"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13901449479"><a name="p13901449479"></a><a name="p13901449479"></a><a href="（beta）torch_npu-one_.md">（beta）torch_npu.one_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001726825729_zh-cn_topic_0000001655404257_p5268123310116"><a name="zh-cn_topic_0000001726825729_zh-cn_topic_0000001655404257_p5268123310116"></a><a name="zh-cn_topic_0000001726825729_zh-cn_topic_0000001655404257_p5268123310116"></a>用1填充self张量。</p>
</td>
</tr>
<tr id="row106843515114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p168510515113"><a name="p168510515113"></a><a name="p168510515113"></a><a href="（beta）torch_npu-npu_group_norm_swish.md">torch_npu.npu_group_norm_swish</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p19342196105315"><a name="p19342196105315"></a><a name="p19342196105315"></a>对输入input进行组归一化计算，并计算Swish。</p>
</td>
</tr>
<tr id="row10636159121815"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p763620598183"><a name="p763620598183"></a><a name="p763620598183"></a><a href="torch_npu-npu_cross_entropy_loss.md">torch_npu.npu_cross_entropy_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18198201124816"><a name="p18198201124816"></a><a name="p18198201124816"></a>将原生CrossEntropyLoss中的log_softmax和nll_loss融合，降低计算时使用的内存。接口允许计算zloss。</p>
</td>
</tr>
<tr id="row724843692712"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p162481336102714"><a name="p162481336102714"></a><a name="p162481336102714"></a><a href="（beta）torch_npu-npu_advance_step_flashattn.md">torch_npu.npu_advance_step_flashattn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p13248936102717"><a name="p13248936102717"></a><a name="p13248936102717"></a>在NPU上实现vLLM库中advance_step_flashattn的功能，在每个生成步骤中原地更新input_tokens，input_positions，seq_lens和slot_mapping。</p>
</td>
</tr>
<tr id="row1526115194711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p9360134415216"><a name="p9360134415216"></a><a name="p9360134415216"></a><a href="torch_npu-npu_all_gather_base_mm.md">torch_npu.npu_all_gather_base_mm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001694916914_p444492882116"><a name="zh-cn_topic_0000001694916914_p444492882116"></a><a name="zh-cn_topic_0000001694916914_p444492882116"></a>TP切分场景下，实现allgather和matmul的融合，融合算子内部实现通信和计算流水并行。</p>
</td>
</tr>
<tr id="row952719519471"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18248651143510"><a name="p18248651143510"></a><a name="p18248651143510"></a><a href="torch_npu-npu_anti_quant.md">torch_npu.npu_anti_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001850161621_p0996174814315"><a name="zh-cn_topic_0000001850161621_p0996174814315"></a><a name="zh-cn_topic_0000001850161621_p0996174814315"></a>将INT4或者INT8数据反量化为FP16或者BF16，其中输入是INT4类型时，将每8个数据看作是一个INT32数据。</p>
</td>
</tr>
<tr id="row16527145114477"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1338142581"><a name="p1338142581"></a><a name="p1338142581"></a><a href="torch_npu-npu_convert_weight_to_int4pack.md">torch_npu.npu_convert_weight_to_int4pack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p16338242582"><a name="p16338242582"></a><a name="p16338242582"></a>将数据类型为int32的输入tensor打包为int4存放，每8个int4数据通过一个int32数据承载，并进行交叠排放。</p>
</td>
</tr>
<tr id="row194591911194818"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p633812455816"><a name="p633812455816"></a><a name="p633812455816"></a><a href="torch_npu-npu_dynamic_quant.md">torch_npu.npu_dynamic_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p93381541583"><a name="p93381541583"></a><a name="p93381541583"></a>为输入的张量进行pre-token对称动态量化。</p>
</td>
</tr>
<tr id="row44591911184819"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p820225218412"><a name="p820225218412"></a><a name="p820225218412"></a><a href="torch_npu-npu_dynamic_quant_asymmetric.md">torch_npu.npu_dynamic_quant_asymmetric</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p271341016220"><a name="p271341016220"></a><a name="p271341016220"></a>对输入的张量进行per-token非对称动态量化。其中输入的最后一个维度对应一个token，每个token作为一组进行量化。</p>
</td>
</tr>
<tr id="row1245931134815"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1055135819577"><a name="p1055135819577"></a><a name="p1055135819577"></a><a href="torch_npu-npu_fast_gelu.md">torch_npu.npu_fast_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p135525810574"><a name="p135525810574"></a><a name="p135525810574"></a>快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function），对输入的每个元素计算FastGelu；输入是具有任何有效形状的张量。</p>
</td>
</tr>
<tr id="row545919116481"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p099812152913"><a name="p099812152913"></a><a name="p099812152913"></a><a href="torch_npu-npu_ffn.md">torch_npu.npu_ffn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1998152112915"><a name="p1998152112915"></a><a name="p1998152112915"></a>该FFN算子提供MoeFFN和FFN的计算功能。在没有专家分组（expert_tokens为空）时是FFN，有专家分组时是MoeFFN。</p>
</td>
</tr>
<tr id="row1045961120487"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p43681016105618"><a name="p43681016105618"></a><a name="p43681016105618"></a><a href="torch_npu-npu_fused_infer_attention_score.md">torch_npu.npu_fused_infer_attention_score</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p836911165560"><a name="p836911165560"></a><a name="p836911165560"></a>适配增量&amp;全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。</p>
</td>
</tr>
<tr id="row14601311184810"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1799710217294"><a name="p1799710217294"></a><a name="p1799710217294"></a><a href="torch_npu-npu_fusion_attention.md">torch_npu.npu_fusion_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p129971221112919"><a name="p129971221112919"></a><a name="p129971221112919"></a>实现“Transformer Attention Score”的融合计算。</p>
</td>
</tr>
<tr id="row24601511114817"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1720095217417"><a name="p1720095217417"></a><a name="p1720095217417"></a><a href="torch_npu-npu_gelu.md">torch_npu.npu_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p9269284217"><a name="p9269284217"></a><a name="p9269284217"></a>计算高斯误差线性单元的激活函数。</p>
</td>
</tr>
<tr id="row3460101118487"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5338134195811"><a name="p5338134195811"></a><a name="p5338134195811"></a><a href="torch_npu-npu_group_norm_silu.md">torch_npu.npu_group_norm_silu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5338546582"><a name="p5338546582"></a><a name="p5338546582"></a>计算输入self的组归一化结果out、均值meanOut、标准差的倒数rstdOut、以及silu的输出。</p>
</td>
</tr>
<tr id="row174571042727"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10203752943"><a name="p10203752943"></a><a name="p10203752943"></a><a href="torch_npu-npu_group_quant.md">torch_npu.npu_group_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p4382954217"><a name="p4382954217"></a><a name="p4382954217"></a>对输入的张量进行分组量化操作。</p>
</td>
</tr>
<tr id="row1045718422215"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p64370382369"><a name="p64370382369"></a><a name="p64370382369"></a><a href="torch_npu-npu_grouped_matmul.md">torch_npu.npu_grouped_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1437638113619"><a name="p1437638113619"></a><a name="p1437638113619"></a><span>npu_grouped_matmul是一种对多个矩阵乘法（matmul）操作进行分组计算的高效方法。</span></p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_incre_flash_attention.md">torch_npu.npu_incre_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>增量FA实现。</p>
</td>
</tr>
<tr id="row1457134215217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p258365124912"><a name="p258365124912"></a><a name="p258365124912"></a><a href="torch_npu-npu_mla_prolog.md">torch_npu.npu_mla_prolog</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>推理场景下，Multi-Head Latent Attention前处理计算接口</span>。</p>
</td>
</tr>
<tr id="row1457134215217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p258365124912"><a name="p258365124912"></a><a name="p258365124912"></a><a href="torch_npu-npu_mla_prolog_v2.md">torch_npu.npu_mla_prolog_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>推理场景下，Multi-Head Latent Attention前处理计算的增强接口</span>。</p>
</td>
</tr>
<tr id="row125419366217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18998182119294"><a name="p18998182119294"></a><a name="p18998182119294"></a><a href="torch_npu-npu_mm_all_reduce_base.md">torch_npu.npu_mm_all_reduce_base</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001721582972_p2122194522714"><a name="zh-cn_topic_0000001721582972_p2122194522714"></a><a name="zh-cn_topic_0000001721582972_p2122194522714"></a>TP切分场景下，实现mm和all_reduce的融合，融合算子内部实现计算和通信流水并行。</p>
</td>
</tr>
<tr id="row102221513433"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p273718419525"><a name="p273718419525"></a><a name="p273718419525"></a><a href="torch_npu-npu_mm_reduce_scatter_base.md">torch_npu.npu_mm_reduce_scatter_base</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1385231745515"><a name="p1385231745515"></a><a name="p1385231745515"></a>TP切分场景下，实现matmul和reduce_scatter的融合，融合算子内部实现计算和通信流水并行。</p>
</td>
</tr>
<tr id="row192221013132"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1474331220582"><a name="p1474331220582"></a><a name="p1474331220582"></a><a href="torch_npu-npu_moe_compute_expert_tokens.md">torch_npu.npu_moe_compute_expert_tokens</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12743131265818"><a name="p12743131265818"></a><a name="p12743131265818"></a>MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。</p>
</td>
</tr>
<tr id="row622320135311"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2743141295810"><a name="p2743141295810"></a><a name="p2743141295810"></a><a href="torch_npu-npu_moe_finalize_routing.md">torch_npu.npu_moe_finalize_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1974341219588"><a name="p1974341219588"></a><a name="p1974341219588"></a>MoE计算中，最后处理合并MoE FFN的输出结果。</p>
</td>
</tr>
<tr id="row11308231313"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p57431312165820"><a name="p57431312165820"></a><a name="p57431312165820"></a><a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14743151275819"><a name="p14743151275819"></a><a name="p14743151275819"></a>MoE计算中，对gating的输出做Softmax计算，取topk操作。</p>
</td>
</tr>
<tr id="row133111231730"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p147431112105811"><a name="p147431112105811"></a><a name="p147431112105811"></a><a href="torch_npu-npu_moe_init_routing.md">torch_npu.npu_moe_init_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1174351212586"><a name="p1174351212586"></a><a name="p1174351212586"></a>MoE的routing计算，根据<a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a>的计算结果做routing处理。</p>
</td>
</tr>
<tr id="row1531023733"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p7819527225"><a name="p7819527225"></a><a name="p7819527225"></a><a href="torch_npu-npu_prefetch.md">torch_npu.npu_prefetch</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p11819162142215"><a name="p11819162142215"></a><a name="p11819162142215"></a>提供网络weight预取功能，将需要预取的权重搬到L2 Cache中（当前仅支持权重的预取，暂不支持KV cache的预取）。尤其在做较大Tensor的MatMul计算且需要搬移到L2 Cache的操作时，可通过该接口提前预取权重，适当提高模型性能，具体效果基于用户对并行的处理。</p>
</td>
</tr>
<tr id="row13116234313"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p35366207112"><a name="p35366207112"></a><a name="p35366207112"></a><a href="torch_npu-npu_prompt_flash_attention.md">torch_npu.npu_prompt_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p20536122017115"><a name="p20536122017115"></a><a name="p20536122017115"></a>全量FA实现。</p>
</td>
</tr>
<tr id="row9719124019218"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p14326218132919"><a name="p14326218132919"></a><a name="p14326218132919"></a><a href="torch_npu-npu_quant_matmul.md">torch_npu.npu_quant_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001814195101_p156512056161014"><a name="zh-cn_topic_0000001814195101_p156512056161014"></a><a name="zh-cn_topic_0000001814195101_p156512056161014"></a>完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。</p>
</td>
</tr>
<tr id="row9368201625615"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p432973317361"><a name="p432973317361"></a><a name="p432973317361"></a><a href="torch_npu-npu_quant_scatter.md">torch_npu.npu_quant_scatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p133291433113614"><a name="p133291433113614"></a><a name="p133291433113614"></a>先将updates进行量化，然后将updates中的值按指定的轴axis和索引indices更新self中的值，并将结果保存到输出tensor，self本身的数据不变。</p>
</td>
</tr>
<tr id="row159729564415"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p109331430113612"><a name="p109331430113612"></a><a name="p109331430113612"></a><a href="torch_npu-npu_quant_scatter_.md">torch_npu.npu_quant_scatter_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p693315307360"><a name="p693315307360"></a><a name="p693315307360"></a>先将updates进行量化，然后将updates中的值按指定的轴axis和索引indices更新self中的值，self中的数据被改变。</p>
</td>
</tr>
<tr id="row1432501813299"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p20338144125815"><a name="p20338144125815"></a><a name="p20338144125815"></a><a href="torch_npu-npu_quantize.md">torch_npu.npu_quantize</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p33382046581"><a name="p33382046581"></a><a name="p33382046581"></a>对输入的张量进行量化处理。</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_scatter_nd_update.md">torch_npu.npu_scatter_nd_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863744477_p15682749165813"><a name="zh-cn_topic_0000001863744477_p15682749165813"></a><a name="zh-cn_topic_0000001863744477_p15682749165813"></a>将updates中的值按指定的索引indices更新self中的值，并将结果保存到输出tensor，self本身的数据不变。</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_top_k_top_p.md">torch_npu.npu_top_k_top_p</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863744477_p15682749165813"><a name="zh-cn_topic_0000001863744477_p15682749165813"></a><a name="zh-cn_topic_0000001863744477_p15682749165813"></a>对原始输入<code>logits</code>进行<code>top-k</code>和<code>top-p</code>采样过滤。</p>
</td>
</tr>
<tr id="row9470191314519"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p83781723143618"><a name="p83781723143618"></a><a name="p83781723143618"></a><a href="torch_npu-npu_scatter_nd_update_.md">torch_npu.npu_scatter_nd_update_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001816704932_p12276121818501"><a name="zh-cn_topic_0000001816704932_p12276121818501"></a><a name="zh-cn_topic_0000001816704932_p12276121818501"></a>将updates中的值按指定的索引indices更新self中的值，并将结果保存到输出tensor，self中的数据被改变。</p>
</td>
</tr>
<tr id="row1243753818365"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1332519186295"><a name="p1332519186295"></a><a name="p1332519186295"></a><a href="torch_npu-npu_trans_quant_param.md">torch_npu.npu_trans_quant_param</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p10325181832913"><a name="p10325181832913"></a><a name="p10325181832913"></a>完成量化计算参数scale数据类型的转换。</p>
</td>
</tr>
<tr id="row1432853315360"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p34567614299"><a name="p34567614299"></a><a name="p34567614299"></a><a href="torch_npu-npu_weight_quant_batchmatmul.md">torch_npu.npu_weight_quant_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p154556613293"><a name="p154556613293"></a><a name="p154556613293"></a>该接口用于实现矩阵乘计算中的weight输入和输出的量化操作，支持pertensor，perchannel，pergroup多场景量化。</p>
</td>
</tr>
<tr id="row10933133063618"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5743101211585"><a name="p5743101211585"></a><a name="p5743101211585"></a><a href="torch_npu-scatter_update.md">torch_npu.scatter_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p874318124582"><a name="p874318124582"></a><a name="p874318124582"></a>将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据不变。</p>
</td>
</tr>
<tr id="row57012280360"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1274381212587"><a name="p1274381212587"></a><a name="p1274381212587"></a><a href="torch_npu-scatter_update_.md">torch_npu.scatter_update_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p3743712125811"><a name="p3743712125811"></a><a name="p3743712125811"></a>将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据被改变。</p>
</td>
</tr>
<tr id="row2074316127585"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p7858135417578"><a name="p7858135417578"></a><a name="p7858135417578"></a><a href="torch_npu-npu-enable_deterministic_with_backward.md">torch_npu.npu.enable_deterministic_with_backward</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1285825435710"><a name="p1285825435710"></a><a name="p1285825435710"></a>开启“确定性”功能。确定性算法是指在模型的前向传播过程中，每次输入相同，输出也相同。确定性算法可以避免模型在每次前向传播时产生的小随机误差累积，在需要重复测试或比较模型性能时非常有用。</p>
</td>
</tr>
<tr id="row16743712125815"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16134251175720"><a name="p16134251175720"></a><a name="p16134251175720"></a><a href="torch_npu-npu-disable_deterministic_with_backward.md">torch_npu.npu.disable_deterministic_with_backward</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p7134205145719"><a name="p7134205145719"></a><a name="p7134205145719"></a>关闭“确定性”功能。确定性算法是指在模型的前向传播过程中，每次输入相同，输出也相同。确定性算法可以避免模型在每次前向传播时产生的小随机误差累积，在需要重复测试或比较模型性能时非常有用。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-empty_with_swapped_memory.md">torch_npu.empty_with_swapped_memory</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>申请一个device信息为NPU且实际内存在host侧的特殊Tensor。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-erase_stream.md">torch_npu.erase_stream</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Tensor通过<code>record_stream</code>在内存池上添加的已被stream使用的标记后，可以通过该接口移除该标记。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_gather_sparse_index.md">torch_npu.npu_gather_sparse_index</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>从输入Tensor的指定维度，按照<code>index</code>中的下标序号提取元素，保存到输出Tensor中。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine.md">torch_npu.npu_moe_distribute_combine</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>先进行reduce_scatterv通信，再进行alltoallv通信，最后将接收的数据整合（乘权重再相加）。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_dispatch.md">torch_npu.npu_moe_distribute_dispatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对Token数据先进行量化（可选），再进行EP（Expert Parallelism）域的alltoallv通信，再进行TP（Tensor Parallelism）域的allgatherv通信（可选）。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine_v2.md">torch_npu.npu_moe_distribute_combine_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>先进行reduce_scatterv通信，再进行alltoallv通信，最后将接收的数据整合（乘权重再相加）。</p>
</td>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_dispatch_v2.md">torch_npu.npu_moe_distribute_dispatch_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对Token数据先进行量化（可选），再进行EP（Expert Parallelism）域的alltoallv通信，再进行TP（Tensor Parallelism）域的allgatherv通信（可选）。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_gating_top_k.md">torch_npu.npu_moe_gating_top_k</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_init_routing_v2.md">torch_npu.npu_moe_init_routing_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE（Mixture of Expert）的routing计算，根据<a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a>的计算结果做routing处理，支持不量化和动态量化模式。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_dequant_swiglu_quant.md">torch_npu.npu_dequant_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>对张量x做dequant反量化+swiglu+quant量化操作，同时支持分组。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_kv_rmsnorm_rope_cache.md">torch_npu.npu_kv_rmsnorm_rope_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_interleave_rope.md">torch_npu.npu_interleave_rope</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>针对单输入x进行旋转位置编码。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_re_routing.md">torch_npu.npu_moe_re_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家序重新排列。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-matmul_checksum.md">torch_npu.matmul_checksum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>提供基于原生torch.matmul和Tensor.matmul接口的aicore错误硬件故障接口，内部执行矩阵计算结果校验过程，并对校验误差和实时计算的校验门限进行对比，判断校验误差是否超越门限，若超越则认为发生了aicore错误。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_alltoallv_gmm.md">torch_npu.npu_alltoallv_gmm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE网络中，完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_gmm_alltoallv.md">torch_npu.npu_gmm_alltoallv</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE网络中，完成路由专家GroupedMatMul、AlltoAllv融合并实现与共享专家MatMul并行融合，先计算后通信。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine_add_rms_norm.md">torch_npu.npu_moe_distribute_combine_add_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>完成moe_distribute_combine+add+rms_norm融合。需与torch_npu.npu_moe_distribute_dispatch配套使用，相当于按npu_moe_distribute_dispatch算子收集数据的路径原路返回后对数据进行add_rms_norm操作。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_transpose_batchmatmul.md">torch_npu.npu_transpose_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>完成张量input与张量weight的矩阵乘计算。</p>
</td>
</tr>
</tbody>
</table>

