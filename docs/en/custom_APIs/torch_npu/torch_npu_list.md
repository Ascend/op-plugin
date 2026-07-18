# torch_npu APIs

This section describes common custom APIs, including tensor creation and computation-related operations.

**Table 1** torch_npu APIs

<a name="table1849611717116"></a>
<table><thead align="left"><tr id="row10496101716111"><th class="cellrowborder" valign="top" width="38.61%" id="mcps1.2.3.1.1"><p id="p1649713174119"><a name="p1649713174119"></a><a name="p1649713174119"></a>API</p>
</th>
<th class="cellrowborder" valign="top" width="61.39%" id="mcps1.2.3.1.2"><p id="p9497217151115"><a name="p9497217151115"></a><a name="p9497217151115"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row1149711715114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13497217191119"><a name="p13497217191119"></a><a name="p13497217191119"></a><a href="(beta)torch_npu-_npu_dropout.md">(beta)torch_npu._npu_dropout</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p349751701118"><a name="p349751701118"></a><a name="p349751701118"></a>Counts dropout results without using a random seed.</p>
</td>
</tr>
<tr id="row13497417121111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p11170526121211"><a name="p11170526121211"></a><a name="p11170526121211"></a><a href="(beta)torch_npu-copy_memory_.md">(beta)torch_npu.copy_memory_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p74971174115"><a name="p74971174115"></a><a name="p74971174115"></a>Copies elements from the source tensor <code>src</code> into the target tensor <code>self</code> and returns <code>self</code> in place.</p>
</td>
</tr>
<tr id="row949751712110"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p164971017121112"><a name="p164971017121112"></a><a name="p164971017121112"></a><a href="(beta)torch_npu-empty_with_format.md">(beta)torch_npu.empty_with_format</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p8497111791114"><a name="p8497111791114"></a><a name="p8497111791114"></a>Returns a tensor filled with uninitialized data.</p>
</td>
</tr>
<tr id="row949712178110"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5497817161113"><a name="p5497817161113"></a><a name="p5497817161113"></a><a href="(beta)torch_npu-fast_gelu.md">(beta)torch_npu.fast_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p9497101717113"><a name="p9497101717113"></a><a name="p9497101717113"></a>Computes the forward result of <code>FastGelu</code> for each input element by using the Fast Gaussian Error Linear Units (FastGELU) activation function.</p>
</td>
</tr>
<tr id="row17497217181119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16497617191116"><a name="p16497617191116"></a><a name="p16497617191116"></a><a href="(beta)torch_npu-npu_alloc_float_status.md">(beta)torch_npu.npu_alloc_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p6497141711115"><a name="p6497141711115"></a><a name="p6497141711115"></a>Allocates a tensor dedicated to storing floating-point operation status flags. This tensor is used to record overflow status during subsequent computations.</p>
</td>
</tr>
<tr id="row104977172114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1849701710119"><a name="p1849701710119"></a><a name="p1849701710119"></a><a href="(beta)torch_npu-npu_anchor_response_flags.md">(beta)torch_npu.npu_anchor_response_flags</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p69023133817"><a name="en-us_topic_0000001655404257_p69023133817"></a><a name="en-us_topic_0000001655404257_p69023133817"></a>Generates anchor response flags in a single feature map.</p>
</td>
</tr>
<tr id="row1120483231119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p152051325111"><a name="p152051325111"></a><a name="p152051325111"></a><a href="(beta)torch_npu-npu_apply_adam.md">(beta)torch_npu.npu_apply_adam</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18205193215119"><a name="p18205193215119"></a><a name="p18205193215119"></a>Obtains the computation results of the Adam optimizer.</p>
</td>
</tr>
<tr id="row1920533291116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13205173251112"><a name="p13205173251112"></a><a name="p13205173251112"></a><a href="(beta)torch_npu-npu_batch_nms.md">(beta)torch_npu.npu_batch_nms</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p820523261113"><a name="p820523261113"></a><a name="p820523261113"></a>Performs multi-batch and multi-class Non-Maximum Suppression (NMS) by evaluating and sorting bounding box scores, and removes redundant input boxes based on the <code>iou_threshold</code> to improve detection accuracy. This operation suppresses non-maximum elements by searching for local maxima, commonly used for detection-based models in computer vision tasks.</p>
</td>
</tr>
<tr id="row162057328112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p11205193211112"><a name="p11205193211112"></a><a name="p11205193211112"></a><a href="(beta)torch_npu-npu_bert_apply_adam.md">(beta)torch_npu.npu_bert_apply_adam</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12061132181116"><a name="p12061132181116"></a><a name="p12061132181116"></a>Obtains the computation results of the Adam optimizer for the BERT model.</p>
</td>
</tr>
<tr id="row182061132151115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p52061332161110"><a name="p52061332161110"></a><a name="p52061332161110"></a><a href="(beta)torch_npu-npu_bmmV2.md">(beta)torch_npu.npu_bmmV2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p1582323323616"><a name="en-us_topic_0000001655404257_p1582323323616"></a><a name="en-us_topic_0000001655404257_p1582323323616"></a>Multiplies matrix <code>a</code> by matrix <code>b</code> to produce matrix <code>a * b</code>.</p>
</td>
</tr>
<tr id="row520615328117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p020653281114"><a name="p020653281114"></a><a name="p020653281114"></a><a href="(beta)torch_npu-npu_bounding_box_decode.md">(beta)torch_npu.npu_bounding_box_decode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p2640134717250"><a name="en-us_topic_0000001655404257_p2640134717250"></a><a name="en-us_topic_0000001655404257_p2640134717250"></a>Generates bounding boxes based on <code>rois</code> and <code>deltas</code>. This is a custom Faster R-CNN operator.</p>
</td>
</tr>
<tr id="row657717509111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2577105001114"><a name="p2577105001114"></a><a name="p2577105001114"></a><a href="(beta)torch_npu-npu_bounding_box_encode.md">(beta)torch_npu.npu_bounding_box_encode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2577105019119"><a name="p2577105019119"></a><a name="p2577105019119"></a>Computes the coordinate changes between anchor boxes and ground-truth boxes. This is a custom Faster R-CNN operator.</p>
</td>
</tr>
<tr id="row989165331120"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p38965316111"><a name="p38965316111"></a><a name="p38965316111"></a><a href="(beta)torch_npu-npu_broadcast.md">(beta)torch_npu.npu_broadcast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p918313310142"><a name="en-us_topic_0000001655404257_p918313310142"></a><a name="en-us_topic_0000001655404257_p918313310142"></a>Returns a new view of <code>self</code> with singleton dimensions expanded, and the result is contiguous. The tensor can also be expanded by more dimensions, and new dimensions are added at the front.</p>
</td>
</tr>
<tr id="row589165316115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1489155321111"><a name="p1489155321111"></a><a name="p1489155321111"></a><a href="(beta)torch_npu-npu_ciou.md">(beta)torch_npu.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p749621223015"><a name="en-us_topic_0000001655404257_p749621223015"></a><a name="en-us_topic_0000001655404257_p749621223015"></a>Applies an NPU-based CIoU operation. A penalty term is added to DIoU to propose CIoU.</p>
</td>
</tr>
<tr id="row138914536113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1289453191113"><a name="p1289453191113"></a><a name="p1289453191113"></a><a href="(beta)torch_npu-npu_clear_float_status.md">(beta)torch_npu.npu_clear_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p289105341110"><a name="p289105341110"></a><a name="p289105341110"></a>Clears the status flags related to overflow detection.</p>
</td>
</tr>
<tr id="row158945317117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5901953141111"><a name="p5901953141111"></a><a name="p5901953141111"></a><a href="(beta)torch_npu-npu_confusion_transpose.md">(beta)torch_npu.npu_confusion_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p7908532119"><a name="p7908532119"></a><a name="p7908532119"></a>Obfuscates reshape and transpose operations.</p>
</td>
</tr>
<tr id="row2090953141119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1690125311117"><a name="p1690125311117"></a><a name="p1690125311117"></a><a href="(beta)torch_npu-npu_conv_transpose2d.md">(beta)torch_npu.npu_conv_transpose2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p122041623817"><a name="en-us_topic_0000001655404257_p122041623817"></a><a name="en-us_topic_0000001655404257_p122041623817"></a>Applies a 2D transposed convolution operator to an input image composed of multiple input planes. Sometimes, this process is also referred to as "deconvolution".</p>
</td>
</tr>
<tr id="row49065381112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p189035310112"><a name="p189035310112"></a><a name="p189035310112"></a><a href="(beta)torch_npu-npu_conv2d.md">(beta)torch_npu.npu_conv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1790753191113"><a name="p1790753191113"></a><a name="p1790753191113"></a>Applies a 2D convolution to an input image composed of multiple input planes.</p>
</td>
</tr>
<tr id="row109045310113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p49035381110"><a name="p49035381110"></a><a name="p49035381110"></a><a href="(beta)torch_npu-npu_conv3d.md">(beta)torch_npu.npu_conv3d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18901253111116"><a name="p18901253111116"></a><a name="p18901253111116"></a>Applies a 3D convolution to an input image composed of multiple input planes.</p>
</td>
</tr>
<tr id="row0901534111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1090253101111"><a name="p1090253101111"></a><a name="p1090253101111"></a><a href="(beta)torch_npu-npu_convolution.md">(beta)torch_npu.npu_convolution</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p890053121116"><a name="p890053121116"></a><a name="p890053121116"></a>Applies a 2D or 3D convolution to an input image composed of multiple input planes.</p>
</td>
</tr>
<tr id="row9577205019119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p7577105010119"><a name="p7577105010119"></a><a name="p7577105010119"></a><a href="(beta)torch_npu-npu_convolution_transpose.md">(beta)torch_npu.npu_convolution_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p35776509119"><a name="p35776509119"></a><a name="p35776509119"></a>Applies a 2D or 3D transposed convolution operator to an input image composed of multiple input planes. Sometimes, this process is also referred to as "deconvolution".</p>
</td>
</tr>
<tr id="row75771250101115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p15771150151111"><a name="p15771150151111"></a><a name="p15771150151111"></a><a href="(beta)torch_npu-npu_deformable_conv2d.md">(beta)torch_npu.npu_deformable_conv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p1888192943717"><a name="en-us_topic_0000001655404257_p1888192943717"></a><a name="en-us_topic_0000001655404257_p1888192943717"></a>Computes the deformed convolution output using the expected input.</p>
</td>
</tr>
<tr id="row125771550121117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16577135019112"><a name="p16577135019112"></a><a name="p16577135019112"></a><a href="(beta)torch_npu-npu_diou.md">(beta)torch_npu.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p12796833183010"><a name="en-us_topic_0000001655404257_p12796833183010"></a><a name="en-us_topic_0000001655404257_p12796833183010"></a>Applies an NPU-based DIoU operation. Considering the distance between targets and the overlap ratio of distance and scope, different targets or boundaries must tend to be stable.</p>
</td>
</tr>
<tr id="row95771950161112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p35771150191112"><a name="p35771150191112"></a><a name="p35771150191112"></a><a href="(beta)torch_npu-npu_dtype_cast.md">(beta)torch_npu.npu_dtype_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p1221456142618"><a name="en-us_topic_0000001655404257_p1221456142618"></a><a name="en-us_topic_0000001655404257_p1221456142618"></a>Converts the data type (<code>dtype</code>) of a tensor.</p>
</td>
</tr>
<tr id="row195782506118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1857885061113"><a name="p1857885061113"></a><a name="p1857885061113"></a><a href="(beta)torch_npu-npu_format_cast.md">(beta)torch_npu.npu_format_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p68591917123"><a name="en-us_topic_0000001655404257_p68591917123"></a><a name="en-us_topic_0000001655404257_p68591917123"></a>Modifies the data layout format of an NPU tensor.</p>
</td>
</tr>
<tr id="row457855012117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p557813506114"><a name="p557813506114"></a><a name="p557813506114"></a><a href="(beta)torch_npu-npu_format_cast_.md">(beta)torch_npu.npu_format_cast_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p125789503117"><a name="p125789503117"></a><a name="p125789503117"></a>Modifies the data layout format of the <code>self</code> tensor in place, matching the format of <code>src</code>.</p>
</td>
</tr>
<tr id="row1757895091111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p17578195010114"><a name="p17578195010114"></a><a name="p17578195010114"></a><a href="(beta)torch_npu-npu_get_float_status.md">(beta)torch_npu.npu_get_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p8144193193519"><a name="en-us_topic_0000001655404257_p8144193193519"></a><a name="en-us_topic_0000001655404257_p8144193193519"></a>Obtains the overflow detection result.</p>
</td>
</tr>
<tr id="row3206113212114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p42065329112"><a name="p42065329112"></a><a name="p42065329112"></a><a href="(beta)torch_npu-npu_giou.md">(beta)torch_npu.npu_giou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p0835111286"><a name="en-us_topic_0000001655404257_p0835111286"></a><a name="en-us_topic_0000001655404257_p0835111286"></a>First computes the minimum enclosing area and IoU of two boxes, then calculates the proportion of the enclosing area that does not belong to either box, and finally subtracts this proportion from the IoU to obtain the GIoU.</p>
</td>
</tr>
<tr id="row5377163616111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p937723691115"><a name="p937723691115"></a><a name="p937723691115"></a><a href="(beta)torch_npu-npu_grid_assign_positive.md">(beta)torch_npu.npu_grid_assign_positive</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p258994112512"><a name="en-us_topic_0000001655404257_p258994112512"></a><a name="en-us_topic_0000001655404257_p258994112512"></a><span>Computes the position-sensitive candidate region pooling gradients.</span></p>
</td>
</tr>
<tr id="row14377936121116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1337753621111"><a name="p1337753621111"></a><a name="p1337753621111"></a><a href="(beta)torch_npu-npu_gru.md">(beta)torch_npu.npu_gru</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p798911210261"><a name="en-us_topic_0000001655404257_p798911210261"></a><a name="en-us_topic_0000001655404257_p798911210261"></a>Computes DynamicGRUV2.</p>
</td>
</tr>
<tr id="row557919470115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p157994710118"><a name="p157994710118"></a><a name="p157994710118"></a><a href="(beta)torch_npu-npu_indexing.md">(beta)torch_npu.npu_indexing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p191786291"><a name="en-us_topic_0000001655404257_p191786291"></a><a name="en-us_topic_0000001655404257_p191786291"></a>Slices the input tensor by using <code>begin</code> as the start index, <code>end</code> as the end index, and <code>strides</code> as the stride.</p>
</td>
</tr>
<tr id="row11579164761115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p195798473119"><a name="p195798473119"></a><a name="p195798473119"></a><a href="(beta)torch_npu-npu_iou.md">(beta)torch_npu.npu_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p10544194202310"><a name="en-us_topic_0000001655404257_p10544194202310"></a><a name="en-us_topic_0000001655404257_p10544194202310"></a>Computes the intersection over union (IoU) or intersection over foreground (IoF) based on the ground-truth boxes and predicted regions.</p>
</td>
</tr>
<tr id="row10579184741114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p35791247141119"><a name="p35791247141119"></a><a name="p35791247141119"></a><a href="(beta)torch_npu-npu_layer_norm_eval.md">(beta)torch_npu.npu_layer_norm_eval</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p75791547101117"><a name="p75791547101117"></a><a name="p75791547101117"></a>Computes the layer normalization result. The semantics are identical to those of <code>torch.nn.functional.layer_norm</code> and is optimized for NPUs.</p>
</td>
</tr>
<tr id="row11579104721117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p135794473113"><a name="p135794473113"></a><a name="p135794473113"></a><a href="(beta)torch_npu-npu_linear.md">(beta)torch_npu.npu_linear</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p8579144781110"><a name="p8579144781110"></a><a name="p8579144781110"></a>Multiplies matrix <code>a</code> by matrix <code>b</code> to produce matrix <code>a * b</code>.</p>
</td>
</tr>
<tr id="row9579184715119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p85793478116"><a name="p85793478116"></a><a name="p85793478116"></a><a href="(beta)torch_npu-npu_lstm.md">(beta)torch_npu.npu_lstm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p757912479111"><a name="p757912479111"></a><a name="p757912479111"></a>Computes DynamicRNN.</p>
</td>
</tr>
<tr id="row3579194701112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16579184717113"><a name="p16579184717113"></a><a name="p16579184717113"></a><a href="(beta)torch_npu-npu_max.md">(beta)torch_npu.npu_max</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1758054719115"><a name="p1758054719115"></a><a name="p1758054719115"></a>Computes the maximum values along <code>dim</code>. This API is similar to <code>torch.max</code> and is optimized for NPUs.</p>
</td>
</tr>
<tr id="row2580347161118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p12580124781118"><a name="p12580124781118"></a><a name="p12580124781118"></a><a href="(beta)torch_npu-npu_min.md">(beta)torch_npu.npu_min</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p1171133519333"><a name="en-us_topic_0000001655404257_p1171133519333"></a><a name="en-us_topic_0000001655404257_p1171133519333"></a>Computes the minimum values along <code>dim</code>. This API is similar to <code>torch.min</code> and is optimized for NPUs.</p>
</td>
</tr>
<tr id="row115511438538"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p75511930538"><a name="p75511930538"></a><a name="p75511930538"></a><a href="(beta)torch_npu-npu_mish.md">(beta)torch_npu.npu_mish</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12741142714357"><a name="p12741142714357"></a><a name="p12741142714357"></a>Computes the hyperbolic tangent of <code>self</code> element-wise.</p>
</td>
</tr>
<tr id="row558010477110"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p17580147201113"><a name="p17580147201113"></a><a name="p17580147201113"></a><a href="(beta)torch_npu-npu_nms_rotated.md">(beta)torch_npu.npu_nms_rotated</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p2907120192218"><a name="en-us_topic_0000001655404257_p2907120192218"></a><a name="en-us_topic_0000001655404257_p2907120192218"></a>Selects a subset of rotated bounding boxes in descending order of scores.</p>
</td>
</tr>
<tr id="row1158054718119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1858004714112"><a name="p1858004714112"></a><a name="p1858004714112"></a><a href="(beta)torch_npu-npu_nms_v4.md">(beta)torch_npu.npu_nms_v4</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p296295602119"><a name="en-us_topic_0000001655404257_p296295602119"></a><a name="en-us_topic_0000001655404257_p296295602119"></a>Selects a subset of bounding boxes in descending order of scores.</p>
</td>
</tr>
<tr id="row85801547131119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p105802478112"><a name="p105802478112"></a><a name="p105802478112"></a><a href="(beta)torch_npu-npu_nms_with_mask.md">(beta)torch_npu.npu_nms_with_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1358054721113"><a name="p1358054721113"></a><a name="p1358054721113"></a>Generates values <code>0</code> or <code>1</code>, which are used by the NMS operator to determine valid bits.</p>
</td>
</tr>
<tr id="row858014718113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p25801047101117"><a name="p25801047101117"></a><a name="p25801047101117"></a><a href="(beta)torch_npu-npu_one_hot.md">(beta)torch_npu.npu_one_hot</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5580547191120"><a name="p5580547191120"></a><a name="p5580547191120"></a>Returns a one-hot tensor. The positions indicated by indices in <code>input</code> take the <code>on_value</code>, whereas all other positions take the <code>off_value</code>.</p>
</td>
</tr>
<tr id="row2037715364114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p173771360112"><a name="p173771360112"></a><a name="p173771360112"></a><a href="(beta)torch_npu-npu_pad.md">(beta)torch_npu.npu_pad</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p93771036191114"><a name="p93771036191114"></a><a name="p93771036191114"></a>Pads a tensor.</p>
</td>
</tr>
<tr id="row1537743614119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p737793615117"><a name="p737793615117"></a><a name="p737793615117"></a><a href="(beta)torch_npu-npu_ps_roi_pooling.md">(beta)torch_npu.npu_ps_roi_pooling</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1037763620111"><a name="p1037763620111"></a><a name="p1037763620111"></a>Performs position-sensitive ROI pooling.</p>
</td>
</tr>
<tr id="row2037743613112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p12377203601118"><a name="p12377203601118"></a><a name="p12377203601118"></a><a href="(beta)torch_npu-npu_ptiou.md">(beta)torch_npu.npu_ptiou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1593155184019"><a name="p1593155184019"></a><a name="p1593155184019"></a>Computes the intersection over union (IoU) or intersection over foreground (IoF) based on the ground-truth boxes and predicted regions.</p>
</td>
</tr>
<tr id="row1337793616112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p133781436181115"><a name="p133781436181115"></a><a name="p133781436181115"></a><a href="(beta)torch_npu-npu_random_choice_with_mask.md">(beta)torch_npu.npu_random_choice_with_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p219311346269"><a name="en-us_topic_0000001655404257_p219311346269"></a><a name="en-us_topic_0000001655404257_p219311346269"></a>Shuffles the indices of non-zero elements.</p>
</td>
</tr>
<tr id="row837833671119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1137810369111"><a name="p1137810369111"></a><a name="p1137810369111"></a><a href="(beta)torch_npu-npu_reshape.md">(beta)torch_npu.npu_reshape</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14378123610113"><a name="p14378123610113"></a><a name="p14378123610113"></a>Reshapes a tensor. This operation only changes the tensor shape while its data remains unchanged.</p>
</td>
</tr>
<tr id="row11378336161116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p83782360112"><a name="p83782360112"></a><a name="p83782360112"></a><a href="(beta)torch_npu-npu_roi_align.md">(beta)torch_npu.npu_roi_align</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p11378133620112"><a name="p11378133620112"></a><a name="p11378133620112"></a>Obtains the candidate region feature matrix from a feature map. This is a custom Faster R-CNN operator.</p>
</td>
</tr>
<tr id="row168411042171118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p11841184214110"><a name="p11841184214110"></a><a name="p11841184214110"></a><a href="(beta)torch_npu-npu_rotated_iou.md">(beta)torch_npu.npu_rotated_iou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p384134214117"><a name="p384134214117"></a><a name="p384134214117"></a>Computes the IoU of rotated bounding boxes.</p>
</td>
</tr>
<tr id="row1284118427112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p0841184211115"><a name="p0841184211115"></a><a name="p0841184211115"></a><a href="(beta)torch_npu-npu_rotated_overlaps.md">(beta)torch_npu.npu_rotated_overlaps</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p20841164221110"><a name="p20841164221110"></a><a name="p20841164221110"></a>Computes the overlap area of rotated bounding boxes.</p>
</td>
</tr>
<tr id="row68411842121111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1384134221114"><a name="p1384134221114"></a><a name="p1384134221114"></a><a href="(beta)torch_npu-npu_sign_bits_pack.md">(beta)torch_npu.npu_sign_bits_pack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p484114213112"><a name="p484114213112"></a><a name="p484114213112"></a>Packs float-type 1-bit Adam parameters into uint8.</p>
</td>
</tr>
<tr id="row128412042131111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p784116424113"><a name="p784116424113"></a><a name="p784116424113"></a><a href="(beta)torch_npu-npu_sign_bits_unpack.md">(beta)torch_npu.npu_sign_bits_unpack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p58411242191112"><a name="p58411242191112"></a><a name="p58411242191112"></a>Unpacks uint8-type 1-bit Adam parameters into float.</p>
</td>
</tr>
<tr id="row17842154219116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1384284211115"><a name="p1384284211115"></a><a name="p1384284211115"></a><a href="(beta)torch_npu-npu_silu.md">(beta)torch_npu.npu_silu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p11842194231120"><a name="p11842194231120"></a><a name="p11842194231120"></a>Computes the Swish activation function of <code>self</code>. Swish is an activation function defined as $[x * \text{sigmoid}(x)]$.</p>
</td>
</tr>
<tr id="row68428425112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1842204218113"><a name="p1842204218113"></a><a name="p1842204218113"></a><a href="(beta)torch_npu-npu_slice.md">(beta)torch_npu.npu_slice</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p0842124210117"><a name="p0842124210117"></a><a name="p0842124210117"></a>Extracts a slice from a tensor.</p>
</td>
</tr>
<tr id="row7842204261115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13842442161112"><a name="p13842442161112"></a><a name="p13842442161112"></a><a href="(beta)torch_npu-npu_softmax_cross_entropy_with_logits.md">(beta)torch_npu.npu_softmax_cross_entropy_with_logits</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1842144291119"><a name="p1842144291119"></a><a name="p1842144291119"></a>Computes the softmax cross-entropy cost.</p>
</td>
</tr>
<tr id="row1084224211115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p188421442111120"><a name="p188421442111120"></a><a name="p188421442111120"></a><a href="(beta)torch_npu-npu_sort_v2.md">(beta)torch_npu.npu_sort_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5842442111110"><a name="p5842442111110"></a><a name="p5842442111110"></a>Sorts the elements of the input tensor in ascending order along the specified dimension without returning indices. If <code>dim</code> is not specified, the last dimension of the input is selected. If <code>descending</code> is set to <code>True</code>, the elements are sorted in descending order by value.</p>
</td>
</tr>
<tr id="row884254219118"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1784284271118"><a name="p1784284271118"></a><a name="p1784284271118"></a><a href="(beta)torch_npu-npu_transpose.md">(beta)torch_npu.npu_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1884294261114"><a name="p1884294261114"></a><a name="p1884294261114"></a>Returns a view of the original tensor with its dimensions permuted, and the result is contiguous.</p>
</td>
</tr>
<tr id="row118421942161116"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p188423429111"><a name="p188423429111"></a><a name="p188423429111"></a><a href="(beta)torch_npu-npu_yolo_boxes_encode.md">(beta)torch_npu.npu_yolo_boxes_encode</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p68421942191116"><a name="p68421942191116"></a><a name="p68421942191116"></a>Generates bounding boxes based on YOLO anchor boxes and ground-truth boxes. This is a custom MMDetection operator.</p>
</td>
</tr>
<tr id="row6842164251115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p19842134241119"><a name="p19842134241119"></a><a name="p19842134241119"></a><a href="(beta)torch_npu-npu_fused_attention_score.md">(beta)torch_npu.npu_fused_attention_score</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p98421642101113"><a name="p98421642101113"></a><a name="p98421642101113"></a>Implements the fused computation logic of Transformer attention scores, primarily fusing operations such as <code>matmul</code>, <code>transpose</code>, <code>add</code>, <code>softmax</code>, <code>dropout</code>, <code>batchmatmul</code>, and <code>permute</code>.</p>
</td>
</tr>
<tr id="row2378193620119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p9378836161118"><a name="p9378836161118"></a><a name="p9378836161118"></a><a href="(beta)torch_npu-npu_multi_head_attention.md">(beta)torch_npu.npu_multi_head_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14378143615111"><a name="p14378143615111"></a><a name="p14378143615111"></a>Implements the Multi-Head Attention (MHA) computation logic in the Transformer module.</p>
</td>
</tr>
<tr id="row14545161172716"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p854661182710"><a name="p854661182710"></a><a name="p854661182710"></a><a href="(beta)torch_npu-npu_rms_norm.md">(beta)torch_npu.npu_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p125461318273"><a name="p125461318273"></a><a name="p125461318273"></a>The RMSNorm operator is a normalization operation commonly used in foundation models. Compared with the LayerNorm operator, it removes the mean subtraction step.</p>
</td>
</tr>
<tr id="row2791419328"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1990671211278"><a name="p1990671211278"></a><a name="p1990671211278"></a><a href="(beta)torch_npu-npu_dropout_with_add_softmax.md">(beta)torch_npu.npu_dropout_with_add_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p15906141216278"><a name="p15906141216278"></a><a name="p15906141216278"></a>Implements the functions of <code>axpy_v2</code>, <code>softmax_v2</code>, and <code>drop_out_domask_v3</code>.</p>
</td>
</tr>
<tr id="row3892520439"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2991782322389"><a name="p2991782322389"></a><a name="p2991782322389"></a><a href="torch_npu-npu_rms_norm_quant.md">torch_npu.npu_rms_norm_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p16917252327389"><a name="p16917252327389"></a><a name="p16917252327389"></a>Fuses the RmsNorm and Quantize operators to reduce data transfer operations.</p>
</td>
</tr>
<tr id="row17801219526"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p89062124277"><a name="p89062124277"></a><a name="p89062124277"></a><a href="torch_npu-npu_rotary_mul.md">torch_npu.npu_rotary_mul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p69061812102715"><a name="p69061812102715"></a><a name="p69061812102715"></a>Implements rotary position embedding (RoPE).</p>
</td>
</tr>
<tr id="row58015194219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1590618121275"><a name="p1590618121275"></a><a name="p1590618121275"></a><a href="torch_npu-npu_scaled_masked_softmax.md">torch_npu.npu_scaled_masked_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1290691218270"><a name="p1290691218270"></a><a name="p1290691218270"></a>Computes the Softmax result after scaling the input tensor <code>x</code> and masking it based on <code>mask</code>.</p>
</td>
</tr>
<tr id="row480181910211"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p137651452352"><a name="p137651452352"></a><a name="p137651452352"></a><a href="(beta)torch_npu-npu_swiglu.md">(beta)torch_npu.npu_swiglu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14325153814812"><a name="p14325153814812"></a><a name="p14325153814812"></a>Provides the SwiGLU activation function.</p>
</td>
</tr>
<tr id="row178010191727"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p13901449479"><a name="p13901449479"></a><a name="p13901449479"></a><a href="(beta)torch_npu-one_.md">(beta)torch_npu.one_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001726825729_en-us_topic_0000001655404257_p5268123310116"><a name="en-us_topic_0000001726825729_en-us_topic_0000001655404257_p5268123310116"></a><a name="en-us_topic_0000001726825729_en-us_topic_0000001655404257_p5268123310116"></a>Fills the <code>self</code> tensor with <code>1</code>s.</p>
</td>
</tr>
<tr id="row106843515114"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p168510515113"><a name="p168510515113"></a><a name="p168510515113"></a><a href="torch_npu-npu_group_norm_swish.md">torch_npu.npu_group_norm_swish</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p19342196105315"><a name="p19342196105315"></a><a name="p19342196105315"></a>Performs fused computation of group normalization (GroupNorm) and Swish activation for the input tensor <code>input</code>. This API generates the group normalization result <code>y</code>, mean <code>mean</code>, reciprocal standard deviation <code>rstd</code>, and the Swish-activated output.</p>
</td>
</tr>
<tr id="row10636159121815"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p763620598183"><a name="p763620598183"></a><a name="p763620598183"></a><a href="torch_npu-npu_cross_entropy_loss.md">torch_npu.npu_cross_entropy_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18198201124816"><a name="p18198201124816"></a><a name="p18198201124816"></a>Fuses the <code>log_softmax</code> and <code>nll_loss</code> operations from the native <code>CrossEntropyLoss</code> framework to reduce memory utilization during computation. It also supports z-loss computation.</p>
</td>
</tr>
<tr id="row142536475869"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p142536475870"><a name="p142536475870"></a><a name="p142536475870"></a><a href="torch_npu-npu_add_rms_norm_quant.md">torch_npu.npu_add_rms_norm_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p142536475871"><a name="p142536475871"></a><a name="p142536475871"></a>Fuses the Add operator before RMSNorm and the Quantize operator after RMSNorm, reducing data transfer operations.</p>
</td>
</tr>
<tr id="row724843692712"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p162481336102714"><a name="p162481336102714"></a><a name="p162481336102714"></a><a href="torch_npu-npu_advance_step_flashattn.md">torch_npu.npu_advance_step_flashattn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p13248936102717"><a name="p13248936102717"></a><a name="p13248936102717"></a>Implements the <code>advance_step_flashattn</code> functionality from the vLLM library on the NPU. It performs in-place updates of <code>input_tokens</code>, <code>input_positions</code>, <code>seq_lens</code>, and <code>slot_mapping</code> during each generation step.</p>
</td>
</tr>
<tr id="row1526115194711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p9360134415216"><a name="p9360134415216"></a><a name="p9360134415216"></a><a href="torch_npu-npu_all_gather_base_mm.md">torch_npu.npu_all_gather_base_mm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001694916914_p444492882116"><a name="en-us_topic_0000001694916914_p444492882116"></a><a name="en-us_topic_0000001694916914_p444492882116"></a>Fuses allgather and matrix multiplication (MatMul) operations in tensor parallelism (TP) scenarios. The fused kernel implements pipelined parallelism between computation and communication.</p>
</td>
</tr>
<tr id="row952719519471"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18248651143510"><a name="p18248651143510"></a><a name="p18248651143510"></a><a href="torch_npu-npu_anti_quant.md">torch_npu.npu_anti_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001850161621_p0996174814315"><a name="en-us_topic_0000001850161621_p0996174814315"></a><a name="en-us_topic_0000001850161621_p0996174814315"></a>Dequantizes <code>int4</code> or <code>int8</code> data into <code>float16</code> or <code>bfloat16</code> data. When the input data type is <code>int4</code>, every eight elements are treated as one <code>int32</code> element.</p>
</td>
</tr>
<tr id="row952719519471"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18248651143510"><a name="p18248651143510"></a><a name="p18248651143510"></a><a href="torch_npu-npu_attention_to_ffn.md">torch_npu.npu_attention_to_ffn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001850161621_p0996174814315"><a name="en-us_topic_0000001850161621_p0996174814315"></a><a name="en-us_topic_0000001850161621_p0996174814315"></a>Sends token data from the Attention node to the FFN node.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_attention_update.md">torch_npu.npu_attention_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Updates the local intermediate variables <code>lse</code> and <code>local_out</code> output by the PagedAttention (PA) operator across each Sequence Parallelism (SP) domain into global results.</p>
</td>
</tr>
<tr id="row_block_sparse_attention"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10326218132921"><a name="p10326218132921"></a><a name="p10326218132921"></a><a href="torch_npu-npu_block_sparse_attention.md">torch_npu.npu_block_sparse_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p10326218132922"><a name="p10326218132922"></a><a name="p10326218132922"></a>Computes <code>BlockSparseAttention</code>. This sparse attention mechanism supports block-level sparsity. It uses <code>block_sparse_mask</code> to specify the KV blocks selected by each Q block to achieve efficient attention computation.</p>
</td>
</tr>
<tr id="row16527145114477"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1338142581"><a name="p1338142581"></a><a name="p1338142581"></a><a href="torch_npu-npu_convert_weight_to_int4pack.md">torch_npu.npu_convert_weight_to_int4pack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p16338242582"><a name="p16338242582"></a><a name="p16338242582"></a>Packs an <code>int32</code> input tensor into the <code>int4</code> data type. Every eight <code>int4</code> elements are carried by a single <code>int32</code> element and stored in an interleaved format.</p>
</td>
</tr>
<tr id="row194591911194818"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p633812455816"><a name="p633812455816"></a><a name="p633812455816"></a><a href="torch_npu-npu_dynamic_quant.md">torch_npu.npu_dynamic_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p93381541583"><a name="p93381541583"></a><a name="p93381541583"></a>Performs pertoken symmetric dynamic quantization on the input tensor.</p>
</td>
</tr>
<tr id="row44591911184819"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p820225218412"><a name="p820225218412"></a><a name="p820225218412"></a><a href="torch_npu-npu_dynamic_quant_asymmetric.md">torch_npu.npu_dynamic_quant_asymmetric</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p271341016220"><a name="p271341016220"></a><a name="p271341016220"></a>Performs pertoken asymmetric dynamic quantization on the input tensor. The last input dimension corresponds to a token, and each token is quantized as a group.</p>
</td>
</tr>
<tr id="row1245931134815"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1055135819577"><a name="p1055135819577"></a><a name="p1055135819577"></a><a href="torch_npu-npu_fast_gelu.md">torch_npu.npu_fast_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p135525810574"><a name="p135525810574"></a><a name="p135525810574"></a>Applies the Fast Gaussian Error Linear Units (FastGelu) activation function for each element in the input tensor. The input must be a tensor with any valid shape.</p>
</td>
</tr>
<tr id="row545919116481"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p099812152913"><a name="p099812152913"></a><a name="p099812152913"></a><a href="torch_npu-npu_ffn.md">torch_npu.npu_ffn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1998152112915"><a name="p1998152112915"></a><a name="p1998152112915"></a>Provides Mixture-of-Experts Feed-Forward Network (MoeFFN) and Feed-Forward Network (FFN) computation features. FFN is used when there are no expert groups (<code>expert_tokens</code> is empty) and MoeFFN is used when there are expert groups (<code>expert_tokens</code> is not empty).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p43681016105618"></a><a name="p43681016105618"></a><a href="torch_npu-npu_fused_infer_attention_score.md">torch_npu.npu_fused_infer_attention_score</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p836911165560"><a name="p836911165560"></a><a name="p836911165560"></a>Adapts to the <code>FlashAttention</code> operator in the incremental and full inference scenarios, supporting both full computation (<code>PromptFlashAttention</code>) and incremental computation (<code>IncreFlashAttention</code>).</p>
<tr id="row545919116481"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p099812152913"><a name="p099812152913"></a><a name="p099812152913"></a><a href="torch_npu-npu_ffn_to_attention.md">torch_npu.npu_ffn_to_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1998152112915"><a name="p1998152112915"></a><a name="p1998152112915"></a>Sends token data from the FFN node to the Attention node.</p>
</td>
</tr>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p43681016105618"></a><a name="p43681016105618"></a><a href="torch_npu-npu_fused_infer_attention_score_v2.md">torch_npu.npu_fused_infer_attention_score_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p836911165560"><a name="p836911165560"></a><a name="p836911165560"></a>Adapts to the <code>FlashAttention</code> operator in the incremental and full inference scenarios, supporting both full computation (<code>PromptFlashAttention</code>) and incremental computation (<code>IncreFlashAttention</code>). Added support for MultiHead Latent Attention (MLA) full quantization.</p>
</td>
</tr>
<tr id="row14601311184810"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1799710217294"><a name="p1799710217294"></a><a name="p1799710217294"></a><a href="torch_npu-npu_fusion_attention.md">torch_npu.npu_fusion_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p129971221112919"><a name="p129971221112919"></a><a name="p129971221112919"></a>Implements fused computation of the Transformer attention score.</p>
</td>
</tr>
<tr id="row14601311184810"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1799710217294"><a name="p1799710217294"></a><a name="p1799710217294"></a><a href="torch_npu-npu_fusion_attention_v3.md">torch_npu.npu_fusion_attention_v3</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p129971221112919"><a name="p129971221112919"></a><a name="p129971221112919"></a>Implements fused computation of the Transformer attention score. Graph mode is supported.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p1720095217417"></a><a name="p1720095217417"></a><a href="torch_npu-npu_gelu.md">torch_npu.npu_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p9269284217"><a name="p9269284217"></a><a name="p9269284217"></a>Computes the Gaussian Error Linear Unit (GELU) activation function.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p1720095217417"></a><a name="p1720095217417"></a><a href="torch_npu-npu_gelu_mul.md">torch_npu.npu_gelu_mul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p9269284217"><a name="p9269284217"></a><a name="p9269284217"></a>Performs fused computation of GELU and MUL. When the last axis of <code>input</code> is 32-byte aligned, this API performs fused computation of GELU and MUL to improve performance. When the last axis is not 32-byte aligned, operator concatenation is recommended. That is, it performs computation step by step using the formulas.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p5338134195811"></a><a name="p5338134195811"></a><a href="torch_npu-npu_group_norm_silu.md">torch_npu.npu_group_norm_silu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5338546582"><a name="p5338546582"></a><a name="p5338546582"></a>Computes group normalization for the input tensor <code>self</code>. This API returns the group normalization result <code>out</code>, the mean <code>meanOut</code>, the reciprocal of the standard deviation <code>rstdOut</code>, and the SILU output.</p>
</td>
</tr>
<tr id="row174571042727"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10203752943"><a name="p10203752943"></a><a name="p10203752943"></a><a href="torch_npu-npu_group_quant.md">torch_npu.npu_group_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p4382954217"><a name="p4382954217"></a><a name="p4382954217"></a>Performs group-wise quantization on the input tensor.</p>
</td>
</tr>
<tr id="row1045718422215"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p64370382369"><a name="p64370382369"></a><a name="p64370382369"></a><a href="torch_npu-npu_grouped_matmul.md">torch_npu.npu_grouped_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1437638113619"><a name="p1437638113619"></a><a name="p1437638113619"></a><span>Provides an efficient method to perform grouped computation of multiple matrix multiplication (MatMul) operations.</span></p>
</td>
</tr>
<tr id="row1231245645712"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p548217963045654"><a name="p548217963045654"></a><a name="p548217963045654"></a><a href="torch_npu-npu_grouped_matmul_swiglu_quant_v2.md">torch_npu.npu_grouped_matmul_swiglu_quant_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5420573896217596"><a name="p5420573896217596"></a><a name="p5420573896217596"></a><span>Provides an efficient method to perform fused computation of grouped matrix multiplication (<code>GroupedMatMul</code>), dequantization (<code>dequant</code>), the <code>SwiGLU</code> activation function, and quantization (<code>quant</code>).</span></p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_incre_flash_attention.md">torch_npu.npu_incre_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Implements incremental FlashAttention (FA).</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_kv_quant_sparse_flash_attention.md">torch_npu.npu_kv_quant_sparse_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Implements fake quantization for Sparse Flash Attention.</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_lightning_indexer.md">torch_npu.npu_lightning_indexer</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Obtains the Top-$k$ positions for each token (full quantization).</p>
</td>
</tr>
<tr id="row1457134215217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p258365124912"><a name="p258365124912"></a><a name="p258365124912"></a><a href="torch_npu-npu_mla_prolog.md">torch_npu.npu_mla_prolog</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>Performs computation during Multi-Head Latent Attention (MLA) preprocessing in inference scenarios.</span></p>
</td>
</tr>
<tr id="row1457134215217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p258365124912"><a name="p258365124912"></a><a name="p258365124912"></a><a href="torch_npu-npu_mla_prolog_v2.md">torch_npu.npu_mla_prolog_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>Performs computation during Multi-Head Latent Attention (MLA) preprocessing in inference scenarios.</span></p>
</td>
</tr>
<tr id="row1457134215217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p258365124912"><a name="p258365124912"></a><a name="p258365124912"></a><a href="torch_npu-npu_mla_prolog_v3.md">torch_npu.npu_mla_prolog_v3</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>Performs computation during Multi-Head Latent Attention (MLA) preprocessing in inference scenarios.</span></p>
</td>
</tr>
<tr id="row125419366217"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18998182119294"><a name="p18998182119294"></a><a name="p18998182119294"></a><a href="torch_npu-npu_mm_all_reduce_base.md">torch_npu.npu_mm_all_reduce_base</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001721582972_p2122194522714"><a name="en-us_topic_0000001721582972_p2122194522714"></a><a name="en-us_topic_0000001721582972_p2122194522714"></a>Fuses matrix multiplication (MatMul) and all_reduce collective communication operations in tensor parallelism (TP) scenarios. The fused kernel implements pipelined parallelism between computation and communication.</p>
</td>
</tr>
<tr id="row102221513433"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p273718419525"><a name="p273718419525"></a><a name="p273718419525"></a><a href="torch_npu-npu_mm_reduce_scatter_base.md">torch_npu.npu_mm_reduce_scatter_base</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1385231745515"><a name="p1385231745515"></a><a name="p1385231745515"></a>Fuses matrix multiplication (MatMul) and reduce_scatter collective communication operations in tensor parallelism (TP) scenarios. The fused kernel implements pipelined parallelism between computation and communication.</p>
</td>
</tr>
<tr id="row192221013132"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1474331220582"><a name="p1474331220582"></a><a name="p1474331220582"></a><a href="torch_npu-npu_moe_compute_expert_tokens.md">torch_npu.npu_moe_compute_expert_tokens</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12743131265818"><a name="p12743131265818"></a><a name="p12743131265818"></a>Uses binary search to locate the position of the last row processed by each expert in mixture of experts (MOE) computation.</p>
</td>
</tr>
<tr id="row622320135311"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2743141295810"><a name="p2743141295810"></a><a name="p2743141295810"></a><a href="torch_npu-npu_moe_finalize_routing.md">torch_npu.npu_moe_finalize_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1974341219588"><a name="p1974341219588"></a><a name="p1974341219588"></a>Combines the output results of the MoE feedforward neural network (FFN) at the end of MoE computation.</p>
</td>
</tr>
<tr id="row11308231313"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p57431312165820"><a name="p57431312165820"></a><a name="p57431312165820"></a><a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14743151275819"><a name="p14743151275819"></a><a name="p14743151275819"></a>Performs Softmax operation on the gating output during MoE computation to obtain the top-k result.</p>
</td>
</tr>
<tr id="row133111231730"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p147431112105811"><a name="p147431112105811"></a><a name="p147431112105811"></a><a href="torch_npu-npu_moe_init_routing.md">torch_npu.npu_moe_init_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1174351212586"><a name="p1174351212586"></a><a name="p1174351212586"></a>Performs mixture of experts (MoE) routing based on the computation results of <a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a>.</p>
</td>
</tr>
<tr id="row1531023733"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p7819527225"><a name="p7819527225"></a><a name="p7819527225"></a><a href="torch_npu-npu_prefetch.md">torch_npu.npu_prefetch</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p11819162142215"><a name="p11819162142215"></a><a name="p11819162142215"></a>Provides a network <code>weight</code> prefetching feature to pre-load specified weight data into the L2 Cache before computation begins, reducing memory access wait time when operators access these weights. For example, performing a prefetch before operators such as MatMul allows the operator to read weights directly from the low-latency L2 Cache during execution, which improves operator data access and computation efficiency. The actual performance gain depends on the parallelism strategy and configurations.</p>
</td>
</tr>
<tr id="row13116234313"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p35366207112"><a name="p35366207112"></a><a name="p35366207112"></a><a href="torch_npu-npu_prompt_flash_attention.md">torch_npu.npu_prompt_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p20536122017115"><a name="p20536122017115"></a><a name="p20536122017115"></a>Implements full FlashAttention (FA).</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_quant_lightning_indexer.md">torch_npu.npu_quant_lightning_indexer</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Performs preprocessing computation for SparseFlashAttention (SFA) in inference scenarios. This API selects key sparse tokens and quantizes the input <code>query</code> and <code>key</code> to implement INT8 storage and INT8 computation to maximize performance gains.</p>
</td>
</tr>
<tr id="row9719124019218"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p14326218132919"><a name="p14326218132919"></a><a name="p14326218132919"></a><a href="torch_npu-npu_quant_matmul.md">torch_npu.npu_quant_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001814195101_p156512056161014"><a name="en-us_topic_0000001814195101_p156512056161014"></a><a name="en-us_topic_0000001814195101_p156512056161014"></a>Performs quantized matrix multiplication, supporting at least 2D and at most 6D input.</p>
</td>
</tr>
<tr id="row9719124019219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p104326218132920"><a name="p104326218132920"></a><a name="p104326218132920"></a><a href="torch_npu-npu_quant_matmul_gelu.md">torch_npu.npu_quant_matmul_gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2026022511100607"><a name="p2026022511100607"></a><a name="p2026022511100607"></a>Performs fused computation of quantized matrix multiplication and the GELU activation function. It supports A8W8 and A4W4 quantization.</p>
</td>
</tr>
<tr id="row202508121056216"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p202508121056216"><a name="p202508121056216"></a><a name="p202508121056216"></a><a href="torch_npu-npu_quant_matmul_reduce_sum.md">torch_npu.npu_quant_matmul_reduce_sum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2025081210578767"><a name="p2025081210578767"></a><a name="p2025081210578767"></a>Performs quantized grouped matrix multiplication, sums up the matrix multiplication results of all groups, and outputs the result.</p>
</td>
</tr>
<tr id="row9368201625615"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p432973317361"><a name="p432973317361"></a><a name="p432973317361"></a><a href="torch_npu-npu_quant_scatter.md">torch_npu.npu_quant_scatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p133291433113614"><a name="p133291433113614"></a><a name="p133291433113614"></a>Quantizes <code>updates</code>, and then updates the values in <code>input</code> using the values in <code>updates</code> according to the specified <code>axis</code> and <code>indices</code>. The data in <code>input</code> remains unchanged.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p20338144125814"></a><a name="p20338144125814"></a><a href="torch_npu-npu_recurrent_gated_delta_rule.md">torch_npu.npu_recurrent_gated_delta_rule</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p33382046580"></a><a name="p33382046580"></a>Implements the computation logic of the variable-step Recurrent Gated Delta Rule (RGDR).</p>
</td>
</tr>
<tr id="row159729564415"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p109331430113612"><a name="p109331430113612"></a><a name="p109331430113612"></a><a href="torch_npu-npu_quant_scatter_.md">torch_npu.npu_quant_scatter_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p693315307360"><a name="p693315307360"></a><a name="p693315307360"></a>Quantizes <code>updates</code>, and then updates the values in <code>input</code> using the values in <code>updates</code> based on the specified <code>axis</code> and <code>indices</code>. The data in <code>input</code> is updated in place.</p>
</td>
</tr>
<tr id="row1432501813299"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p20338144125815"><a name="p20338144125815"></a><a name="p20338144125815"></a><a href="torch_npu-npu_quantize.md">torch_npu.npu_quantize</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p33382046581"><a name="p33382046581"></a><a name="p33382046581"></a>Quantizes the input tensor.</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_scatter_nd_update.md">torch_npu.npu_scatter_nd_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001863744477_p15682749165813"><a name="en-us_topic_0000001863744477_p15682749165813"></a><a name="en-us_topic_0000001863744477_p15682749165813"></a>Updates the values in <code>input</code> at the specified indices using the values from <code>updates</code>, and saves the result to the output tensor. The data in <code>input</code> remains unchanged.</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_sparse_flash_attention.md">torch_npu.npu_sparse_flash_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Implements Sparse Flash Attention.</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_sparse_lightning_indexer_grad_kl_loss.md">torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Implements the backward computation of <code>npu_lightning_indexer</code> and integrates the computation of loss.</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_top_k_top_p.md">torch_npu.npu_top_k_top_p</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001863744477_p15682749165813"><a name="en-us_topic_0000001863744477_p15682749165813"></a><a name="en-us_topic_0000001863744477_p15682749165813"></a>Performs <code>top-k</code> and <code>top-p</code> sampling and filtering on the original input <code>logits</code>.</p>
</td>
</tr>
<tr id="row16875125182712"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_top_k_top_p_sample.md">torch_npu.npu_top_k_top_p_sample</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001863744477_p15682749165813"><a name="en-us_topic_0000001863744477_p15682749165813"></a><a name="en-us_topic_0000001863744477_p15682749165813"></a>Performs top-K and top-P sampling computations based on the input logit tensor <code>logits</code>, sampling parameters (<code>top_k</code> and <code>top_p</code>), and random sampling weight distribution <code>q</code>. It outputs the index of the maximum logit value for each batch (<code>logits_select_idx</code>) and the logit distribution after top-K and top-P sampling (<code>logits_top_kp_select</code>).</p>
</td>
</tr>
<tr id="row16875125182713"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_scatter_pa_kv_cache.md">torch_npu.npu_scatter_pa_kv_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001863744477_p15682749165813"><a name="en-us_topic_0000001863744477_p15682749165813"></a><a name="en-us_topic_0000001863744477_p15682749165813"></a>Updates the <code>key</code> and <code>value</code> at the specified positions in the KV cache.</p>
</td>
</tr>
<tr id="row9470191314519"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p83781723143618"><a name="p83781723143618"></a><a name="p83781723143618"></a><a href="torch_npu-npu_scatter_nd_update_.md">torch_npu.npu_scatter_nd_update_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001816704932_p12276121818501"><a name="en-us_topic_0000001816704932_p12276121818501"></a><a name="en-us_topic_0000001816704932_p12276121818501"></a>Updates the values in <code>input</code> at the specified indices using the values from <code>updates</code>, and saves the result to the output tensor. The data in <code>input</code> is modified.</p>
</td>
</tr>
<tr id="row1243753818365"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1332519186295"><a name="p1332519186295"></a><a name="p1332519186295"></a><a href="torch_npu-npu_trans_quant_param.md">torch_npu.npu_trans_quant_param</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p10325181832913"><a name="p10325181832913"></a><a name="p10325181832913"></a>Converts the data type of the quantization parameter <code>scale</code>.</p>
</td>
</tr>
<tr id="row1432853315360"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p34567614299"><a name="p34567614299"></a><a name="p34567614299"></a><a href="torch_npu-npu_weight_quant_batchmatmul.md">torch_npu.npu_weight_quant_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p154556613293"><a name="p154556613293"></a><a name="p154556613293"></a>Performs matrix multiplication with quantization support for the <code>weight</code> input and the output. <code>pertensor</code>, <code>perchannel</code>, and <code>pergroup</code> quantization modes are supported.</p>
</td>
</tr>
<tr id="row10933133063618"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5743101211585"><a name="p5743101211585"></a><a name="p5743101211585"></a><a href="torch_npu-scatter_update.md">torch_npu.scatter_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p874318124582"><a name="p874318124582"></a><a name="p874318124582"></a>Updates the values in the <code>data</code> tensor with the values from the <code>updates</code> tensor according to the specified <code>axis</code> and <code>indices</code>, and saves the results to an output tensor. The data within the original <code>data</code> tensor remains unchanged.</p>
</td>
</tr>
<tr id="row57012280360"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1274381212587"><a name="p1274381212587"></a><a name="p1274381212587"></a><a href="torch_npu-scatter_update_.md">torch_npu.scatter_update_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p3743712125811"><a name="p3743712125811"></a><a name="p3743712125811"></a>Updates values in the <code>data</code> tensor with values from the <code>updates</code> tensor according to the specified <code>axis</code> and <code>indices</code>, and saves the results to an output tensor. The data within the original <code>data</code> tensor is modified in-place.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-empty_with_swapped_memory.md">torch_npu.empty_with_swapped_memory</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Allocates a special tensor with its device type set to NPU, while its actual memory resides on the host side.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-erase_stream.md">torch_npu.erase_stream</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Removes the tensor marker (added by <code>record_stream</code> to the memory pool) after the tensor has been used by the stream.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_gather_sparse_index.md">torch_npu.npu_gather_sparse_index</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Extracts elements from the specified dimension of the input tensor based on the indices in <code>index</code> and saves them to the output tensor.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine.md">torch_npu.npu_moe_distribute_combine</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses <code>reduce_scatterv</code> communication, <code>alltoallv</code> communication, and final data aggregation by multiplying the corresponding weights and summing the results.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_dispatch.md">torch_npu.npu_moe_distribute_dispatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs quantization on token data (optional), followed by <code>alltoallv</code> communication in the Expert Parallelism (EP) domain, and then <code>allgatherv</code> communication in the Tensor Parallelism (TP) domain (optional).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine_v2.md">torch_npu.npu_moe_distribute_combine_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses <code>reduce_scatterv</code> communication, <code>alltoallv</code> communication, and final data aggregation by multiplying the corresponding weights and summing the results.</p>
</td>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_dispatch_v2.md">torch_npu.npu_moe_distribute_dispatch_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs quantization on token data (optional), followed by <code>alltoallv</code> communication in the Expert Parallelism (EP) domain, and then <code>allgatherv</code> communication in the Tensor Parallelism (TP) domain (optional).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_gating_top_k.md">torch_npu.npu_moe_gating_top_k</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs <code>sigmoid</code> computation on the input <code>x</code> during MoE computation, groups and sorts the computation results, and selects the top-k experts based on the group sorting results.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_init_routing_v2.md">torch_npu.npu_moe_init_routing_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs mixture of experts (MoE) routing based on the computation results of <a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a>. Non-quantization, dynamic quantization, and static quantization configurations are supported.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_swiglu_quant.md">torch_npu.npu_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Adds a quantization operation after the SwiGLU activation function to perform <code>SwiGluQuant</code> computation on the input <code>x</code>. This API supports <code>int8</code> or <code>int4</code> quantized outputs, MoE and non-MoE scenarios (when <code>group_index</code> is omitted), group quantization, and dynamic or static quantization.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_dequant_swiglu_quant.md">torch_npu.npu_dequant_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses dequantization, SwiGLU activation, and quantization operations on the tensor <code>x</code>, with support for grouped computation.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_kv_rmsnorm_rope_cache.md">torch_npu.npu_kv_rmsnorm_rope_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses Root Mean Square Normalization (RMSNorm), Rotary Position Embedding (RoPE), and KV cache update operations (ScatterUpdate) within the Multi-head Latent Attention (MLA) structure.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_interleave_rope.md">torch_npu.npu_interleave_rope</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Applies rotary positional encoding (RoPE) to a single input tensor <code>x</code>.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_mrope.md">torch_npu.npu_mrope</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Applies precomputed <code>sin</code> and <code>cos</code> positional encoding caches to <code>query</code> and <code>key</code> in inference scenarios to implement rotary positional embeddings (RoPE), and supports multi-modal MRoPE.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_re_routing.md">torch_npu.npu_moe_re_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Rearranges tokens by expert order in the Mixture of Experts (MoE) network after AlltoAll communication across ranks.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-matmul_checksum.md">torch_npu.matmul_checksum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Detects AI Core hardware faults based on native <code>torch.matmul</code> and <code>Tensor.matmul</code> APIs. This API internally verifies matrix computation results. It compares the verification error against a real-time computed verification threshold. If the verification error exceeds the threshold, this API raises an AI Core error.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_alltoallv_gmm.md">torch_npu.npu_alltoallv_gmm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses <code>AlltoAllv</code>, <code>Permute</code>, and <code>GroupedMatMul</code> for routed experts in an MoE network, implements parallel fused computation with the shared expert <code>MatMul</code>, and uses a communication-before-computation sequence.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_gmm_alltoallv.md">torch_npu.npu_gmm_alltoallv</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses <code>AlltoAllv</code>, <code>Permute</code>, and <code>GroupedMatMul</code> for routed experts in an MoE network, implements parallel fused computation with the shared expert <code>MatMul</code>, and uses a communication-before-computation sequence.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine_add_rms_norm.md">torch_npu.npu_moe_distribute_combine_add_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses <code>moe_distribute_combine</code>, <code>add</code>, and <code>rms_norm</code> operations. This API must be used together with <code>torch_npu.npu_moe_distribute_dispatch</code>. It returns data along the original data collection path of the <code>npu_moe_distribute_dispatch</code> operator and performs an <code>add_rms_norm</code> operation.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_transpose_batchmatmul.md">torch_npu.npu_transpose_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs batch matrix multiplication between the <code>input</code> and <code>weight</code> tensors.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_update_expert.md">torch_npu.npu_moe_update_expert</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Implements the Expert Parallelism Load Balancer (EPLB) algorithm commonly used in MoE networks for redundant expert deployment to address load imbalance issues. The <code>MoeUpdateExpert</code> operator maps the logical expert IDs of the top-K experts for each token to physical expert instance IDs. It also supports pruning the top-K experts assigned to tokens based on a threshold.</p>
</td>
</tr>
<tr id="row285193313382"><td class="cellrowborder" valign="top" width="38.22%" headers="mcps1.2.3.1.1 "><p id="p2851433193819"><a name="p2851433193819"></a><a name="p2851433193819"></a><a href="torch_npu-npu_dynamic_block_quant.md">torch_npu.npu_dynamic_block_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.78%" headers="mcps1.2.3.1.2 "><p id="p785143323817"><a name="p785143323817"></a><a name="p785143323817"></a>Divides the input tensor into multiple data blocks based on the specified <code>row_block_size</code> and <code>col_block_size</code>, and performs quantization at the block level.</p>
</td>
</tr>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-set_device_limit.md">torch_npu.set_device_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Sets the number of Cube and Vector cores used by the current process for operator execution on the specified device.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-get_device_limit.md">torch_npu.get_device_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Obtains the number of Cube and Vector cores used for operator execution on a specified device.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-set_stream_limit.md">torch_npu.set_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Sets the number of Cube and Vector cores used for operator execution on a specified stream.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-reset_stream_limit.md">torch_npu.reset_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Restores the number of Cube and Vector cores used for operator execution on a specified stream to the default configuration.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-get_stream_limit.md">torch_npu.get_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Obtains the number of Cube and Vector cores used for operator execution on a specified stream.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_sim_exponential_.md">torch_npu.npu_sim_exponential_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs in-place sampling from an exponential distribution for elements in the input tensor, modifying the input tensor in-place.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_dense_lightning_indexer_softmax_lse.md">torch_npu.npu_dense_lightning_indexer_softmax_lse</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>Optimizes device memory usage as a frontend API of <code>npu_dense_lightning_indexer_grad_kl_loss</code> by precomputing the maximum and sum values used in the Softmax computation of the Lightning Indexer component.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_dense_lightning_indexer_grad_kl_loss.md">torch_npu.npu_dense_lightning_indexer_grad_kl_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>Implements the backward gradient computation during the warmup phase training of the Lightning Indexer component, while integrating the loss computation.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm.md">torch_npu.npu_add_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses Add computation with RMSNorm normalization, commonly used in foundation models to normalize tensors after residual connections.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_dynamic_quant.md">torch_npu.npu_add_rms_norm_dynamic_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>The RMSNorm operator is a normalization operation commonly used in foundation models. Compared with the LayerNorm operator, it removes the mean subtraction step. The DynamicQuant operator performs symmetric dynamic quantization on the input tensor. The AddRmsNormDynamicQuant operator fuses the Add operator before RMSNorm and 1 or 2 DynamicQuant operators applied to the RMSNorm normalization output, reducing data transfer operations.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-save_npugraph_tensor.md">torch_npu.save_npugraph_tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>Provides a tensor dumping capability similar to the native print feature without affecting <code>aclgraph</code> replay. This API allows the tensor data, data types, and shape information of intermediate nodes within an <code>aclgraph</code> to be saved to a specified <code>.pt</code> or <code>.bin</code> file, enabling users to inspect tensor data during <code>aclgraph</code> execution and quickly locate issues.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_fused_floyd_attention.md">torch_npu.npu_fused_floyd_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>In training scenarios, <code>npu_fused_floyd_attention</code> differs from traditional FlashAttention (<code>npu_fusion_attention</code>) by treating the sequence dimension (<code>seq</code>) as an additional batch axis during QK/PV attention computation, thereby converting the attention computation into batch matrix multiplication (<code>batchMatmul</code>).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_clipped_swiglu.md">torch_npu.npu_clipped_swiglu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Implements a variant SwiGLU activation function with a truncated Swish gating linear unit.</p>
</td>
</tr>
</tbody>
</table>
