# torch_npu APIs

This section describes commonly used custom APIs, including Tensor creation and computation operations.

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
<tr id="row589165316115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1489155321111"><a name="p1489155321111"></a><a name="p1489155321111"></a><a href="(beta)torch_npu-npu_ciou.md">(beta)torch_npu.npu_ciou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p749621223015"><a name="en-us_topic_0000001655404257_p749621223015"></a><a name="en-us_topic_0000001655404257_p749621223015"></a>Computes the Complete Intersection over Union (CIoU) loss between the predicted bounding box and the ground-truth bounding box using an NPU-based CIoU operation.</p>
</td>
</tr>
<tr id="row138914536113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1289453191113"><a name="p1289453191113"></a><a name="p1289453191113"></a><a href="(beta)torch_npu-npu_clear_float_status.md">(beta)torch_npu.npu_clear_float_status</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p289105341110"><a name="p289105341110"></a><a name="p289105341110"></a>Clears the status flags related to overflow detection.</p>
</td>
</tr>
<tr id="row158945317117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p5901953141111"><a name="p5901953141111"></a><a name="p5901953141111"></a><a href="(beta)torch_npu-npu_confusion_transpose.md">(beta)torch_npu.npu_confusion_transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p7908532119"><a name="p7908532119"></a><a name="p7908532119"></a>Fuses the reshape and transpose operations.</p>
</td>
</tr>
<tr id="row109045310113"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p49035381110"><a name="p49035381110"></a><a name="p49035381110"></a><a href="(beta)torch_npu-npu_conv3d.md">(beta)torch_npu.npu_conv3d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p18901253111116"><a name="p18901253111116"></a><a name="p18901253111116"></a>Applies a 3D convolution to an input image composed of multiple input planes.</p>
</td>
</tr>
<tr id="row75771250101115"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p15771150151111"><a name="p15771150151111"></a><a name="p15771150151111"></a><a href="(beta)torch_npu-npu_deformable_conv2d.md">(beta)torch_npu.npu_deformable_conv2d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p1888192943717"><a name="en-us_topic_0000001655404257_p1888192943717"></a><a name="en-us_topic_0000001655404257_p1888192943717"></a>Computes the deformed convolution output using the expected input.</p>
</td>
</tr>
<tr id="row125771550121117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16577135019112"><a name="p16577135019112"></a><a name="p16577135019112"></a><a href="(beta)torch_npu-npu_diou.md">(beta)torch_npu.npu_diou</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001655404257_p12796833183010"><a name="en-us_topic_0000001655404257_p12796833183010"></a><a name="en-us_topic_0000001655404257_p12796833183010"></a>This API implements NPU-based DIoU (Distance Intersection over Union) computation. The algorithm comprehensively considers the distance between the centers of predicted and ground-truth bounding boxes as well as the overlap ratio of bounding boxes, making bounding box regression more stable and improving object localization accuracy.</p>
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
<tr id="row11579104721117"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p135794473113"><a name="p135794473113"></a><a name="p135794473113"></a><a href="(beta)torch_npu-npu_linear.md">(beta)torch_npu.npu_linear</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p8579144781110"><a name="p8579144781110"></a><a name="p8579144781110"></a>Multiplies matrix <code>a</code> by matrix <code>b</code> to produce matrix <code>a * b</code>.</p>
</td>
</tr>
<tr id="row9579184715119"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p85793478116"><a name="p85793478116"></a><a name="p85793478116"></a><a href="(beta)torch_npu-npu_lstm.md">(beta)torch_npu.npu_lstm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p757912479111"><a name="p757912479111"></a><a name="p757912479111"></a>Computes the forward pass of a Long Short-Term Memory (LSTM) network.</p>
</td>
</tr>
<tr id="row3579194701112"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p16579184717113"><a name="p16579184717113"></a><a name="p16579184717113"></a><a href="(beta)torch_npu-npu_max.md">(beta)torch_npu.npu_max</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1758054719115"><a name="p1758054719115"></a><a name="p1758054719115"></a>Computes the maximum values along <code>dim</code>. This API is similar to <code>torch.max</code> and is optimized for NPUs.</p>
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
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p484114213112"><a name="p484114213112"></a><a name="p484114213112"></a>Packs float-type 1-bit Adam parameters into <code>torch.uint8</code>.</p>
</td>
</tr>
<tr id="row128412042131111"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p784116424113"><a name="p784116424113"></a><a name="p784116424113"></a><a href="(beta)torch_npu-npu_sign_bits_unpack.md">(beta)torch_npu.npu_sign_bits_unpack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p58411242191112"><a name="p58411242191112"></a><a name="p58411242191112"></a>Unpacks <code>torch.uint8</code>-type 1-bit Adam parameters into float.</p>
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
<tr id="row3892520439"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2991782322389"><a name="p2991782322389"></a><a name="p2991782322389"></a><a href="torch_npu-npu_rms_norm_quant_v2.md">torch_npu.npu_rms_norm_quant_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_rotate_quant_desc"><a name="p_rotate_quant_desc"></a><a name="p_rotate_quant_desc"></a><span>Fuses rotation (Rotate) and quantization (Quant) for model performance optimization.</span></p>
</td>
</tr>
<tr id="row3892520439"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2991782322389"><a name="p2991782322389"></a><a name="p2991782322389"></a><a href="torch_npu-npu_rms_norm_dynamic_mx_quant.md">torch_npu.npu_rms_norm_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_save_npugraph_tensor_desc"><a name="p_save_npugraph_tensor_desc"></a><a name="p_save_npugraph_tensor_desc"></a><span>Provides functionality similar to the native PyTorch <code>torch.save</code> API without affecting tensor dump capabilities during aclgraph capture and replay. It allows tensor data, data types, and shape information of intermediate nodes in aclgraph to be saved to a specified binary file, enabling users to inspect tensor data during aclgraph execution and quickly locate issues.</span></p>
</td>
</tr>
<tr id="row17801219526"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p89062124277"><a name="p89062124277"></a><a name="p89062124277"></a><a href="torch_npu-npu_rotary_mul.md">torch_npu.npu_rotary_mul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p69061812102715"><a name="p69061812102715"></a><a name="p69061812102715"></a>Implements rotary position embedding (RoPE).</p>
</td>
</tr>
<tr id="row58015194219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1590618121275"><a name="p1590618121275"></a><a name="p1590618121275"></a><a href="torch_npu-npu_scaled_masked_softmax.md">torch_npu.npu_scaled_masked_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1290691218270"><a name="p1290691218270"></a><a name="p1290691218270"></a>Computes the Softmax result after scaling the input tensor <code>x</code> and applying the <code>mask</code>.</p>
</td>
</tr>
<tr id="row480181910211"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p137651452352"><a name="p137651452352"></a><a name="p137651452352"></a><a href="(beta)torch_npu-npu_swiglu.md">(beta)torch_npu.npu_swiglu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14325153814812"><a name="p14325153814812"></a><a name="p14325153814812"></a>Provides the SwiGLU activation function.</p>
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
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_all_gather_quant_mm.md">torch_npu.npu_all_gather_quant_mm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In the quantized TP scenario, this API fuses allgather and matmul and outputs the dequantization result.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_all_to_all_matmul.md">torch_npu.npu_all_to_all_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses the AlltoAll communication, Permute (to keep addresses contiguous after communication), and Matmul computation. Communication is performed before computation.</p>
</td>
</tr>
<tr id="row950000000021"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p950000000022"><a name="p950000000022"></a><a name="p950000000022"></a><a href="torch_npu-npu_all_to_all_quant_matmul.md">torch_npu.npu_all_to_all_quant_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p950000000023"><a name="p950000000023"></a><a name="p950000000023"></a>Fuses the quantized Matmul computation, Permute (to keep addresses contiguous after communication), and AlltoAll communication. Communication is performed before computation. The K-C dynamic quantization mode and the mx quantization mode are supported.</p>
</td>
</tr>
<tr id="npu_anti_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_anti_mx_quant"><a name="p_npu_anti_mx_quant"></a><a name="p_npu_anti_mx_quant"></a><a href="torch_npu-npu_anti_mx_quant.md">torch_npu.npu_anti_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_anti_mx_quant_desc"><a name="p_npu_anti_mx_quant_desc"></a><a name="p_npu_anti_mx_quant_desc"></a><span>The torch.float16, torch.bfloat16, and torch.float32 formats. It is the inverse process of npu_dynamic_mx_quant.<code>torch.float16</code>/<code>torch.bfloat16</code>/<code>torch.float32</code></span></p></td>
</tr>
<tr id="row952719519471"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18248651143510"><a name="p18248651143510"></a><a name="p18248651143510"></a><a href="torch_npu-npu_anti_quant.md">torch_npu.npu_anti_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001850161621_p0996174814315"><a name="en-us_topic_0000001850161621_p0996174814315"></a><a name="en-us_topic_0000001850161621_p0996174814315"></a>Performs a dequantization operation on the tensor x, that is, restores the quantized low-precision data to floating-point numbers. Dequantization is the reverse process of quantization and is used to convert low-precision data (such as <code>torch.int8</code>, <code>torch_npu.int4</code>, <code>torch_npu.hifloat8</code>, <code>torch.float8_e4m3fn</code>, and <code>torch.float8_e5m2</code>) back to high-precision floating-point numbers.</p>
</td>
</tr>
<tr id="row952719519471"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p18248651143510"><a name="p18248651143510"></a><a name="p18248651143510"></a><a href="torch_npu-npu_attention_to_ffn.md">torch_npu.npu_attention_to_ffn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001850161621_p0996174814315"><a name="en-us_topic_0000001850161621_p0996174814315"></a><a name="en-us_topic_0000001850161621_p0996174814315"></a>Sends token data from the Attention node to the FFN node.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_attention_update.md">torch_npu.npu_attention_update</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Updates the two local variable results lse and local_out output by the PA operator of each SP domain to global results.</p>
</td>
</tr>
<tr id="row_block_sparse_attention"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10326218132921"><a name="p10326218132921"></a><a name="p10326218132921"></a><a href="torch_npu-npu_block_sparse_attention.md">torch_npu.npu_block_sparse_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p10326218132922"><a name="p10326218132922"></a><a name="p10326218132922"></a>Computes <code>BlockSparseAttention</code>. This sparse attention mechanism supports block-level sparsity. It uses <code>block_sparse_mask</code> to specify the KV blocks selected by each Q block to achieve efficient attention computation.</p>
</td>
</tr>
<tr id="row_chunk_gated_delta_rule"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10326218132923"><a name="p10326218132923"></a><a name="p10326218132923"></a><a href="torch_npu-npu_chunk_gated_delta_rule.md">torch_npu.npu_chunk_gated_delta_rule</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p10326218132924"><a name="p10326218132924"></a><a name="p10326218132924"></a>Chunked Gated Delta Rule (CGDR) is a chunked implementation of Gated Delta Rule (GDR). It divides the input sequence into chunks to achieve a certain degree of parallelism. In long-context scenarios, it offers higher computational efficiency than Recurrent Gated Delta Rule and is suitable for the prefill stage.</p>
</td>
</tr>
<tr id="row16527145114477"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1338142581"><a name="p1338142581"></a><a name="p1338142581"></a><a href="torch_npu-npu_convert_weight_to_int4pack.md">torch_npu.npu_convert_weight_to_int4pack</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p16338242582"><a name="p16338242582"></a><a name="p16338242582"></a>Packs an <code>torch.int32</code> input tensor into the <code>torch_npu.int4</code> data type. Every eight <code>torch_npu.int4</code> elements are carried by a single <code>torch.int32</code> element and stored in an interleaved format.</p>
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
<tr id="npu_dynamic_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_mx_quant"><a name="p_npu_dynamic_mx_quant"></a><a name="p_npu_dynamic_mx_quant"></a><a href="torch_npu-npu_dynamic_mx_quant.md">torch_npu.npu_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_mx_quant_desc"><a name="p_npu_dynamic_mx_quant_desc"></a><a name="p_npu_dynamic_mx_quant_desc"></a><span>Performs dynamic MX quantization of the input tensor with a target data type of float4 or float8. On the given axis, the corresponding quantization scale is computed for every block_size elements, and the data in the group is divided by this scale and then converted to the target data type.</span></p></td>
</tr>
<tr id="npu_dynamic_block_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_block_mx_quant"><a name="p_npu_dynamic_block_mx_quant"></a><a name="p_npu_dynamic_block_mx_quant"></a><a href="torch_npu-npu_dynamic_block_mx_quant.md">torch_npu.npu_dynamic_block_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_block_mx_quant_desc"><a name="p_npu_dynamic_block_mx_quant_desc"></a><a name="p_npu_dynamic_block_mx_quant_desc"></a><span>Performs MX quantization on the input variable with data blocks (32*32) as the basic blocks and converts it to the target data type.</span></p></td>
</tr>
<tr id="npu_gelu_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_gelu_quant"><a name="p_npu_gelu_quant"></a><a name="p_npu_gelu_quant"></a><a href="torch_npu-npu_gelu_quant.md">torch_npu.npu_gelu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_gelu_quant_desc"><a name="p_npu_gelu_quant_desc"></a><a name="p_npu_gelu_quant_desc"></a><span>Performs the GELU (Gaussian Error Linear Unit) activation operation on the tensor and then performs static or dynamic quantization on the result.</span></p></td>
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
</td>
</tr>
<tr id="row545919116481"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p099812152913"><a name="p099812152913"></a><a name="p099812152913"></a><a href="torch_npu-npu_ffn_to_attention.md">torch_npu.npu_ffn_to_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1998152112915"><a name="p1998152112915"></a><a name="p1998152112915"></a>Sends token data from the FFN node to the Attention node.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p43681016105618"></a><a name="p43681016105618"></a><a href="torch_npu-npu_fused_infer_attention_score_v2.md">torch_npu.npu_fused_infer_attention_score_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p836911165560"><a name="p836911165560"></a><a name="p836911165560"></a>Adapts to the <code>FlashAttention</code> operator in the incremental and full inference scenarios, supporting both full computation (<code>PromptFlashAttention</code>) and incremental computation (<code>IncreFlashAttention</code>). Added support for Multi-Head Latent Attention (MLA) full quantization.</p>
</td>
</tr>
<tr id="row_npu_fused_matmul"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_fused_matmul.md">torch_npu.npu_fused_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses matrix multiplication and general vector computation to reduce data movement and improve performance.</p>
</td>
</tr>
<tr id="row14601311184810"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p1799710217294"><a name="p1799710217294"></a><a name="p1799710217294"></a><a href="torch_npu-npu_fusion_attention.md">torch_npu.npu_fusion_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p129971221112919"><a name="p129971221112919"></a><a name="p129971221112919"></a>Implements fused computation of the Transformer attention score.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_fusion_attention_v2.md">torch_npu.npu_fusion_attention_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Implements the fused computation of the Transformer Attention Score, supporting RoPE extension, bias sink, and GQA.</p>
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
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p5338546582"><a name="p5338546582"></a><a name="p5338546582"></a>Applies GroupNorm followed by the SiLU activation function to the input tensor <code>input</code> and returns three tensors: <code>out</code> (the output after SiLU activation), <code>meanOut</code> (the normalization mean), and <code>rstdOut</code> (the reciprocal of the normalization standard deviation).</p>
</td>
</tr>
<tr id="row174571042727"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p10203752943"><a name="p10203752943"></a><a name="p10203752943"></a><a href="torch_npu-npu_group_quant.md">torch_npu.npu_group_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p4382954217"><a name="p4382954217"></a><a name="p4382954217"></a>Performs group-wise quantization on the input tensor.</p>
</td>
</tr>
<tr id="npu_grouped_dynamic_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_grouped_dynamic_mx_quant"><a name="p_npu_grouped_dynamic_mx_quant"></a><a name="p_npu_grouped_dynamic_mx_quant"></a><a href="torch_npu-npu_grouped_dynamic_mx_quant.md">torch_npu.npu_grouped_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_grouped_dynamic_mx_quant_desc"><a name="p_npu_grouped_dynamic_mx_quant_desc"></a><a name="p_npu_grouped_dynamic_mx_quant_desc"></a><span>Performs dynamic MX quantization with the target data type float8/float4 on each group at the granularity of basic blocks (blocksize) according to the start value of the passed group index (group_index), and outputs the quantization scale mxscale (<code>torch.float8_e8m0fnu</code>).</span></p>
</td>
</tr>
<tr id="row1045718422215"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p64370382369"><a name="p64370382369"></a><a name="p64370382369"></a><a href="torch_npu-npu_grouped_matmul_add.md">torch_npu.npu_grouped_matmul_add</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1437638113619"><a name="p1437638113619"></a><a name="p1437638113619"></a><span>Provides an efficient method to perform grouped computation of multiple matrix multiplication (MatMul) operations.</span></p>
</td>
</tr>
<tr id="row_npu_grouped_matmul_add"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_grouped_matmul_add_.md">torch_npu.npu_grouped_matmul_add_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In micro-batch training scenarios, micro-batch gradient accumulation involves a large number of scenarios in which GroupedMatmul operations are fused with InplaceAdd operations. This operator (GroupedMatmulAdd) fuses the preceding operators in non-quantization scenarios to improve network performance.</p>
</td>
</tr>
<tr id="row1045718422215"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p64370382369"><a name="p64370382369"></a><a name="p64370382369"></a><a href="torch_npu-npu_grouped_matmul.md">torch_npu.npu_grouped_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1437638113619"><a name="p1437638113619"></a><a name="p1437638113619"></a><span>npu_grouped_matmul is an efficient method for performing grouped computation of multiple matrix multiplication (matmul) operations.</span></p>
</td>
</tr>
<tr id="npu_grouped_matmul_finalize_routing"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_grouped_matmul_finalize_routing"><a name="p_npu_grouped_matmul_finalize_routing"></a><a name="p_npu_grouped_matmul_finalize_routing"></a><a href="torch_npu-npu_grouped_matmul_finalize_routing.md">torch_npu.npu_grouped_matmul_finalize_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_grouped_matmul_finalize_routing_desc"><a name="p_npu_grouped_matmul_finalize_routing_desc"></a><a name="p_npu_grouped_matmul_finalize_routing_desc"></a><span>Fusion operator of GroupedMatMul and MoeFinalizeRouting.</span></p>
</td>
</tr>
<tr id="row1045718422215"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p64370382369"><a name="p64370382369"></a><a name="p64370382369"></a><a href="torch_npu-npu_grouped_matmul_swiglu_quant.md">torch_npu.npu_grouped_matmul_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1437638113619"><a name="p1437638113619"></a><a name="p1437638113619"></a><span>npu_grouped_matmul_swiglu_quant is a computation method that fuses grouped matrix multiplication (GroupedMatmul), dequantization (dequant), the SwiGLu activation function, and quantization (quant), and is used for model performance optimization.</span></p>
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
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Implements fake-quantization for Sparse Flash Attention.</p>
</td>
</tr>
<tr id="row_npu_kronecker_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_kronecker_quant.md">torch_npu.npu_kronecker_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs two matrix multiplications on the input tensor in sequence and quantizes the matrix multiplication result to the <code>torch_npu.int4</code> or <code>torch_npu.float4_e2m1fn_x2</code> type.</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_lightning_indexer.md">torch_npu.npu_lightning_indexer</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Obtains the Top-$k$ positions for each token (full quantization).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_matmul_all_to_all.md">torch_npu.npu_matmul_all_to_all</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>Fuses the Matmul computation, Permute (to keep addresses contiguous after communication), and AlltoAll communication. Computation is performed before communication.</p>
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
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p55830512496"><a name="p55830512496"></a><a name="p55830512496"></a><span>Performs computation during Multi-Head Latent Attention (MLA) preprocessing in inference scenarios. Compared with v2, v3 adds the `query_norm` and `dequant_scale_q_norm` outputs, the pertoken-pergroup quantization mode, a scale correction factor, support for multiple `cache_mode` formats, and several quantization configuration parameters, and makes `cache_index` optional.</span></p>
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
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p12743131265818"><a name="p12743131265818"></a><a name="p12743131265818"></a>Uses binary search to locate the position of the last row processed by each expert in Mixture-of-Experts (MOE) computation.</p>
</td>
</tr>
<tr id="row622320135311"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p2743141295810"><a name="p2743141295810"></a><a name="p2743141295810"></a><a href="torch_npu-npu_moe_finalize_routing.md">torch_npu.npu_moe_finalize_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1974341219588"><a name="p1974341219588"></a><a name="p1974341219588"></a>Combines the output results of the MoE feedforward neural network (FFN) at the end of MoE computation.</p>
</td>
</tr>
<tr id="row11308231313"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p57431312165820"><a name="p57431312165820"></a><a name="p57431312165820"></a><a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p14743151275819"><a name="p14743151275819"></a><a name="p14743151275819"></a>Uses the gating mechanism of a Mixture-of-Experts (MoE) architecture for expert routing, and applies Softmax to the input <code>x</code>, followed by TopK to select the top-K experts with the highest weights.</p>
</td>
</tr>
<tr id="row133111231730"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p147431112105811"><a name="p147431112105811"></a><a name="p147431112105811"></a><a href="torch_npu-npu_moe_init_routing.md">torch_npu.npu_moe_init_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1174351212586"><a name="p1174351212586"></a><a name="p1174351212586"></a>Performs Mixture-of-Experts (MoE) routing based on the computation results of <a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a>.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_permute.md">torch_npu.npu_moe_token_permute</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In the MoE computation, expands the input tokens according to the index indices, sorts them by expert index, and returns the sorted tokens and their mapping to the original tokens.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_unpermute.md">torch_npu.npu_moe_token_unpermute</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In the MoE computation, the inverse operation of torch_npu.npu_moe_token_permute. It maps the expanded and sorted token features back to the original token positions according to sorted_indices and accumulates them, supporting probs weighting.</p>
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
<tr id="row950000000031"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p950000000032"><a name="p950000000032"></a><a name="p950000000032"></a><a href="torch_npu-npu_quant_gmm_alltoallv.md">torch_npu.npu_quant_gmm_alltoallv</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p950000000033"><a name="p950000000033"></a><a name="p950000000033"></a>Fuses the routed-expert GroupedMatmul and AlltoAllv. Computation is performed before communication, and it is also fused in parallel with the shared-expert MatMul computation. The T-T quantization mode and the mx quantization mode are supported.</p>
</td>
</tr>
<tr id="row1545717422219"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_quant_lightning_indexer.md">torch_npu.npu_quant_lightning_indexer</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>QuantLightningIndexer is a preprocessing computation module for SparseFlashAttention (SFA) in inference scenarios. It selects the critical sparse tokens and quantizes the input query and key to implement 8-bit storage and 8-bit computation, maximizing the benefit.</p>
</td>
</tr>
<tr id="row9719124019218"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p14326218132919"><a name="p14326218132919"></a><a name="p14326218132919"></a><a href="torch_npu-npu_quant_matmul.md">torch_npu.npu_quant_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001814195101_p156512056161014"><a name="en-us_topic_0000001814195101_p156512056161014"></a><a name="en-us_topic_0000001814195101_p156512056161014"></a>Performs quantized matrix multiplication, supporting at least 2D and at most 6D input.</p>
</td>
</tr>
<tr id="row950000000041"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p950000000042"><a name="p950000000042"></a><a name="p950000000042"></a><a href="torch_npu-npu_quant_matmul_all_to_all.md">torch_npu.npu_quant_matmul_all_to_all</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p950000000043"><a name="p950000000043"></a><a name="p950000000043"></a>Fuses the quantized Matmul computation, Permute (to keep addresses contiguous after communication), and AlltoAll communication. Communication is performed before computation. The K-C quantization mode and the mx quantization mode are supported.</p>
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
<tr id="row202508121056216"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p202508121056216"><a name="p202508121056216"></a><a name="p202508121056216"></a><a href="torch_npu-npu_quant_mm_reduce_scatter.md">torch_npu.npu_quant_mm_reduce_scatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p2025081210578767"><a name="p2025081210578767"></a><a name="p2025081210578767"></a>Fuses MatMul and the ReduceScatter collective communication computation. The matrix multiplication result is reduced and scattered by the number of devices, and low-precision quantization and dequantization are supported.</p>
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
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p1281815913362"><a name="p1281815913362"></a><a name="p1281815913362"></a>Implements the backward computation of <code>npu_lightning_indexer</code> and integrates the computation of loss.</p>
</td>
</tr>
<tr id="row16875125182711"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p58181296368"><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_sparse_lightning_indexer_grad_kl_loss.md">torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>Optimizes device memory usage as a frontend API of <code>npu_dense_lightning_indexer_grad_kl_loss</code> by precomputing the maximum and sum values used in the Softmax computation of the Lightning Indexer component.</p>
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
<tr id="row16875125182713"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_scatter_pa_cache.md">torch_npu.npu_scatter_pa_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001863744477_p15682749165813"><a name="en-us_topic_0000001863744477_p15682749165813"></a><a name="en-us_topic_0000001863744477_p15682749165813"></a>Updates the <code>key</code> and <code>value</code> at the specified positions in the KV cache.</p>
</td>
</tr>
<tr id="row16875125182713"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p77002833617"><a name="p77002833617"></a><a name="p77002833617"></a><a href="torch_npu-npu_scatter_pa_kv_cache.md">torch_npu.npu_scatter_pa_kv_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001863744477_p15682749165813"><a name="en-us_topic_0000001863744477_p15682749165813"></a><a name="en-us_topic_0000001863744477_p15682749165813"></a>Updates the <code>key</code> and <code>value</code> at the specified positions in the KVCache.<code>key</code><code>value</code>.</p>
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
<tr id="row250825154001"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p250825154002"><a name="p250825154002"></a><a name="p250825154002"></a><a href="torch_npu-npu_weight_quant_preprocess.md">torch_npu.npu_weight_quant_preprocess</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p250825154003"><a name="p250825154003"></a><a name="p250825154003"></a>This API preprocesses the pseudo-quantization parameters of Matmul operators.</p>
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
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Requests a special tensor whose device information is NPU but whose actual memory is on the host side.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-erase_stream.md">torch_npu.erase_stream</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>After a tensor is marked on the memory pool through record_stream, the mark has been used by the stream and can be removed through this API.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_gather_sparse_index.md">torch_npu.npu_gather_sparse_index</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Extracts elements from the specified dimension of the input tensor according to the indices in <code>index</code> and saves them to the output tensor.<code>index</code></p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine.md">torch_npu.npu_moe_distribute_combine</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs reduce_scatterv communication first, then alltoallv communication, and finally integrates the received data (multiplies by the weight and adds).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_dispatch.md">torch_npu.npu_moe_distribute_dispatch</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Quantizes the Token data first (optional), then performs alltoallv communication in the EP (Expert Parallelism) domain, and then performs allgatherv communication in the TP (Tensor Parallelism) domain (optional).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine_v2.md">torch_npu.npu_moe_distribute_combine_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs reduce_scatterv communication first, then alltoallv communication, and finally integrates the received data (multiplies by the weight and adds). The v2 version adds support for special expert scenarios (zero experts, copy experts, and constant experts) and works with assist_info_for_combine of dispatch_v2 to implement more efficient all-device synchronization.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_dispatch_v2.md">torch_npu.npu_moe_distribute_dispatch_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Quantizes the Token data first (optional), then performs alltoallv communication in the EP (Expert Parallelism) domain, and then performs allgatherv communication in the TP (Tensor Parallelism) domain (optional). The v2 version replaces the return value expand_idx with assist_info_for_combine to include more detailed token information, and adds the comm_alg parameter to replace the HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE environment variables.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_gating_top_k.md">torch_npu.npu_moe_gating_top_k</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In the MoE computation, performs the Sigmoid computation on the input x, sorts the computation results by group, and finally selects the top k experts according to the group sorting result.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_init_routing_v2.md">torch_npu.npu_moe_init_routing_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs Mixture-of-Experts (MoE) routing based on the computation results of <a href="torch_npu-npu_moe_gating_top_k_softmax.md">torch_npu.npu_moe_gating_top_k_softmax</a>. Non-quantization, dynamic quantization, and static quantization configurations are supported.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_swiglu_quant.md">torch_npu.npu_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Adds a quantization operation after the SwiGLU activation function to perform <code>SwiGluQuant</code> computation on the input <code>x</code>. This API supports <code>torch.int8</code> or <code>torch_npu.int4</code> quantized outputs, MoE and non-MoE scenarios (when <code>group_index</code> is omitted), group quantization, and dynamic or static quantization.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_dequant_swiglu_quant.md">torch_npu.npu_dequant_swiglu_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs the dequant dequantization + swiglu + quant quantization operations on the tensor x, and also supports grouping.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_swiglu_mx_quant.md">torch_npu.npu_swiglu_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Adds the DynamicMxQuant operation after the Swiglu activation function to implement the SwigluMxQuant computation of the input x. The MoE scenario and the non-MoE scenario (empty group_index) are supported, and multiple swiglu computation modes are supported.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_kv_rmsnorm_rope_cache.md">torch_npu.npu_kv_rmsnorm_rope_cache</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses the RMSNorm normalization computation, RoPE (Rotary Position Embedding) position encoding, and the ScatterUpdate operation for updating the KVCache in the MLA (Multi-head Latent Attention) structure.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_kv_rmsnorm_rope_cache_v2.md">torch_npu.npu_kv_rmsnorm_rope_cache_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses the RMSNorm normalization computation, RoPE (Rotary Position Embedding) position encoding, and the ScatterUpdate operation for updating the KVCache in the MLA (Multi-head Latent Attention) structure.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_interleave_rope.md">torch_npu.npu_interleave_rope</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs rotary position encoding on a single input x.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_mrope.md">torch_npu.npu_mrope</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In inference scenarios, applies the precomputed sin and cos position encoding caches to query and key to implement rotary position embedding (RoPE), supporting multimodal MRoPE.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_re_routing.md">torch_npu.npu_moe_re_routing</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In an MoE network, after obtaining the tokens to be computed from other devices through the AlltoAll operation, rearranges the tokens in expert order.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-matmul_checksum.md">torch_npu.matmul_checksum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Provides an AI Core error hardware fault API based on the native torch.matmul and Tensor.matmul APIs. It performs a matrix multiplication result verification process internally and compares the verification error with the verification threshold computed in real time to determine whether the verification error exceeds the threshold. If it does, an AI Core error is considered to have occurred.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_alltoallv_gmm.md">torch_npu.npu_alltoallv_gmm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In an MoE network, fuses the routed-expert AlltoAllv, Permute, and GroupedMatMul and implements parallel fusion with the shared-expert MatMul. Communication is performed before computation.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_gmm_alltoallv.md">torch_npu.npu_gmm_alltoallv</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In an MoE network, fuses the routed-expert GroupedMatMul and AlltoAllv and implements parallel fusion with the shared-expert MatMul. Computation is performed before communication.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_alltoallv_quant_gmm.md">torch_npu.npu_alltoallv_quant_gmm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Implements the quantized fusion of the routed-expert AlltoAllv and GroupedMatMul. Communication is performed before computation. It is also fused in parallel with the shared-expert MatMul computation.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_distribute_combine_add_rms_norm.md">torch_npu.npu_moe_distribute_combine_add_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses moe_distribute_combine + add + rms_norm. It must be used together with torch_npu.npu_moe_distribute_dispatch. It is equivalent to returning the data along the original path of the data collected by the npu_moe_distribute_dispatch operator and then performing the add_rms_norm operation on the data.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_transpose_batchmatmul.md">torch_npu.npu_transpose_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs the matrix multiplication of the tensor input and the tensor weight.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_update_expert.md">torch_npu.npu_moe_update_expert</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>This API supports load balancing and expert pruning. The mapped expert table and mask can be passed to the MoE layer for data distribution and processing.</p>
</td>
</tr>
<tr id="row285193313382"><td class="cellrowborder" valign="top" width="38.22%" headers="mcps1.2.3.1.1 "><p id="p2851433193819"><a name="p2851433193819"></a><a name="p2851433193819"></a><a href="torch_npu-npu_dynamic_block_quant.md">torch_npu.npu_dynamic_block_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.78%" headers="mcps1.2.3.1.2 "><p id="p785143323817"><a name="p785143323817"></a><a name="p785143323817"></a>Divides the input tensor into multiple data blocks based on the specified <code>row_block_size</code> and <code>col_block_size</code>, and performs quantization at the block level.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-set_device_limit.md">torch_npu.set_device_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Sets the number of cube and vector cores used when an operator is executed on the specified device in a process.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-get_device_limit.md">torch.npu.get_device_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Obtains the resource limit on the specified Device.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-set_stream_limit.md">torch.npu.set_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Sets the number of cube and vector cores used when operators are executed on the specified stream.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-reset_stream_limit.md">torch.npu.reset_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Restores the number of cube and vector cores used when operators are executed on the specified stream to the default configuration.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-get_stream_limit.md">torch.npu.get_stream_limit</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Obtains the number of cube and vector cores used by operators executed on the specified stream.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_sim_exponential_.md">torch_npu.npu_sim_exponential_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>In-place obtains exponential distribution sampling of the elements in the input tensor.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_dense_lightning_indexer_softmax_lse.md">torch_npu.npu_dense_lightning_indexer_softmax_lse</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>Implements the backward gradient computation during the warmup phase training of the Lightning Indexer component, while integrating the loss computation.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_dense_lightning_indexer_grad_kl_loss.md">torch_npu.npu_dense_lightning_indexer_grad_kl_loss</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>In training scenarios, <code>npu_fused_floyd_attention</code> differs from traditional FlashAttention (<code>npu_fusion_attention</code>) by treating the sequence dimension (<code>seq</code>) as an additional batch axis during QK/PV attention computation, thereby converting the attention computation into batch matrix multiplication (<code>batchMatmul</code>).</p>
</td>
</tr>
<tr id="batch_norm_reduce"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_batch_norm_reduce"><a name="p_batch_norm_reduce"></a><a name="p_batch_norm_reduce"></a><a href="torch_npu-batch_norm_reduce.md">torch_npu.batch_norm_reduce</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_batch_norm_reduce_desc"><a name="p_batch_norm_reduce_desc"></a><a name="p_batch_norm_reduce_desc"></a><span>This API reduces the input Tensor along the BatchNorm channel dimension and computes the element sum sum and the sum of squares square_sum over dimensions other than the channel dimension.<code>sum</code><code>square_sum</code>.</span></p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_apply_adam_w.md">torch_npu.npu_apply_adam_w</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Implements the parameter update computation of the AdamW optimizer, which decouples weight decay from the gradient update on the basis of Adam. It is commonly used for parameter optimization in large-model training scenarios.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm.md">torch_npu.npu_add_rms_norm</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Fuses the Add computation and RMSNorm normalization, and is commonly used to normalize the tensor after the residual connection in large models.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_v2.md">torch_npu.npu_add_rms_norm_v2</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>AddRmsNorm is the fusion of the Add operator and the RmsNorm operator. RmsNorm is a normalization operation commonly used by large models and removes the part that subtracts the mean compared with the LayerNorm operator.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_cast.md">torch_npu.npu_add_rms_norm_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>RmsNorm is a normalization operation commonly used by large models. AddRmsNormCast fuses the Cast operator after AddRmsNorm to reduce load and store operations.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_dynamic_quant.md">torch_npu.npu_add_rms_norm_dynamic_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>RmsNorm is a normalization operation commonly used by large models and removes the part that subtracts the mean compared with the LayerNorm operator. DynamicQuant is an operator that performs symmetric dynamic quantization on the input tensor. AddRmsNormDynamicQuant fuses the Add operator before RmsNorm and the one or two DynamicQuant operators to which the normalized output of RmsNorm is passed, reducing load and store operations.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_add_rms_norm_dynamic_mx_quant.md">torch_npu.npu_add_rms_norm_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>RmsNorm is a normalization operation commonly used by large models and removes the part that subtracts the mean compared with the LayerNorm operator. DynamicMxQuant is an operator that performs dynamic MX quantization by blocksize grouping on the last axis. AddRmsNormDynamicMxQuant fuses the Add operator before RmsNorm and the DynamicMxQuant operator to which the normalized output of RmsNorm is passed, reducing load and store operations.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a name="p58181296368"></a><a name="p58181296368"></a><a href="torch_npu-npu_fused_floyd_attention.md">torch_npu.npu_fused_floyd_attention</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><a name="p1281815913362"></a><a name="p1281815913362"></a>In training scenarios, compared with the traditional FA (npu_fusion_attention), npu_fused_floyd_attention additionally processes the dimension (seq) as the batch axis when computing the QK/PV attention, thereby converting the attention computation into batch matrix multiplication (batchMatmul).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_masked_causal_conv1d.md">torch_npu.npu_masked_causal_conv1d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Performs a masked causal 1D grouped convolution operation between the tokens of the hidden layer.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_clipped_swiglu.md">torch_npu.npu_clipped_swiglu</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Swish-gated linear unit activation function with clipping, which implements the variant SwiGlu computation of x.</p>
</td>
</tr>
<tr id="row_rotate_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_rotate_quant"><a name="p_rotate_quant"></a><a name="p_rotate_quant"></a><a href="torch_npu-npu_rotate_quant.md">torch_npu.npu_rotate_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_rotate_quant_desc"><a name="p_rotate_quant_desc"></a><a name="p_rotate_quant_desc"></a><span>npu_rotate_quant is a computation method that fuses rotation (Rotate) and quantization (Quant), and is used for model performance optimization.</span></p>
</td>
</tr>
<tr id="save_npugraph_tensor"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_save_npugraph_tensor"><a name="p_save_npugraph_tensor"></a><a name="p_save_npugraph_tensor"></a><a href="torch_npu-save_npugraph_tensor.md">torch_npu.save_npugraph_tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_save_npugraph_tensor_desc"><a name="p_save_npugraph_tensor_desc"></a><a name="p_save_npugraph_tensor_desc"></a><span>This API provides tensor dump capabilities similar to the native PyTorch torch.save API without affecting aclgraph capture and replay, allowing the tensor data, data type, and shape of intermediate nodes of aclgraph to be saved to a specified binary file so that the user can observe the tensor data during aclgraph execution to quickly locate problems.</span></p>
</td>
</tr>
<tr id="print_npugraph_tensor"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_print_npugraph_tensor"><a name="p_print_npugraph_tensor"></a><a name="p_print_npugraph_tensor"></a><a href="torch_npu-print_npugraph_tensor.md">torch_npu.print_npugraph_tensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_print_npugraph_tensor_desc"><a name="p_print_npugraph_tensor_desc"></a><a name="p_print_npugraph_tensor_desc"></a><span>The print_npugraph_tensor API provides tensor printing capabilities similar to the native Python print API without affecting aclgraph capture and replay, allowing the tensor data, data type, and shape of intermediate nodes of aclgraph to be printed directly, so that the user can observe the tensor data during aclgraph execution to quickly locate problems.</span></p>
</td>
</tr>
<tr id="npu_apply_rotary_pos_emb">
<td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_apply_rotary_pos_emb"><a name="p_npu_apply_rotary_pos_emb"></a><a name="p_npu_apply_rotary_pos_emb"></a><a href="torch_npu-apply_rotary_pos_emb.md">torch_npu.npu_apply_rotary_pos_emb</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_apply_rotary_pos_emb_desc"><a name="p_npu_apply_rotary_pos_emb_desc"></a><a name="p_npu_apply_rotary_pos_emb_desc"></a><span>To improve the inference network performance, the query and key operators are fused into a single path, and the result is updated in place directly in the rotary position encoding computation.</span></p>
</td>
</tr>
<tr id="npu_add_quant_gmm_"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_add_quant_gmm_"><a name="p_npu_add_quant_gmm_"></a><a name="p_npu_add_quant_gmm_"></a><a href="torch_npu-npu_add_quant_gmm_.md">torch_npu.npu_add_quant_gmm_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_add_quant_gmm__desc"><a name="p_npu_add_quant_gmm__desc"></a><a name="p_npu_add_quant_gmm__desc"></a><span>In micro-batch training scenarios, fuses the quantized GroupedMatmul and InplaceAdd operations to improve network performance.</span></p>
</td>
</tr>
<tr id="npu_add_quant_matmul_"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_add_quant_matmul_"><a name="p_npu_add_quant_matmul_"></a><a name="p_npu_add_quant_matmul_"></a><a href="torch_npu-npu_add_quant_matmul_.md">torch_npu.npu_add_quant_matmul_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_add_quant_matmul__desc"><a name="p_npu_add_quant_matmul__desc"></a><a name="p_npu_add_quant_matmul__desc"></a><span>In training scenarios, micro-batch gradient accumulation involves a large number of scenarios in which QuantBatchMatmul operations are followed by InplaceAdd operations. This API fuses the operations to improve network performance.</span></p>
</td>
</tr>
<tr id="npu_dynamic_dual_level_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_dual_level_mx_quant"><a name="p_npu_dynamic_dual_level_mx_quant"></a><a name="p_npu_dynamic_dual_level_mx_quant"></a><a href="torch_npu-npu_dynamic_dual_level_mx_quant.md">torch_npu.npu_dynamic_dual_level_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_dual_level_mx_quant_desc"><a name="p_npu_dynamic_dual_level_mx_quant_desc"></a><a name="p_npu_dynamic_dual_level_mx_quant_desc"></a><span>Implements MX quantization with a target data type of the FLOAT4 class. Only the last axis of the input tensor is quantized, and the other axes are merged.</span></p>
</td>
</tr>
<tr id="npu_dynamic_mx_quant_with_dual_axis"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_mx_quant_with_dual_axis"><a name="p_npu_dynamic_mx_quant_with_dual_axis"></a><a name="p_npu_dynamic_mx_quant_with_dual_axis"></a><a href="torch_npu-npu_dynamic_mx_quant_with_dual_axis.md">torch_npu.npu_dynamic_mx_quant_with_dual_axis</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_mx_quant_with_dual_axis_desc"><a name="p_npu_dynamic_mx_quant_with_dual_axis_desc"></a><a name="p_npu_dynamic_mx_quant_with_dual_axis_desc"></a><span>Performs MX quantization with target data types of the float4 and float8 classes on the -1 and -2 axes of the input tensor at the same time, and outputs the quantization results y1 and y2 and the quantization scales mxscale1 and mxscale2 of the two axes.</span></p>
</td>
</tr>
<tr id="npu_fused_causal_conv1d"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_fused_causal_conv1d"><a name="p_npu_fused_causal_conv1d"></a><a name="p_npu_fused_causal_conv1d"></a><a href="torch_npu-npu_fused_causal_conv1d.md">torch_npu.npu_fused_causal_conv1d</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_fused_causal_conv1d_desc"><a name="p_npu_fused_causal_conv1d_desc"></a><a name="p_npu_fused_causal_conv1d_desc"></a><span>Performs a causal 1D convolution on the sequence, using cached data (of a length equal to the convolution kernel width minus 1) to pad the head of each sequence along the sequence dimension, ensuring that the output depends on the current and historical inputs. After the convolution, the data of the current sequence part is updated to the cache. On the basis of the output of the causal 1D convolution, the original input is added to the output to implement the residual connection. Features such as APC (Automatic Prefix Caching), MTP (speculative decoding), and the residual connection are supported.</span></p>
</td>
</tr>
<tr id="npu_grouped_dynamic_block_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_grouped_dynamic_block_quant"><a name="p_npu_grouped_dynamic_block_quant"></a><a name="p_npu_grouped_dynamic_block_quant"></a><a href="torch_npu-npu_grouped_dynamic_block_quant.md">torch_npu.npu_grouped_dynamic_block_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_grouped_dynamic_block_quant_desc"><a name="p_npu_grouped_dynamic_block_quant_desc"></a><a name="p_npu_grouped_dynamic_block_quant_desc"></a><span>Quantizes each group at block granularity based on the starting values of the group indices (<code>group_list</code>) to FP8/HiFP8 and outputs the quantization parameter <code>scale</code> (<code>torch.float32</code>).</span></p></td>
</tr>
<tr id="npu_mhc_post"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_mhc_post"><a name="p_npu_mhc_post"></a><a name="p_npu_mhc_post"></a><a href="torch_npu-npu_mhc_post.md">torch_npu.npu_mhc_post</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_mhc_post_desc"><a name="p_npu_mhc_post_desc"></a><a name="p_npu_mhc_post_desc"></a><span>Performs Post Mapping on the output of the l-th layer h_l^out and Res Mapping on the input of the l-th layer x_l in the mHC (Manifold-Constrained Hyper-Connections) architecture, and then applies a residual connection to the two to obtain the input of the (l+1)-th layer x_{l+1}. When h_res is passed as None, Res Mapping is skipped and the residual addition is performed directly.<sub>l</sub><sup>out</sup><sub>l</sub><sub>l+1</sub><code>h_res</code><code>None</code></span></p></td>
</tr>
<tr id="npu_mhc_sinkhorn"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_mhc_sinkhorn"><a name="p_npu_mhc_sinkhorn"></a><a name="p_npu_mhc_sinkhorn"></a><a href="torch_npu-npu_mhc_sinkhorn.md">torch_npu.npu_mhc_sinkhorn</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_mhc_sinkhorn_desc"><a name="p_npu_mhc_sinkhorn_desc"></a><a name="p_npu_mhc_sinkhorn_desc"></a><span>Performs the Sinkhorn iterative normalization transformation on the H_res matrix in the mHC architecture (that is, the input data of the mHC layer in the network), finally obtaining the doubly stochastic matrix H_res. It supports outputting the intermediate normalization result (norm_out) and the sum result (sum_out) of the iteration process for backward gradient computation.<sub>res</sub><sub>res</sub></span></p></td>
</tr>
<tr id="npu_transpose_quant_batchmatmul"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_transpose_quant_batchmatmul"><a name="p_npu_transpose_quant_batchmatmul"></a><a name="p_npu_transpose_quant_batchmatmul"></a><a href="torch_npu-npu_transpose_quant_batchmatmul.md">torch_npu.npu_transpose_quant_batchmatmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_transpose_quant_batchmatmul_desc"><a name="p_npu_transpose_quant_batchmatmul_desc"></a><a name="p_npu_transpose_quant_batchmatmul_desc"></a><span>Performs the quantized matrix multiplication of the tensor x1 and the tensor x2. Only 3D tensors are supported. The tensors support transposition, and the transposition sequence changes according to the passed sequence. perm_x1 indicates the transposition sequence of the tensor x1, and perm_x2 indicates the transposition sequence of the tensor x2. The dimension with the sequence value 0 is the batch dimension, and the matrix multiplication is performed on the other two dimensions.</span></p></td>
</tr>
<tr id="npu_fused_linear_online_max_sum"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_fused_linear_online_max_sum"><a name="p_npu_fused_linear_online_max_sum"></a><a name="p_npu_fused_linear_online_max_sum"></a><a href="torch_npu-npu_fused_linear_online_max_sum.md">torch_npu.npu_fused_linear_online_max_sum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_fused_linear_online_max_sum_desc"><a name="p_npu_fused_linear_online_max_sum_desc"></a><a name="p_npu_fused_linear_online_max_sum_desc"></a><span>Fuses matrix multiplication and the cross-entropy preprocessing operator in vocabulary parallel scenarios. It is functionally equivalent to the MatMul and fused_vocab_parallel_cross_entropy implementations of Megatron and supports splitting the vocabulary_size dimension to fuse MatMul and CELoss. This API is the pre-communication part and must be used with npu_fused_cross_entropy_loss_with_max_sum.</span></p></td>
</tr>
<tr id="npu_fused_cross_entropy_loss_with_max_sum"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_fused_cross_entropy_loss_with_max_sum"><a name="p_npu_fused_cross_entropy_loss_with_max_sum"></a><a name="p_npu_fused_cross_entropy_loss_with_max_sum"></a><a href="torch_npu-npu_fused_cross_entropy_loss_with_max_sum.md">torch_npu.npu_fused_cross_entropy_loss_with_max_sum</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_fused_cross_entropy_loss_with_max_sum_desc"><a name="p_npu_fused_cross_entropy_loss_with_max_sum_desc"></a><a name="p_npu_fused_cross_entropy_loss_with_max_sum_desc"></a><span>Part of the cross-entropy computation module in vocabulary parallel scenarios, which solves the memory and computation efficiency problems with ultra-large vocabularies. This API is the part that computes the Loss and the Softmax result and must be used with npu_fused_linear_online_max_sum. In multi-device scenarios, global communication (all-reduce) must be performed on logits_max and sum_exp_logits before this API is called.</span></p></td>
</tr>
<tr id="npu_fused_linear_cross_entropy_loss_with_max_sum_backward"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward"><a name="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward"></a><a name="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward"></a><a href="torch_npu-npu_fused_linear_cross_entropy_loss_with_max_sum_backward.md">torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward_desc"><a name="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward_desc"></a><a name="p_npu_fused_linear_cross_entropy_loss_with_max_sum_backward_desc"></a><span>Gradient operator for cross-entropy loss computation in vocabulary parallel scenarios. It is used to compute the gradients of the leaf nodes input and weight. The related outputs of npu_fused_linear_online_max_sum and npu_fused_cross_entropy_loss_with_max_sum must be obtained as the inputs of this API. The high-performance mode (passing Softmax) and the memory-saving mode (passing logits_max and sum_exp_logits) are supported.</span></p></td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_dual_level_quant_matmul.md">torch_npu.npu_dual_level_quant_matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p><span>Secondary quantization of QuantMatmul to reduce precision loss.</span></p></td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_permute.md">torch_npu.npu_moe_token_permute</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE permute computation, which broadcasts and sorts the tokens according to the index indices.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_permute_with_routing_map.md">torch_npu.npu_moe_token_permute_with_routing_map</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE permute computation, which takes the labels of tokens and experts as routing_map and broadcasts and sorts the tokens and the optional probs according to routing_map.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_unpermute.md">torch_npu.npu_moe_token_unpermute</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>MoE unpermute computation, which obtains the input data in permuted_tokens according to the indices stored in sorted_indices, multiplies it by the optional probs, accumulates the sum, and outputs the restored computation result.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu_moe_token_unpermute_with_routing_map.md">torch_npu.npu_moe_token_unpermute_with_routing_map</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>For the permuted_tokens processed by the permute computation, accumulates them back to the original unpermuted_tokens according to sorted_indices and routing_map, supporting the drop_and_pad filling mode.</p>
</td>
</tr>
<tr id="npu_add_quant_gmm_"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_add_quant_gmm_"><a name="p_npu_add_quant_gmm_"></a><a name="p_npu_add_quant_gmm_"></a><a href="torch_npu-npu_add_quant_gmm_.md">torch_npu.npu_add_quant_gmm_</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_add_quant_gmm__desc"><a name="p_npu_add_quant_gmm__desc"></a><a name="p_npu_add_quant_gmm__desc"></a><span>In training scenarios, micro-batch gradient accumulation involves a large number of scenarios in which GroupedMatMul operations are followed by InplaceAdd operations. This API fuses the operations to implement grouped matrix multiplication computation and addition computation. Only the mx quantization, T-C quantization, and T-T quantization scenarios are supported.</span></p>
</td>
</tr>
<tr id="npu_apply_adam_w"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_apply_adam_w"><a name="p_npu_apply_adam_w"></a><a name="p_npu_apply_adam_w"></a><a href="torch_npu-npu_apply_adam_w.md">torch_npu.npu_apply_adam_w</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_apply_adam_w_desc"><a name="p_npu_apply_adam_w_desc"></a><a name="p_npu_apply_adam_w_desc"></a><span>Implements the AdamW optimizer function, decoupling weight decay from the gradient-based parameter update and updating the weight parameters and the first- and second-order moment estimates in place.</span></p></td>
</tr>
<tr id="npu_dynamic_mx_quant"><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p id="p_npu_dynamic_mx_quant"><a name="p_npu_dynamic_mx_quant"></a><a name="p_npu_dynamic_mx_quant"></a><a href="torch_npu-npu_dynamic_mx_quant.md">torch_npu.npu_dynamic_mx_quant</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p id="p_npu_dynamic_mx_quant_desc"><a name="p_npu_dynamic_mx_quant_desc"></a><a name="p_npu_dynamic_mx_quant_desc"></a><span>MX quantization with target data types of the FLOAT4 and FLOAT8 classes. On the given axis, every block_size values form a group, the quantization scale mxscale corresponding to this group is computed, and the elements in the group are then converted to the corresponding dst_type according to round_mode.</span></p>
</td>
</tr>
</tbody>
</table>
