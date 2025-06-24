# （beta）torch\_npu.npu\_batch\_nms

## 函数原型

```
torch_npu.npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## 功能说明

以批量处理方式对每个类别的检测框进行非极大值抑制（Non-Maximum Suppression，NMS），从而去除冗余检测框，输出保留下来的检测框及其对应的类别和得分。

## 参数说明

-   self \(Tensor\) - 必填值，输入框的tensor，包含batch大小，数据类型Float16，输入示例：\[batch\_size, num\_anchors, q, 4\]，其中q=1或q=num\_classes。
-   scores \(Tensor\) - 必填值，输入tensor，数据类型Float16，输入示例：\[batch\_size, num\_anchors, num\_classes\]。
-   score\_threshold \(Float32\) - 必填值，指定评分过滤器的iou\_threshold，用于筛选框，去除得分较低的框，数据类型Float32。
-   iou\_threshold \(Float32\) - 必填值，指定nms的iou\_threshold，用于设定阈值，去除高于阈值的框，数据类型Float32。
-   max\_size\_per\_class \(Int\) - 必填值，指定每个类别的最大可选的框数，数据类型Int。
-   max\_total\_size \(Int\) - 必填值，指定每个batch最大可选的框数，数据类型Int。
-   change\_coordinate\_frame \(Bool，默认值为False\) -可选值，是否正则化输出框坐标矩阵，数据类型Bool。
-   transpose\_box \(Bool，默认值为False\) - 可选值，确定是否在此op之前插入转置，数据类型Bool。True表示boxes使用4，N排布。 False表示boxes使用N，4排布。

## 输出说明

-   nmsed\_boxes \(Tensor\) - shape为\(batch, max\_total\_size, 4\)的3D张量，指定每批次输出的nms框，数据类型Float16。
-   nmsed\_scores \(Tensor\) - shape为\(batch, max\_total\_size\)的2D张量，指定每批次输出的nms分数，数据类型Float16。
-   nmsed\_classes \(Tensor\) - shape为\(batch, max\_total\_size\)的2D张量，指定每批次输出的nms类，数据类型Float16。
-   nmsed\_num \(Tensor\) - shape为\(batch\)的1D张量，指定nmsed\_boxes的有效数量，数据类型Int32。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> boxes = torch.randn(8, 2, 4, 4, dtype = torch.float32).to("npu")
>>> scores = torch.randn(3, 2, 4, dtype = torch.float32).to("npu")
>>> nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
```

