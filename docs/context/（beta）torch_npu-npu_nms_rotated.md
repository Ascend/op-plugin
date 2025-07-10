# （beta）torch_npu.npu_nms_rotated

>**须知：**<br>
>该接口计划废弃，可以参考[小算子拼接方案](https://gitee.com/ascend/pytorch/blob/v2.1.0-6.0.rc1/test/network_ops/test_nms_rotated.py)进行替换。

## 函数原型

```
torch_npu.npu_nms_rotated(dets, scores, iou_threshold, scores_threshold=0, max_output_size=-1, mode=0) -> (Tensor, Tensor)
```

## 功能说明

按分数降序选择旋转标注框的子集。

## 参数说明

- dets (Tensor) - shape为[num_boxes, 5]的2D浮点张量。
- scores (Tensor) - shape为[num_boxes]的1D浮点张量，表示每个框（每行框）对应的一个分数。
- iou_threshold (Float) - 表示框与IoU重叠上限阈值的标量。
- scores_threshold (Float，默认值为0) - 表示决定何时删除框的分数阈值的标量。
- max_output_size (Int，默认值为-1) - 标量整数张量，表示非最大抑制下要选择的最大框数。为-1时即不施加任何约束。
- mode (Int，默认值为0) - 指定dets布局类型。如果mode设置为0，则dets的输入值为x、y、w、h和角度。如果mode设置为1，则dets的输入值为x1、y1、x2、y2和角度。

## 输出说明

- selected_index (Tensor) - shape为[M]的1D整数张量，表示从dets张量中选定的index，其中M <= max_output_size。
- selected_num (Tensor) - 0D整数张量，表示selected_index中有效元素的数量。

## 约束说明

目前不支持mode=1的场景。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> dets=torch.randn(100,5).npu()
>>> scores=torch.randn(100).npu()
>>> dets.uniform_(0,100)
>>> scores.uniform_(0,1)
>>> output1, output2 = torch_npu.npu_nms_rotated(dets, scores, 0.2, 0, -1, 1)
>>> output1
tensor([76, 48, 15, 65, 91, 82, 21, 96, 62, 90, 13, 59,  0, 18, 47, 23,  8, 56,
        55, 63, 72, 39, 97, 81, 16, 38, 17, 25, 74, 33, 79, 44, 36, 88, 83, 37,
        64, 45, 54, 41, 22, 28, 98, 40, 30, 20,  1, 86, 69, 57, 43,  9, 42, 27,
        71, 46, 19, 26, 78, 66,  3, 52], device='npu:0', dtype=torch.int32)
>>> output2
tensor([62], device='npu:0', dtype=torch.int32)
```

