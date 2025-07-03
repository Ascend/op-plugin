# torch_npu.profiler接口列表

本章节包含采集profiling相关的自定义接口，提供性能优化所需要的数据。

**表1** torch_npu.profiler API

<a name="table135421629408"></a>
<table><thead align="left"><tr id="row155421329703"><th class="cellrowborder" valign="top" width="40.69%" id="mcps1.2.3.1.1"><p id="p05427299018"><a name="p05427299018"></a><a name="p05427299018"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="59.309999999999995%" id="mcps1.2.3.1.2"><p id="p1154313291017"><a name="p1154313291017"></a><a name="p1154313291017"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row254315291604"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p125435291302"><a name="p125435291302"></a><a name="p125435291302"></a><a href="torch_npu-profiler-profile.md">torch_npu.profiler.profile</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p138642502919"><a name="p138642502919"></a><a name="p138642502919"></a>提供PyTorch训练过程中的性能数据采集功能。</p>
</td>
</tr>
<tr id="row43841933144413"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p18384193364414"><a name="p18384193364414"></a><a name="p18384193364414"></a><a href="torch_npu-profiler-_KinetoProfile.md">torch_npu.profiler._KinetoProfile</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p133841633194420"><a name="p133841633194420"></a><a name="p133841633194420"></a>提供PyTorch训练过程中的性能数据采集功能。</p>
</td>
</tr>
<tr id="row4543529804"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p554302918019"><a name="p554302918019"></a><a name="p554302918019"></a><a href="torch_npu-profiler-ProfilerActivity.md">torch_npu.profiler.ProfilerActivity</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p554302915014"><a name="p554302915014"></a><a name="p554302915014"></a>事件采集列表，枚举类。用于赋值给torch_npu.profiler.profile的activities参数。</p>
</td>
</tr>
<tr id="row176551336013"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p186553333018"><a name="p186553333018"></a><a name="p186553333018"></a><a href="torch_npu-profiler-tensorboard_trace_handler.md">torch_npu.profiler.tensorboard_trace_handler</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p1565516332011"><a name="p1565516332011"></a><a name="p1565516332011"></a>将采集到的性能数据导出为TensorBoard工具支持的格式。作为torch_npu.profiler.profile on_trace_ready参数的执行操作。</p>
</td>
</tr>
<tr id="row265515339017"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p36557332019"><a name="p36557332019"></a><a name="p36557332019"></a><a href="torch_npu-profiler-schedule.md">torch_npu.profiler.schedule</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p365512331600"><a name="p365512331600"></a><a name="p365512331600"></a>设置不同step的行为。用于构造torch_npu.profiler.profile的schedule参数。</p>
</td>
</tr>
<tr id="row1655233801"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p17655193313011"><a name="p17655193313011"></a><a name="p17655193313011"></a><a href="torch_npu-profiler-ProfilerAction.md">torch_npu.profiler.ProfilerAction</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p12987192292514"><a name="p12987192292514"></a><a name="p12987192292514"></a>Profiler状态，Enum类型。</p>
</td>
</tr>
<tr id="row1656193313017"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p1865615334014"><a name="p1865615334014"></a><a name="p1865615334014"></a><a href="torch_npu-profiler-_ExperimentalConfig.md">torch_npu.profiler._ExperimentalConfig</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p365613331404"><a name="p365613331404"></a><a name="p365613331404"></a>性能数据采集扩展参数。用于构造torch_npu.profiler.profile的experimental_config参数。</p>
</td>
</tr>
<tr id="row49606141276"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p179611614777"><a name="p179611614777"></a><a name="p179611614777"></a><a href="torch_npu-profiler-ExportType.md">torch_npu.profiler.ExportType</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p119611814176"><a name="p119611814176"></a><a name="p119611814176"></a>设置导出的性能数据结果文件格式，作为_ExperimentalConfig类的export_type参数。</p>
</td>
</tr>
<tr id="row17322317713"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p9662554587"><a name="p9662554587"></a><a name="p9662554587"></a><a href="torch_npu-profiler-ProfilerLevel.md">torch_npu.profiler.ProfilerLevel</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p6662105410818"><a name="p6662105410818"></a><a name="p6662105410818"></a>采集等级，作为_ExperimentalConfig类的profiler_level参数。</p>
</td>
</tr>
<tr id="row528113287720"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p1466285413814"><a name="p1466285413814"></a><a name="p1466285413814"></a><a href="torch_npu-profiler-AiCMetrics.md">torch_npu.profiler.AiCMetrics</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p26626542082"><a name="p26626542082"></a><a name="p26626542082"></a>AI Core的性能指标采集项，作为_ExperimentalConfig类的aic_metrics参数。</p>
</td>
</tr>
<tr id="row108437314618"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p688716324611"><a name="p688716324611"></a><a name="p688716324611"></a><a href="torch_npu-profiler-supported_activities.md">torch_npu.profiler.supported_activities</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p118875328614"><a name="p118875328614"></a><a name="p118875328614"></a>查询当前支持采集的activities参数的CPU、NPU事件。</p>
</td>
</tr>
<tr id="row354311291203"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p115438291306"><a name="p115438291306"></a><a name="p115438291306"></a><a href="torch_npu-profiler-supported_profiler_level.md">torch_npu.profiler.supported_profiler_level</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p45431291202"><a name="p45431291202"></a><a name="p45431291202"></a>查询当前支持的torch_npu.profiler.ProfilerLevel级别。</p>
</td>
</tr>
<tr id="row175430294013"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p75431529404"><a name="p75431529404"></a><a name="p75431529404"></a><a href="torch_npu-profiler-supported_ai_core_metrics.md">torch_npu.profiler.supported_ai_core_metrics</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p1954312296015"><a name="p1954312296015"></a><a name="p1954312296015"></a>查询当前支持的torch_npu.profiler. AiCMetrics的AI Core性能指标采集项。</p>
</td>
</tr>
<tr id="row413512061712"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p213600191710"><a name="p213600191710"></a><a name="p213600191710"></a><a href="torch_npu-profiler-supported_export_type.md">torch_npu.profiler.supported_export_type</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p1713613016176"><a name="p1713613016176"></a><a name="p1713613016176"></a>查询当前支持的torch_npu.profiler.ExportType的性能数据结果文件类型。</p>
</td>
</tr>
<tr id="row1323800164710"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p152396016477"><a name="p152396016477"></a><a name="p152396016477"></a><a href="torch_npu-profiler-dynamic_profile-init.md">torch_npu.profiler.dynamic_profile.init</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p1132914219543"><a name="p1132914219543"></a><a name="p1132914219543"></a>初始化dynamic_profile动态采集。</p>
</td>
</tr>
<tr id="row1284413214717"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p084515210473"><a name="p084515210473"></a><a name="p084515210473"></a><a href="torch_npu-profiler-dynamic_profile-step.md">torch_npu.profiler.dynamic_profile.step</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p165601129165418"><a name="p165601129165418"></a><a name="p165601129165418"></a>dynamic_profile动态采集划分step。</p>
</td>
</tr>
<tr id="row11554662471"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p6554166114717"><a name="p6554166114717"></a><a name="p6554166114717"></a><a href="torch_npu-profiler-dynamic_profile-start.md">torch_npu.profiler.dynamic_profile.start</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p167431338135410"><a name="p167431338135410"></a><a name="p167431338135410"></a>触发一次dynamic_profile动态采集。</p>
</td>
</tr>
<tr id="row193641541191512"><td class="cellrowborder" valign="top" width="40.69%" headers="mcps1.2.3.1.1 "><p id="p1236434117157"><a name="p1236434117157"></a><a name="p1236434117157"></a><a href="torch_npu-profiler-profiler-analyse.md">torch_npu.profiler.profiler.analyse</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.309999999999995%" headers="mcps1.2.3.1.2 "><p id="p73641041111511"><a name="p73641041111511"></a><a name="p73641041111511"></a>Ascend PyTorch Profiler性能数据离线解析。</p>
</td>
</tr>
</tbody>
</table>

