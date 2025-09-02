# torch_npu.npu接口列表

本章节包含各种子模块接口，如随机数、内存管理等。npu模块支持torch.npu和torch_npu.npu两种调用方式，功能一致。

**表1** torch_npu.npu API

<a name="table0664117101518"></a>
<table><thead align="left"><tr id="row466412731514"><th class="cellrowborder" valign="top" width="37.4%" id="mcps1.2.3.1.1"><p id="p11664187201510"><a name="p11664187201510"></a><a name="p11664187201510"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.6%" id="mcps1.2.3.1.2"><p id="p10664197111510"><a name="p10664197111510"></a><a name="p10664197111510"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row96642716156"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p8664172152"><a name="p8664172152"></a><a name="p8664172152"></a><a href="（beta）torch_npu-npu-get_npu_overflow_flag.md">（beta）torch_npu.npu.get_npu_overflow_flag</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p10664117181519"><a name="p10664117181519"></a><a name="p10664117181519"></a>检测npu计算过程中是否有数值溢出。</p>
</td>
</tr>
<tr id="row1066418791520"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p166411751513"><a name="p166411751513"></a><a name="p166411751513"></a><a href="（beta）torch_npu-npu-clear_npu_overflow_flag.md">（beta）torch_npu.npu.clear_npu_overflow_flag</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p466412710151"><a name="p466412710151"></a><a name="p466412710151"></a>对npu芯片溢出检测为进行清零。</p>
</td>
</tr>
<tr id="row11664117121512"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p116643710154"><a name="p116643710154"></a><a name="p116643710154"></a><a href="torch_npu-npu-matmul-allow_hf32.md">torch_npu.npu.matmul.allow_hf32</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p26648741510"><a name="p26648741510"></a><a name="p26648741510"></a>功能和调用方式与torch.backends.cuda.matmul.allow_tf32类似。</p>
</td>
</tr>
<tr id="row5664107121515"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p36648714151"><a name="p36648714151"></a><a name="p36648714151"></a><a href="torch_npu-npu-conv-allow_hf32.md">torch_npu.npu.conv.allow_hf32</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p1366417191513"><a name="p1366417191513"></a><a name="p1366417191513"></a>功能和调用方式与torch.backends.cudnn.allow_tf32类似。</p>
</td>
</tr>
<tr id="row196641378158"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p36640731519"><a name="p36640731519"></a><a name="p36640731519"></a><a href="（beta）torch_npu-npu-set_option.md">（beta）torch_npu.npu.set_option</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p566467161517"><a name="p566467161517"></a><a name="p566467161517"></a>详细使用参见<span id="ph4356918191716"><a name="ph4356918191716"></a><a name="ph4356918191716"></a>《PyTorch 训练模型迁移调优指南》中的“<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/trainingmigrguide/PT_LMTMOG_0076.html">设置算子编译选项</a>”章节</span>。</p>
</td>
</tr>
<tr id="row14664573152"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p1366415716151"><a name="p1366415716151"></a><a name="p1366415716151"></a><a href="（beta）torch_npu-npu-config-allow_internal_format.md">（beta）torch_npu.npu.config.allow_internal_format</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p136645713159"><a name="p136645713159"></a><a name="p136645713159"></a>是否使用私有格式，设置为True时允许使用私有格式，设置为False时，不允许申请任何私有格式的tensor，避免了适配层出现私有格式流通。</p>
</td>
</tr>
<tr id="row103074675219"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p73114695216"><a name="p73114695216"></a><a name="p73114695216"></a><a href="（beta）torch_npu-npu-stress_detect.md">（beta）torch_npu.npu.stress_detect</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p14952120183714"><a name="p14952120183714"></a><a name="p14952120183714"></a>提供硬件精度在线检测接口，供模型调用。</p>
</td>
</tr>
<tr id="row67928114188"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p67934119181"><a name="p67934119181"></a><a name="p67934119181"></a><a href="（beta）torch_npu-npu-stop_device.md">（beta）torch_npu.npu.stop_device</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p141360364332"><a name="p141360364332"></a><a name="p141360364332"></a>停止对应device上的计算，对于没有执行的计算进行清除，后续在此device上执行计算会报错。</p>
</td>
</tr>
<tr id="row1050720144182"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p850721410186"><a name="p850721410186"></a><a name="p850721410186"></a><a href="（beta）torch_npu-npu-restart_device.md">（beta）torch_npu.npu.restart_device</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p750791411180"><a name="p750791411180"></a><a name="p750791411180"></a>恢复对应device上的状态，后续在此device上进行计算可以继续进行计算执行。</p>
</td>
</tr>
<tr id="row20250175212437"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p1525165254318"><a name="p1525165254318"></a><a name="p1525165254318"></a><a href="（beta）torch_npu-npu-check_uce_in_memory.md">（beta）torch_npu.npu.check_uce_in_memory</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p188835611457"><a name="p188835611457"></a><a name="p188835611457"></a>提供故障内存地址类型检测接口，供<span id="ph19255162231216"><a name="ph19255162231216"></a><a name="ph19255162231216"></a>MindCluster</span>进行故障恢复策略的决策。其功能是在出现UCE片上内存故障时，判断故障内存地址类型。</p>
</td>
</tr>
<tr id="row1154664011216"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p11546040142118"><a name="p11546040142118"></a><a name="p11546040142118"></a><a href="torch_npu-npu-SyncLaunchStream.md">torch_npu.npu.SyncLaunchStream</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p12548340152110"><a name="p12548340152110"></a><a name="p12548340152110"></a>创建一条同步下发NPUStream，在该流上下发的任务不再使用taskqueue异步下发。</p>
</td>
</tr>
<tr id="row41388371065"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p3139203712613"><a name="p3139203712613"></a><a name="p3139203712613"></a><a href="（beta）torch_npu-npu-utils-is_support_inf_nan.md">（beta）torch_npu.npu.utils.is_support_inf_nan</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p278034820716"><a name="p278034820716"></a><a name="p278034820716"></a>判断当前使用的溢出检测模式，True为INF_NAN模式，False为饱和模式。</p>
</td>
</tr>
<tr id="row10736640964"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p107361340869"><a name="p107361340869"></a><a name="p107361340869"></a><a href="（beta）torch_npu-npu-utils-npu_check_overflow.md">（beta）torch_npu.npu.utils.npu_check_overflow</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p47368401662"><a name="p47368401662"></a><a name="p47368401662"></a>检测梯度是否溢出，INF_NAN模式下检测输入Tensor是否溢出；饱和模式检查硬件溢出标志位判断。</p>
</td>
</tr>
<tr id="row7615177121914"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p961514711913"><a name="p961514711913"></a><a name="p961514711913"></a><a href="torch_npu-npu-Event()-recorded_time-().md">（beta）torch_npu.npu.Event().recorded_time()</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p1619614018512"><a name="p1619614018512"></a><a name="p1619614018512"></a>获取NPU Event对象在设备上被记录的时间。</p>
</td>
</tr>
<tr id="row1153442018911"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p1636717430248"><a name="p1636717430248"></a><a name="p1636717430248"></a><a href="（beta）torch_npu-npu-set_dump.md">（beta）torch_npu.npu.set_dump</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p1036714436247"><a name="p1036714436247"></a><a name="p1036714436247"></a>传入配置文件来配置dump参数。</p>
</td>
</tr>
<tr id="row18488172715910"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p1036734319241"><a name="p1036734319241"></a><a name="p1036734319241"></a><a href="（beta）torch_npu-npu-init_dump.md">（beta）torch_npu.npu.init_dump</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p428393116322"><a name="p428393116322"></a><a name="p428393116322"></a>初始化dump配置。</p>
</td>
</tr>
<tr id="row262919311295"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p13367154312418"><a name="p13367154312418"></a><a name="p13367154312418"></a><a href="（beta）torch_npu-npu-finalize_dump.md">（beta）torch_npu.npu.finalize_dump</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p8367104315247"><a name="p8367104315247"></a><a name="p8367104315247"></a>结束dump。</p>
</td>
</tr>
<tr id="row1629143112919"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p1936712431249"><a name="p1936712431249"></a><a name="p1936712431249"></a><a href="（beta）torch_npu-npu-set_compile_mode.md">（beta）torch_npu.npu.set_compile_mode</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p1236764317242"><a name="p1236764317242"></a><a name="p1236764317242"></a>设置是否开启二进制。</p>
</td>
</tr>
<tr id="row2027910351793"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p16367343112416"><a name="p16367343112416"></a><a name="p16367343112416"></a><a href="（beta）torch_npu-npu-is_jit_compile_false.md">（beta）torch_npu.npu.is_jit_compile_false</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p1934652175911"><a name="p1934652175911"></a><a name="p1934652175911"></a>确认算子计算是否采用的二进制，如果是二进制计算，返回True，否则返回False。</p>
</td>
</tr>
<tr id="row102805356916"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p163671443142413"><a name="p163671443142413"></a><a name="p163671443142413"></a><a href="（beta）torch_npu-npu-set_mm_bmm_format_nd.md">（beta）torch_npu.npu.set_mm_bmm_format_nd</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p1436714436245"><a name="p1436714436245"></a><a name="p1436714436245"></a>设置线性module里面的mm和bmm算子是否用ND格式。</p>
</td>
</tr>
<tr id="row628019359916"><td class="cellrowborder" valign="top" width="37.4%" headers="mcps1.2.3.1.1 "><p id="p736894362414"><a name="p736894362414"></a><a name="p736894362414"></a><a href="（beta）torch_npu-npu-get_mm_bmm_format_nd.md">（beta）torch_npu.npu.get_mm_bmm_format_nd</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.6%" headers="mcps1.2.3.1.2 "><p id="p183681243112415"><a name="p183681243112415"></a><a name="p183681243112415"></a>确认线性module里面的mm和bmm算子是否有使能ND格式，如果使能了ND，返回True，否则，返回False。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu-ExternalEvent.md">torch_npu.npu.ExternalEvent</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>ExternalEvent是AscendCL Event的封装。NPUGraph场景在执行图捕获时，ExternalEvent会被作为图外部节点被捕获，用于控制非图内时序控制场景。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu-ExternalEvent().record().md">torch_npu.npu.ExternalEvent().record()</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>在指定stream上记录Event事件。本接口被调用时，会捕获当前Stream上已下发的任务，并记录到Event事件中，因此后续若调用wait接口，会等待该Event事件中所捕获的任务都已经完成。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu-ExternalEvent().reset().md">torch_npu.npu.ExternalEvent().reset()</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>复位一个Event。Event复用场景，用于复位因record任务完成置位的标志位。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu-ExternalEvent().wait().md">torch_npu.npu.ExternalEvent().wait()</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>阻塞指定Stream的运行，直到指定的Event完成，支持多个Stream等待同一个Event的场景。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu-graph_task_group_begin.md">torch_npu.npu.graph_task_group_begin</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>NPUGraph场景下，用于标记任务组起始位置。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu-graph_task_group_end.md">torch_npu.npu.graph_task_group_end</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>NPUGraph场景下，用于标记任务组结束位置。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu-graph_task_update_begin.md">torch_npu.npu.graph_task_update_begin</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>NPUGraph场景下，用于标记待更新任务的起始。</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="torch_npu-npu-graph_task_update_end.md">torch_npu.npu.graph_task_update_end</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>NPUGraph场景下，用于标记待更新任务的结束。</p>
</td>
</tr>
<tr id="row1085193319387"><td class="cellrowborder" valign="top" width="38.22%" headers="mcps1.2.3.1.1 "><p id="p9851103393810"><a name="p9851103393810"></a><a name="p9851103393810"></a><a href="（beta）torch_npu-npu-aclnn-version.md">（beta）torch_npu.npu.aclnn.version</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.78%" headers="mcps1.2.3.1.2 "><p id="p1218213753310"><a name="p1218213753310"></a><a name="p1218213753310"></a>查询aclnn版本信息。</p>
</td>
</tr>
<tr id="row285193313382"><td class="cellrowborder" valign="top" width="38.22%" headers="mcps1.2.3.1.1 "><p id="p2851433193819"><a name="p2851433193819"></a><a name="p2851433193819"></a><a href="torch_npu-npu-aclnn-allow_hf32.md">torch_npu.npu.aclnn.allow_hf32</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.78%" headers="mcps1.2.3.1.2 "><p id="p785143323817"><a name="p785143323817"></a><a name="p785143323817"></a>设置conv算子是否支持hf32，一个属性值，对aclnn的allow_hf32属性的设置和查询。</p>
</td>
</tr>
</tbody>
</table>

**表2** amp API

<a name="table746424818186"></a>
<table><thead align="left"><tr id="row946484816186"><th class="cellrowborder" valign="top" width="37.669999999999995%" id="mcps1.2.3.1.1"><p id="p34041727199"><a name="p34041727199"></a><a name="p34041727199"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.33%" id="mcps1.2.3.1.2"><p id="p240416261919"><a name="p240416261919"></a><a name="p240416261919"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1146411483186"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p946414851817"><a name="p946414851817"></a><a name="p946414851817"></a><a href="（beta）torch_npu-npu-get_amp_supported_dtype.md">（beta）torch_npu.npu.get_amp_supported_dtype</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p1831912712373"><a name="p1831912712373"></a><a name="p1831912712373"></a>获取npu设备支持的数据类型，可能设备支持不止一种数据类型。</p>
</td>
</tr>
<tr id="row1746504816188"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p10465648191813"><a name="p10465648191813"></a><a name="p10465648191813"></a><a href="（beta）torch_npu-npu-is_autocast_enabled.md">（beta）torch_npu.npu.is_autocast_enabled</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p16465144810182"><a name="p16465144810182"></a><a name="p16465144810182"></a>确认autocast是否可用。</p>
</td>
</tr>
<tr id="row646514813186"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p04651248181813"><a name="p04651248181813"></a><a name="p04651248181813"></a><a href="（beta）torch_npu-npu-set_autocast_enabled.md">（beta）torch_npu.npu.set_autocast_enabled</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p4465194871813"><a name="p4465194871813"></a><a name="p4465194871813"></a>是否在设备上使能AMP。</p>
</td>
</tr>
<tr id="row174651448111817"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p946574881819"><a name="p946574881819"></a><a name="p946574881819"></a><a href="（beta）torch_npu-npu-get_autocast_dtype.md">（beta）torch_npu.npu.get_autocast_dtype</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p19465848131818"><a name="p19465848131818"></a><a name="p19465848131818"></a>在amp场景获取设备支持的数据类型，该dtype由torch_npu.npu.set_autocast_dtype设置或者默认数据类型torch.float16。</p>
</td>
</tr>
<tr id="row1846564819185"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p146584811184"><a name="p146584811184"></a><a name="p146584811184"></a><a href="（beta）torch_npu-npu-set_autocast_dtype.md">（beta）torch_npu.npu.set_autocast_dtype</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p1646519482184"><a name="p1646519482184"></a><a name="p1646519482184"></a>设置设备在AMP场景支持的数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表3** Random Number Generator API

<a name="table7835196103611"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001788617484_row835974815375"><th class="cellrowborder" valign="top" width="38.76%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001788617484_p435984813370"><a name="zh-cn_topic_0000001788617484_p435984813370"></a><a name="zh-cn_topic_0000001788617484_p435984813370"></a>API接口</p>
</th>
<th class="cellrowborder" valign="top" width="61.24000000000001%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001788617484_p113431258183712"><a name="zh-cn_topic_0000001788617484_p113431258183712"></a><a name="zh-cn_topic_0000001788617484_p113431258183712"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001788617484_row1616454104711"><td class="cellrowborder" valign="top" width="38.76%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p16616105411477"><a name="zh-cn_topic_0000001788617484_p16616105411477"></a><a name="zh-cn_topic_0000001788617484_p16616105411477"></a>（<span id="zh-cn_topic_0000001788617484_ph168289415199"><a name="zh-cn_topic_0000001788617484_ph168289415199"></a><a name="zh-cn_topic_0000001788617484_ph168289415199"></a>beta</span>）torch_npu.npu.get_rng_state</p>
</td>
<td class="cellrowborder" rowspan="9" valign="top" width="61.24000000000001%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001788617484_p45278137545"><a name="zh-cn_topic_0000001788617484_p45278137545"></a><a name="zh-cn_topic_0000001788617484_p45278137545"></a>Torch_npu提供随机数相关的部分接口，具体可参考<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/PyTorchNativeapi/ptaoplist_000158.html">torch.cuda</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001788617484_row1661655484720"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p116160540479"><a name="zh-cn_topic_0000001788617484_p116160540479"></a><a name="zh-cn_topic_0000001788617484_p116160540479"></a>（<span id="zh-cn_topic_0000001788617484_ph81516268336"><a name="zh-cn_topic_0000001788617484_ph81516268336"></a><a name="zh-cn_topic_0000001788617484_ph81516268336"></a>beta</span>）torch_npu.npu.set_rng_state</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001788617484_row126167543473"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p11616195434714"><a name="zh-cn_topic_0000001788617484_p11616195434714"></a><a name="zh-cn_topic_0000001788617484_p11616195434714"></a>（<span id="zh-cn_topic_0000001788617484_ph4272112818331"><a name="zh-cn_topic_0000001788617484_ph4272112818331"></a><a name="zh-cn_topic_0000001788617484_ph4272112818331"></a>beta</span>）torch_npu.npu.get_rng_state_all</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001788617484_row10616185434716"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p261675474716"><a name="zh-cn_topic_0000001788617484_p261675474716"></a><a name="zh-cn_topic_0000001788617484_p261675474716"></a>（<span id="zh-cn_topic_0000001788617484_ph166121030133313"><a name="zh-cn_topic_0000001788617484_ph166121030133313"></a><a name="zh-cn_topic_0000001788617484_ph166121030133313"></a>beta</span>）torch_npu.npu.set_rng_state_all</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001788617484_row3616135494714"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p19616105410475"><a name="zh-cn_topic_0000001788617484_p19616105410475"></a><a name="zh-cn_topic_0000001788617484_p19616105410475"></a>（<span id="zh-cn_topic_0000001788617484_ph10518103217339"><a name="zh-cn_topic_0000001788617484_ph10518103217339"></a><a name="zh-cn_topic_0000001788617484_ph10518103217339"></a>beta</span>）torch_npu.npu.manual_seed</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001788617484_row15616155411472"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p13616254184720"><a name="zh-cn_topic_0000001788617484_p13616254184720"></a><a name="zh-cn_topic_0000001788617484_p13616254184720"></a>（<span id="zh-cn_topic_0000001788617484_ph151451035113319"><a name="zh-cn_topic_0000001788617484_ph151451035113319"></a><a name="zh-cn_topic_0000001788617484_ph151451035113319"></a>beta</span>）torch_npu.npu.manual_seed_all</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001788617484_row12616854134712"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p461635474710"><a name="zh-cn_topic_0000001788617484_p461635474710"></a><a name="zh-cn_topic_0000001788617484_p461635474710"></a>（<span id="zh-cn_topic_0000001788617484_ph1734073763311"><a name="zh-cn_topic_0000001788617484_ph1734073763311"></a><a name="zh-cn_topic_0000001788617484_ph1734073763311"></a>beta</span>）torch_npu.npu.seed</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001788617484_row18616135418478"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p166169540471"><a name="zh-cn_topic_0000001788617484_p166169540471"></a><a name="zh-cn_topic_0000001788617484_p166169540471"></a>（<span id="zh-cn_topic_0000001788617484_ph19137173903319"><a name="zh-cn_topic_0000001788617484_ph19137173903319"></a><a name="zh-cn_topic_0000001788617484_ph19137173903319"></a>beta</span>）torch_npu.npu.seed_all</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001788617484_row661675419475"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001788617484_p96165545476"><a name="zh-cn_topic_0000001788617484_p96165545476"></a><a name="zh-cn_topic_0000001788617484_p96165545476"></a>（<span id="zh-cn_topic_0000001788617484_ph1394313404331"><a name="zh-cn_topic_0000001788617484_ph1394313404331"></a><a name="zh-cn_topic_0000001788617484_ph1394313404331"></a>beta</span>）torch_npu.npu.initial_seed</p>
</td>
</tr>
</tbody>
</table>

**表4** NPU device API

<a name="table14367204362414"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001835217389_row1888615477417"><th class="cellrowborder" valign="top" width="54.559999999999995%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001835217389_p188634764114"><a name="zh-cn_topic_0000001835217389_p188634764114"></a><a name="zh-cn_topic_0000001835217389_p188634764114"></a>API接口</p>
</th>
<th class="cellrowborder" valign="top" width="45.440000000000005%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001835217389_p188862474415"><a name="zh-cn_topic_0000001835217389_p188862474415"></a><a name="zh-cn_topic_0000001835217389_p188862474415"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001835217389_row939163412547"><td class="cellrowborder" valign="top" width="54.559999999999995%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p13919348541"><a name="zh-cn_topic_0000001835217389_p13919348541"></a><a name="zh-cn_topic_0000001835217389_p13919348541"></a>（<span id="zh-cn_topic_0000001835217389_ph168289415199"><a name="zh-cn_topic_0000001835217389_ph168289415199"></a><a name="zh-cn_topic_0000001835217389_ph168289415199"></a>beta</span>）torch_npu.npu.is_initialized</p>
</td>
<td class="cellrowborder" rowspan="11" valign="top" width="45.440000000000005%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001835217389_p20972816174218"><a name="zh-cn_topic_0000001835217389_p20972816174218"></a><a name="zh-cn_topic_0000001835217389_p20972816174218"></a>Torch_npu提供设备相关的部分接口，具体可参考<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/PyTorchNativeapi/ptaoplist_000158.html">torch.cuda</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row13391173465410"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p173915341547"><a name="zh-cn_topic_0000001835217389_p173915341547"></a><a name="zh-cn_topic_0000001835217389_p173915341547"></a>（<span id="zh-cn_topic_0000001835217389_ph1943317523417"><a name="zh-cn_topic_0000001835217389_ph1943317523417"></a><a name="zh-cn_topic_0000001835217389_ph1943317523417"></a>beta</span>）torch_npu.npu.init</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row17391173465415"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p0391173495420"><a name="zh-cn_topic_0000001835217389_p0391173495420"></a><a name="zh-cn_topic_0000001835217389_p0391173495420"></a>（<span id="zh-cn_topic_0000001835217389_ph351513733413"><a name="zh-cn_topic_0000001835217389_ph351513733413"></a><a name="zh-cn_topic_0000001835217389_ph351513733413"></a>beta</span>）torch_npu.npu.get_device_name</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row939118343543"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p15391534165418"><a name="zh-cn_topic_0000001835217389_p15391534165418"></a><a name="zh-cn_topic_0000001835217389_p15391534165418"></a>（<span id="zh-cn_topic_0000001835217389_ph718317101345"><a name="zh-cn_topic_0000001835217389_ph718317101345"></a><a name="zh-cn_topic_0000001835217389_ph718317101345"></a>beta</span>）torch_npu.npu.can_device_access_peer</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row539173445416"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p23911634165410"><a name="zh-cn_topic_0000001835217389_p23911634165410"></a><a name="zh-cn_topic_0000001835217389_p23911634165410"></a>（<span id="zh-cn_topic_0000001835217389_ph9238212103412"><a name="zh-cn_topic_0000001835217389_ph9238212103412"></a><a name="zh-cn_topic_0000001835217389_ph9238212103412"></a>beta</span>）torch_npu.npu.get_device_properties</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row1139193417545"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p133913348548"><a name="zh-cn_topic_0000001835217389_p133913348548"></a><a name="zh-cn_topic_0000001835217389_p133913348548"></a>（<span id="zh-cn_topic_0000001835217389_ph310411493415"><a name="zh-cn_topic_0000001835217389_ph310411493415"></a><a name="zh-cn_topic_0000001835217389_ph310411493415"></a>beta</span>）torch_npu.npu.device_of</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row19391103416545"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p20391153411547"><a name="zh-cn_topic_0000001835217389_p20391153411547"></a><a name="zh-cn_topic_0000001835217389_p20391153411547"></a>（<span id="zh-cn_topic_0000001835217389_ph171181816173418"><a name="zh-cn_topic_0000001835217389_ph171181816173418"></a><a name="zh-cn_topic_0000001835217389_ph171181816173418"></a>beta</span>）torch_npu.npu.current_blas_handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row1739173414545"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p239173495417"><a name="zh-cn_topic_0000001835217389_p239173495417"></a><a name="zh-cn_topic_0000001835217389_p239173495417"></a>（<span id="zh-cn_topic_0000001835217389_ph1024311813418"><a name="zh-cn_topic_0000001835217389_ph1024311813418"></a><a name="zh-cn_topic_0000001835217389_ph1024311813418"></a>beta</span>）torch_npu.npu.set_stream</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row13391163465411"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p33911343541"><a name="zh-cn_topic_0000001835217389_p33911343541"></a><a name="zh-cn_topic_0000001835217389_p33911343541"></a>（<span id="zh-cn_topic_0000001835217389_ph43843205344"><a name="zh-cn_topic_0000001835217389_ph43843205344"></a><a name="zh-cn_topic_0000001835217389_ph43843205344"></a>beta</span>）torch_npu.npu.set_sync_debug_mode</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row73912034125411"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p173918344543"><a name="zh-cn_topic_0000001835217389_p173918344543"></a><a name="zh-cn_topic_0000001835217389_p173918344543"></a>（<span id="zh-cn_topic_0000001835217389_ph15373102263415"><a name="zh-cn_topic_0000001835217389_ph15373102263415"></a><a name="zh-cn_topic_0000001835217389_ph15373102263415"></a>beta</span>）torch_npu.npu.get_sync_debug_mode</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row19886154310548"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p1688617434546"><a name="zh-cn_topic_0000001835217389_p1688617434546"></a><a name="zh-cn_topic_0000001835217389_p1688617434546"></a>（<span id="zh-cn_topic_0000001835217389_ph94981924173410"><a name="zh-cn_topic_0000001835217389_ph94981924173410"></a><a name="zh-cn_topic_0000001835217389_ph94981924173410"></a>beta</span>）torch_npu.npu.utilization</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835217389_row16363129161417"><td class="cellrowborder" valign="top" width="54.559999999999995%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835217389_p83919341546"><a name="zh-cn_topic_0000001835217389_p83919341546"></a><a name="zh-cn_topic_0000001835217389_p83919341546"></a>（<span id="zh-cn_topic_0000001835217389_ph871092653413"><a name="zh-cn_topic_0000001835217389_ph871092653413"></a><a name="zh-cn_topic_0000001835217389_ph871092653413"></a>beta</span>）torch_npu.npu.get_device_capability</p>
</td>
<td class="cellrowborder" valign="top" width="45.440000000000005%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001835217389_p1036412951413"><a name="zh-cn_topic_0000001835217389_p1036412951413"></a><a name="zh-cn_topic_0000001835217389_p1036412951413"></a>预留接口，暂不支持，接口默认返回None。</p>
</td>
</tr>
</tbody>
</table>

**表5** Memory management API

<a name="table1766013612371"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001863035964_row1357083844218"><th class="cellrowborder" valign="top" width="63.61%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001863035964_p1757053818424"><a name="zh-cn_topic_0000001863035964_p1757053818424"></a><a name="zh-cn_topic_0000001863035964_p1757053818424"></a>API接口</p>
</th>
<th class="cellrowborder" valign="top" width="36.39%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001863035964_p2057018381424"><a name="zh-cn_topic_0000001863035964_p2057018381424"></a><a name="zh-cn_topic_0000001863035964_p2057018381424"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001863035964_row12860101605513"><td class="cellrowborder" valign="top" width="63.61%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p486019163553"><a name="zh-cn_topic_0000001863035964_p486019163553"></a><a name="zh-cn_topic_0000001863035964_p486019163553"></a>（<span id="zh-cn_topic_0000001863035964_ph168289415199"><a name="zh-cn_topic_0000001863035964_ph168289415199"></a><a name="zh-cn_topic_0000001863035964_ph168289415199"></a>beta</span>）torch_npu.npu.caching_allocator_alloc</p>
</td>
<td class="cellrowborder" rowspan="18" valign="top" width="36.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863035964_p6146131084320"><a name="zh-cn_topic_0000001863035964_p6146131084320"></a><a name="zh-cn_topic_0000001863035964_p6146131084320"></a>Torch_npu提供内存管理相关的部分接口，具体可参考<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/PyTorchNativeapi/ptaoplist_000158.html">torch.cuda</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row0860131635519"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p19860111610554"><a name="zh-cn_topic_0000001863035964_p19860111610554"></a><a name="zh-cn_topic_0000001863035964_p19860111610554"></a>（<span id="zh-cn_topic_0000001863035964_ph1545853943420"><a name="zh-cn_topic_0000001863035964_ph1545853943420"></a><a name="zh-cn_topic_0000001863035964_ph1545853943420"></a>beta</span>）torch_npu.npu.caching_allocator_delete</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row1486071645519"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p1886031615519"><a name="zh-cn_topic_0000001863035964_p1886031615519"></a><a name="zh-cn_topic_0000001863035964_p1886031615519"></a>（<span id="zh-cn_topic_0000001863035964_ph62311441113420"><a name="zh-cn_topic_0000001863035964_ph62311441113420"></a><a name="zh-cn_topic_0000001863035964_ph62311441113420"></a>beta</span>）torch_npu.npu.set_per_process_memory_fraction</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row12860161610554"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p4860151635519"><a name="zh-cn_topic_0000001863035964_p4860151635519"></a><a name="zh-cn_topic_0000001863035964_p4860151635519"></a>（<span id="zh-cn_topic_0000001863035964_ph78914314341"><a name="zh-cn_topic_0000001863035964_ph78914314341"></a><a name="zh-cn_topic_0000001863035964_ph78914314341"></a>beta</span>）torch_npu.npu.empty_cache</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row38601116185513"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p9860516115513"><a name="zh-cn_topic_0000001863035964_p9860516115513"></a><a name="zh-cn_topic_0000001863035964_p9860516115513"></a>（<span id="zh-cn_topic_0000001863035964_ph11300154553410"><a name="zh-cn_topic_0000001863035964_ph11300154553410"></a><a name="zh-cn_topic_0000001863035964_ph11300154553410"></a>beta</span>）torch_npu.npu.memory_stats</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row1186031665513"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p10860316205512"><a name="zh-cn_topic_0000001863035964_p10860316205512"></a><a name="zh-cn_topic_0000001863035964_p10860316205512"></a>（<span id="zh-cn_topic_0000001863035964_ph111981047103411"><a name="zh-cn_topic_0000001863035964_ph111981047103411"></a><a name="zh-cn_topic_0000001863035964_ph111981047103411"></a>beta</span>）torch_npu.npu.memory_stats_as_nested_dict</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row16860116135518"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p88601416155512"><a name="zh-cn_topic_0000001863035964_p88601416155512"></a><a name="zh-cn_topic_0000001863035964_p88601416155512"></a>（<span id="zh-cn_topic_0000001863035964_ph12169174953412"><a name="zh-cn_topic_0000001863035964_ph12169174953412"></a><a name="zh-cn_topic_0000001863035964_ph12169174953412"></a>beta</span>）torch_npu.npu.reset_accumulated_memory_stats</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row68601916145518"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p4860141675510"><a name="zh-cn_topic_0000001863035964_p4860141675510"></a><a name="zh-cn_topic_0000001863035964_p4860141675510"></a>（<span id="zh-cn_topic_0000001863035964_ph678145115340"><a name="zh-cn_topic_0000001863035964_ph678145115340"></a><a name="zh-cn_topic_0000001863035964_ph678145115340"></a>beta</span>）torch_npu.npu.reset_peak_memory_stats</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row3860716175511"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p178604162559"><a name="zh-cn_topic_0000001863035964_p178604162559"></a><a name="zh-cn_topic_0000001863035964_p178604162559"></a>（<span id="zh-cn_topic_0000001863035964_ph59721052153419"><a name="zh-cn_topic_0000001863035964_ph59721052153419"></a><a name="zh-cn_topic_0000001863035964_ph59721052153419"></a>beta</span>）torch_npu.npu.reset_max_memory_allocated</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row886012165551"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p1786031612559"><a name="zh-cn_topic_0000001863035964_p1786031612559"></a><a name="zh-cn_topic_0000001863035964_p1786031612559"></a>（<span id="zh-cn_topic_0000001863035964_ph1373115517349"><a name="zh-cn_topic_0000001863035964_ph1373115517349"></a><a name="zh-cn_topic_0000001863035964_ph1373115517349"></a>beta</span>）torch_npu.npu.reset_max_memory_cached</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row286091645515"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p9860111655513"><a name="zh-cn_topic_0000001863035964_p9860111655513"></a><a name="zh-cn_topic_0000001863035964_p9860111655513"></a>（<span id="zh-cn_topic_0000001863035964_ph0844859143419"><a name="zh-cn_topic_0000001863035964_ph0844859143419"></a><a name="zh-cn_topic_0000001863035964_ph0844859143419"></a>beta</span>）torch_npu.npu.memory_allocated</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row286071611551"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p10860101655517"><a name="zh-cn_topic_0000001863035964_p10860101655517"></a><a name="zh-cn_topic_0000001863035964_p10860101655517"></a>（<span id="zh-cn_topic_0000001863035964_ph128776112351"><a name="zh-cn_topic_0000001863035964_ph128776112351"></a><a name="zh-cn_topic_0000001863035964_ph128776112351"></a>beta</span>）torch_npu.npu.max_memory_allocated</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row78608164553"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p18603168555"><a name="zh-cn_topic_0000001863035964_p18603168555"></a><a name="zh-cn_topic_0000001863035964_p18603168555"></a>（<span id="zh-cn_topic_0000001863035964_ph2032720410358"><a name="zh-cn_topic_0000001863035964_ph2032720410358"></a><a name="zh-cn_topic_0000001863035964_ph2032720410358"></a>beta</span>）torch_npu.npu.memory_reserved</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row1686071675513"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p5860121613559"><a name="zh-cn_topic_0000001863035964_p5860121613559"></a><a name="zh-cn_topic_0000001863035964_p5860121613559"></a>（<span id="zh-cn_topic_0000001863035964_ph3247476355"><a name="zh-cn_topic_0000001863035964_ph3247476355"></a><a name="zh-cn_topic_0000001863035964_ph3247476355"></a>beta</span>）torch_npu.npu.max_memory_reserved</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row1486051675515"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p48601516195510"><a name="zh-cn_topic_0000001863035964_p48601516195510"></a><a name="zh-cn_topic_0000001863035964_p48601516195510"></a>（<span id="zh-cn_topic_0000001863035964_ph6190189103516"><a name="zh-cn_topic_0000001863035964_ph6190189103516"></a><a name="zh-cn_topic_0000001863035964_ph6190189103516"></a>beta</span>）torch_npu.npu.memory_cached</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row0860416105516"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p15860151685516"><a name="zh-cn_topic_0000001863035964_p15860151685516"></a><a name="zh-cn_topic_0000001863035964_p15860151685516"></a>（<span id="zh-cn_topic_0000001863035964_ph188631121350"><a name="zh-cn_topic_0000001863035964_ph188631121350"></a><a name="zh-cn_topic_0000001863035964_ph188631121350"></a>beta</span>）torch_npu.npu.max_memory_cached</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row186031675517"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p486041618550"><a name="zh-cn_topic_0000001863035964_p486041618550"></a><a name="zh-cn_topic_0000001863035964_p486041618550"></a>（<span id="zh-cn_topic_0000001863035964_ph844815154352"><a name="zh-cn_topic_0000001863035964_ph844815154352"></a><a name="zh-cn_topic_0000001863035964_ph844815154352"></a>beta</span>）torch_npu.npu.memory_snapshot</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row10860101635512"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p1486012165555"><a name="zh-cn_topic_0000001863035964_p1486012165555"></a><a name="zh-cn_topic_0000001863035964_p1486012165555"></a>（<span id="zh-cn_topic_0000001863035964_ph34464179352"><a name="zh-cn_topic_0000001863035964_ph34464179352"></a><a name="zh-cn_topic_0000001863035964_ph34464179352"></a>beta</span>）torch_npu.npu.memory_summary</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row1933424017243"><td class="cellrowborder" valign="top" width="63.61%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p7819165411224"><a name="zh-cn_topic_0000001863035964_p7819165411224"></a><a name="zh-cn_topic_0000001863035964_p7819165411224"></a>torch.npu.npu.NPUPluggableAllocator</p>
</td>
<td class="cellrowborder" valign="top" width="36.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863035964_p172891387259"><a name="zh-cn_topic_0000001863035964_p172891387259"></a><a name="zh-cn_topic_0000001863035964_p172891387259"></a>该接口涉及高危操作，使用请参考<a href="torch-npu-npu-NPUPluggableAllocator.md">torch.npu.npu.NPUPluggableAllocator</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001863035964_row12478193618244"><td class="cellrowborder" valign="top" width="63.61%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001863035964_p681913541224"><a name="zh-cn_topic_0000001863035964_p681913541224"></a><a name="zh-cn_topic_0000001863035964_p681913541224"></a>torch.npu.npu.change_current_allocator</p>
</td>
<td class="cellrowborder" valign="top" width="36.39%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001863035964_p75031810165517"><a name="zh-cn_topic_0000001863035964_p75031810165517"></a><a name="zh-cn_topic_0000001863035964_p75031810165517"></a>该接口涉及高危操作，使用请参考<a href="torch-npu-npu-change_current_allocator.md">torch.npu.npu.change_current_allocator</a>。</p>
</td>
</tr>
</tbody>
</table>

**表6** aoe API

<a name="table225713163012"></a>
<table><thead align="left"><tr id="row192578383010"><th class="cellrowborder" valign="top" width="37.730000000000004%" id="mcps1.2.3.1.1"><p id="p141019192305"><a name="p141019192305"></a><a name="p141019192305"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.27%" id="mcps1.2.3.1.2"><p id="p17101121913019"><a name="p17101121913019"></a><a name="p17101121913019"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row22571533307"><td class="cellrowborder" valign="top" width="37.730000000000004%" headers="mcps1.2.3.1.1 "><p id="p0257163143012"><a name="p0257163143012"></a><a name="p0257163143012"></a><a href="（beta）torch_npu-npu-set_aoe.md">（beta）torch_npu.npu.set_aoe</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.27%" headers="mcps1.2.3.1.2 "><p id="p172571338306"><a name="p172571338306"></a><a name="p172571338306"></a>AOE调优使能。</p>
</td>
</tr>
</tbody>
</table>

**表7** Profiler API

<a name="table17382716193111"></a>
<table><thead align="left"><tr id="row1238261612314"><th class="cellrowborder" valign="top" width="37.97%" id="mcps1.2.3.1.1"><p id="p18285834183113"><a name="p18285834183113"></a><a name="p18285834183113"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="62.029999999999994%" id="mcps1.2.3.1.2"><p id="p14285193423110"><a name="p14285193423110"></a><a name="p14285193423110"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1261071265817"><td class="cellrowborder" valign="top" width="37.97%" headers="mcps1.2.3.1.1 "><p id="p2611712185812"><a name="p2611712185812"></a><a name="p2611712185812"></a><a href="torch_npu-npu-mstx.md">torch_npu.npu.mstx</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.029999999999994%" headers="mcps1.2.3.1.2 "><p id="p5611012105820"><a name="p5611012105820"></a><a name="p5611012105820"></a>打点接口。</p>
</td>
</tr>
<tr id="row35671815125310"><td class="cellrowborder" valign="top" width="37.97%" headers="mcps1.2.3.1.1 "><p id="p756718158538"><a name="p756718158538"></a><a name="p756718158538"></a><a href="torch_npu-npu-mstx-mark.md">torch_npu.npu.mstx.mark</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.029999999999994%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001983773542_p2038911486504"><a name="zh-cn_topic_0000001983773542_p2038911486504"></a><a name="zh-cn_topic_0000001983773542_p2038911486504"></a>标记瞬时事件。</p>
</td>
</tr>
<tr id="row2068217208539"><td class="cellrowborder" valign="top" width="37.97%" headers="mcps1.2.3.1.1 "><p id="p568362035310"><a name="p568362035310"></a><a name="p568362035310"></a><a href="torch_npu-npu-mstx-range_start.md">torch_npu.npu.mstx.range_start</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.029999999999994%" headers="mcps1.2.3.1.2 "><p id="p2038911486504"><a name="p2038911486504"></a><a name="p2038911486504"></a>标识打点开始。</p>
</td>
</tr>
<tr id="row1373517244535"><td class="cellrowborder" valign="top" width="37.97%" headers="mcps1.2.3.1.1 "><p id="p673592411537"><a name="p673592411537"></a><a name="p673592411537"></a><a href="torch_npu-npu-mstx-range_end.md">torch_npu.npu.mstx.range_end</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.029999999999994%" headers="mcps1.2.3.1.2 "><p id="p1995741111568"><a name="p1995741111568"></a><a name="p1995741111568"></a>标识打点结束。</p>
</td>
</tr>
<tr id="row1159973155314"><td class="cellrowborder" valign="top" width="37.97%" headers="mcps1.2.3.1.1 "><p id="p115992314539"><a name="p115992314539"></a><a name="p115992314539"></a><a href="torch_npu-npu-mstx-mstx_range.md">torch_npu.npu.mstx.mstx_range</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.029999999999994%" headers="mcps1.2.3.1.2 "><p id="p1059918319532"><a name="p1059918319532"></a><a name="p1059918319532"></a>range装饰器，用来采集被装饰函数的range执行耗时。</p>
</td>
</tr>
</tbody>
</table>

**表8** torch_npu Storage API

<a name="table15668753173210"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001835257369_row12327105314411"><th class="cellrowborder" valign="top" width="55.25%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001835257369_p43277535444"><a name="zh-cn_topic_0000001835257369_p43277535444"></a><a name="zh-cn_topic_0000001835257369_p43277535444"></a>API接口</p>
</th>
<th class="cellrowborder" valign="top" width="44.75%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001835257369_p11327125394418"><a name="zh-cn_topic_0000001835257369_p11327125394418"></a><a name="zh-cn_topic_0000001835257369_p11327125394418"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001835257369_row16495752115611"><td class="cellrowborder" valign="top" width="55.25%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p7524101655810"><a name="zh-cn_topic_0000001835257369_p7524101655810"></a><a name="zh-cn_topic_0000001835257369_p7524101655810"></a>torch_npu.npu.BoolStorage</p>
</td>
<td class="cellrowborder" rowspan="9" valign="top" width="44.75%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001835257369_p8060118"><a name="zh-cn_topic_0000001835257369_p8060118"></a><a name="zh-cn_topic_0000001835257369_p8060118"></a>功能和调用方式与torch.Storage相同，具体请参考<a href="https://pytorch.org/docs/stable/storage.html#" target="_blank" rel="noopener noreferrer">https://pytorch.org/docs/stable/storage.html#</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257369_row134951052155613"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p16191172114583"><a name="zh-cn_topic_0000001835257369_p16191172114583"></a><a name="zh-cn_topic_0000001835257369_p16191172114583"></a>torch_npu.npu.ByteStorage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257369_row10495145210565"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p9495175211563"><a name="zh-cn_topic_0000001835257369_p9495175211563"></a><a name="zh-cn_topic_0000001835257369_p9495175211563"></a>torch_npu.npu.ShortStorage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257369_row17495195275616"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p8495252135612"><a name="zh-cn_topic_0000001835257369_p8495252135612"></a><a name="zh-cn_topic_0000001835257369_p8495252135612"></a>torch_npu.npu.LongStorage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257369_row18006055911"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p07506945914"><a name="zh-cn_topic_0000001835257369_p07506945914"></a><a name="zh-cn_topic_0000001835257369_p07506945914"></a>torch_npu.npu.IntStorage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257369_row14341175718581"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p03411057145811"><a name="zh-cn_topic_0000001835257369_p03411057145811"></a><a name="zh-cn_topic_0000001835257369_p03411057145811"></a>torch_npu.npu.HalfStorage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257369_row14341115716584"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p734119575584"><a name="zh-cn_topic_0000001835257369_p734119575584"></a><a name="zh-cn_topic_0000001835257369_p734119575584"></a>torch_npu.npu.CharStorage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257369_row149709474586"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p69703478581"><a name="zh-cn_topic_0000001835257369_p69703478581"></a><a name="zh-cn_topic_0000001835257369_p69703478581"></a>torch_npu.npu.DoubleStorage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257369_row5813195115817"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001835257369_p9813125165810"><a name="zh-cn_topic_0000001835257369_p9813125165810"></a><a name="zh-cn_topic_0000001835257369_p9813125165810"></a>torch_npu.npu.FloatStorage</p>
</td>
</tr>
</tbody>
</table>

**表9** NPU Tensor API(beta)

<a name="table1576442216332"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001835257345_row74720376177"><th class="cellrowborder" valign="top" width="26.75%" id="mcps1.2.5.1.1"><p id="zh-cn_topic_0000001835257345_p18161141918211"><a name="zh-cn_topic_0000001835257345_p18161141918211"></a><a name="zh-cn_topic_0000001835257345_p18161141918211"></a>PyTorch原生API名称</p>
</th>
<th class="cellrowborder" valign="top" width="30.349999999999998%" id="mcps1.2.5.1.2"><p id="zh-cn_topic_0000001835257345_p3164023235"><a name="zh-cn_topic_0000001835257345_p3164023235"></a><a name="zh-cn_topic_0000001835257345_p3164023235"></a>NPU形式名称</p>
</th>
<th class="cellrowborder" valign="top" width="6.22%" id="mcps1.2.5.1.3"><p id="zh-cn_topic_0000001835257345_p216112194214"><a name="zh-cn_topic_0000001835257345_p216112194214"></a><a name="zh-cn_topic_0000001835257345_p216112194214"></a>是否支持</p>
</th>
<th class="cellrowborder" valign="top" width="36.68%" id="mcps1.2.5.1.4"><p id="zh-cn_topic_0000001835257345_p151413620217"><a name="zh-cn_topic_0000001835257345_p151413620217"></a><a name="zh-cn_topic_0000001835257345_p151413620217"></a>参考链接</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001835257345_row16728461210"><td class="cellrowborder" valign="top" width="26.75%" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p137218461814"><a name="zh-cn_topic_0000001835257345_p137218461814"></a><a name="zh-cn_topic_0000001835257345_p137218461814"></a>torch.cuda.DoubleTensor</p>
</td>
<td class="cellrowborder" valign="top" width="30.349999999999998%" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p1996044910114"><a name="zh-cn_topic_0000001835257345_p1996044910114"></a><a name="zh-cn_topic_0000001835257345_p1996044910114"></a>torch_npu.npu.DoubleTensor</p>
</td>
<td class="cellrowborder" valign="top" width="6.22%" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p27215465118"><a name="zh-cn_topic_0000001835257345_p27215465118"></a><a name="zh-cn_topic_0000001835257345_p27215465118"></a>是</p>
</td>
<td class="cellrowborder" rowspan="9" valign="top" width="36.68%" headers="mcps1.2.5.1.4 "><p id="zh-cn_topic_0000001835257345_p225252418262"><a name="zh-cn_topic_0000001835257345_p225252418262"></a><a name="zh-cn_topic_0000001835257345_p225252418262"></a><a href="https://pytorch.org/docs/stable/tensors.html#data-types" target="_blank" rel="noopener noreferrer">https://pytorch.org/docs/stable/tensors.html#data-types</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257345_row47210461517"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p4725461314"><a name="zh-cn_topic_0000001835257345_p4725461314"></a><a name="zh-cn_topic_0000001835257345_p4725461314"></a>torch.cuda.ShortTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p49600493114"><a name="zh-cn_topic_0000001835257345_p49600493114"></a><a name="zh-cn_topic_0000001835257345_p49600493114"></a>torch_npu.npu.ShortTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p772146719"><a name="zh-cn_topic_0000001835257345_p772146719"></a><a name="zh-cn_topic_0000001835257345_p772146719"></a>是</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257345_row19728461116"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p19728461012"><a name="zh-cn_topic_0000001835257345_p19728461012"></a><a name="zh-cn_topic_0000001835257345_p19728461012"></a>torch.cuda.CharTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p109609491714"><a name="zh-cn_topic_0000001835257345_p109609491714"></a><a name="zh-cn_topic_0000001835257345_p109609491714"></a>torch_npu.npu.CharTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p8724460120"><a name="zh-cn_topic_0000001835257345_p8724460120"></a><a name="zh-cn_topic_0000001835257345_p8724460120"></a>是</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257345_row47317466110"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p1873154618118"><a name="zh-cn_topic_0000001835257345_p1873154618118"></a><a name="zh-cn_topic_0000001835257345_p1873154618118"></a>torch.cuda.ByteTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p1896013491711"><a name="zh-cn_topic_0000001835257345_p1896013491711"></a><a name="zh-cn_topic_0000001835257345_p1896013491711"></a>torch_npu.npu.ByteTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p11739461719"><a name="zh-cn_topic_0000001835257345_p11739461719"></a><a name="zh-cn_topic_0000001835257345_p11739461719"></a>是</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257345_row1509152312215"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p5471124912115"><a name="zh-cn_topic_0000001835257345_p5471124912115"></a><a name="zh-cn_topic_0000001835257345_p5471124912115"></a>torch.cuda.FloatTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p195091923112118"><a name="zh-cn_topic_0000001835257345_p195091923112118"></a><a name="zh-cn_topic_0000001835257345_p195091923112118"></a>torch_npu.npu.FloatTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p155091223162115"><a name="zh-cn_topic_0000001835257345_p155091223162115"></a><a name="zh-cn_topic_0000001835257345_p155091223162115"></a>是</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257345_row16510192372120"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p8471144914219"><a name="zh-cn_topic_0000001835257345_p8471144914219"></a><a name="zh-cn_topic_0000001835257345_p8471144914219"></a>torch.cuda.HalfTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p155102231215"><a name="zh-cn_topic_0000001835257345_p155102231215"></a><a name="zh-cn_topic_0000001835257345_p155102231215"></a>torch_npu.npu.HalfTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p951012235213"><a name="zh-cn_topic_0000001835257345_p951012235213"></a><a name="zh-cn_topic_0000001835257345_p951012235213"></a>是</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257345_row17861230162113"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p9584195616219"><a name="zh-cn_topic_0000001835257345_p9584195616219"></a><a name="zh-cn_topic_0000001835257345_p9584195616219"></a>torch.cuda.IntTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p158615305219"><a name="zh-cn_topic_0000001835257345_p158615305219"></a><a name="zh-cn_topic_0000001835257345_p158615305219"></a>torch_npu.npu.IntTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p5861530152110"><a name="zh-cn_topic_0000001835257345_p5861530152110"></a><a name="zh-cn_topic_0000001835257345_p5861530152110"></a>是</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257345_row8752162662111"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p95841156132111"><a name="zh-cn_topic_0000001835257345_p95841156132111"></a><a name="zh-cn_topic_0000001835257345_p95841156132111"></a>torch.cuda.BoolTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p1575212662118"><a name="zh-cn_topic_0000001835257345_p1575212662118"></a><a name="zh-cn_topic_0000001835257345_p1575212662118"></a>torch_npu.npu.BoolTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p13752162642116"><a name="zh-cn_topic_0000001835257345_p13752162642116"></a><a name="zh-cn_topic_0000001835257345_p13752162642116"></a>是</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001835257345_row97533261212"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0000001835257345_p6753182613210"><a name="zh-cn_topic_0000001835257345_p6753182613210"></a><a name="zh-cn_topic_0000001835257345_p6753182613210"></a>torch.cuda.LongTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0000001835257345_p775314269219"><a name="zh-cn_topic_0000001835257345_p775314269219"></a><a name="zh-cn_topic_0000001835257345_p775314269219"></a>torch_npu.npu.LongTensor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0000001835257345_p975372632115"><a name="zh-cn_topic_0000001835257345_p975372632115"></a><a name="zh-cn_topic_0000001835257345_p975372632115"></a>是</p>
</td>
</tr>
</tbody>
</table>

