# C++ APIs

This section describes the extended C++ APIs provided by Ascend Extension for PyTorch for deep optimization.

**Table 1** C++ APIs
<a name="table1728514451512"></a>
<table><thead align="left"><tr id="row1128511418151"><th class="cellrowborder" valign="top" width="36.13%" id="mcps1.2.3.1.1"><p id="p102854481517"><a name="p102854481517"></a><a name="p102854481517"></a>API</p>
</th>
<th class="cellrowborder" valign="top" width="63.870000000000005%" id="mcps1.2.3.1.2"><p id="p102851481518"><a name="p102851481518"></a><a name="p102851481518"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row182857401519"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p14285194191518"><a name="p14285194191518"></a><a name="p14285194191518"></a><a href="(beta)torch_npu-init_npu.md">(beta) torch_npu::init_npu</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p3285204101519"><a name="p3285204101519"></a><a name="p3285204101519"></a>Initializes the NPU device.</p>
</td>
</tr>
<tr id="row374312587556"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p11667145619"><a name="p11667145619"></a><a name="p11667145619"></a><a href="(beta)torch_npu-finalize_npu.md">(beta) torch_npu::finalize_npu</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1549042791610"><a name="p1549042791610"></a><a name="p1549042791610"></a>Deinitializes the NPU device, which releases NPU resources.</p>
</td>
</tr>
<tr id="row18285164121511"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p52853416159"><a name="p52853416159"></a><a name="p52853416159"></a><a href="(beta)torch-npu-synchronize.md">(beta) torch::npu::synchronize</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p14285184141513"><a name="p14285184141513"></a><a name="p14285184141513"></a>Synchronizes the NPU device. This function is identical to <code>void torch::cuda::synchronize(int64_t <em id="i5516134825616"><a name="i5516134825616"></a><a name="i5516134825616"></a>device_index </em>= -1)</code>.</p>
</td>
</tr>
<tr id="row13285194191516"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p13285154101510"><a name="p13285154101510"></a><a name="p13285154101510"></a><a href="(beta)c10-npu-current_device.md">(beta) c10::npu::current_device</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p42853421514"><a name="p42853421514"></a><a name="p42853421514"></a>Obtains the NPU device ID. The return value type is <code>DeviceIndex</code>. This function is identical to <code>c10::DeviceIndex c10::cuda::current_device()</code>.</p>
</td>
</tr>
<tr id="row7285114101517"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p12850416158"><a name="p12850416158"></a><a name="p12850416158"></a><a href="(beta)at-Device.md">(beta) at::Device</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p6285046154"><a name="p6285046154"></a><a name="p6285046154"></a>After <code>torch_npu</code> is installed, the <code>NPU</code> field is supported for the Device type, allowing a device to be specified using a string description.</p>
</td>
</tr>
<tr id="row1482153472113"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p197043314361"><a name="p197043314361"></a><a name="p197043314361"></a><a href="(beta)struct-c10_npu-NPUEvent.md">(beta) struct c10_npu::NPUEvent</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1297013314368"><a name="p1297013314368"></a><a name="p1297013314368"></a>Implements NPU event management functionality as an event class. It can be used to monitor device progress, accurately measure elapsed time, or synchronize NPU streams.</p>
</td>
</tr>
<tr id="row134831934102116"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p16592356133615"><a name="p16592356133615"></a><a name="p16592356133615"></a><a href="(beta)class-at_npu-NPUGeneratorImpl.md">(beta) class at_npu::NPUGeneratorImpl</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p142133114239"><a name="p142133114239"></a><a name="p142133114239"></a>Acts as a random number generator class that implements NPU device random number functionality and supports various random-dependent methods.</p>
</td>
</tr>
<tr id="row8483934102114"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p6592656183613"><a name="p6592656183613"></a><a name="p6592656183613"></a><a href="(beta)at_npu-detail-getDefaultNPUGenerator.md">(beta) at_npu::detail::getDefaultNPUGenerator</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1592956123617"><a name="p1592956123617"></a><a name="p1592956123617"></a>Obtains the default generator for an NPU device. The return value type is <code>Generator</code>, which is identical to <code>at::Generator& at::cuda::detail::getDefaultCUDAGenerator(c10::DeviceIndex <em id="i1647844813422"><a name="i1647844813422"></a><a name="i1647844813422"></a>device_index</em> = -1)</code>.</p>
</td>
</tr>
<tr id="row124831534202119"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p15593155612361"><a name="p15593155612361"></a><a name="p15593155612361"></a><a href="(beta)at_npu-detail-createNPUGenerator.md">(beta) at_npu::detail::createNPUGenerator</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1059375683612"><a name="p1059375683612"></a><a name="p1059375683612"></a>Creates the default generator for an NPU device. The return value type is <code>Generator</code>, which is identical to <code>at::Generator at::cuda::detail::createCUDAGenerator(c10::DeviceIndex <em id="i5642556194214"><a name="i5642556194214"></a><a name="i5642556194214"></a>device_index</em> = -1)</code>.</p>
</td>
</tr>
<tr id="row1582212372214"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p1145460103719"><a name="p1145460103719"></a><a name="p1145460103719"></a><a href="(beta)class-c10_npu-NPUStream.md">(beta) class c10_npu::NPUStream</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1245413083714"><a name="p1245413083714"></a><a name="p1245413083714"></a>Implements NPU stream management functionality as an NPU stream class. An NPU stream is a linear execution sequence belonging to an NPU device.</p>
</td>
</tr>
<tr id="row78220372215"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p174555033712"><a name="p174555033712"></a><a name="p174555033712"></a><a href="(beta)c10_npu-getNPUStreamFromPool.md">(beta) c10_npu::getNPUStreamFromPool</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p124552011374"><a name="p124552011374"></a><a name="p124552011374"></a>Obtains a new stream from the NPU stream pool. The return value type is <code>NPUStream</code>. This function is identical to <code>c10::cuda::CUDAStream c10::cuda::getStreamFromPool(const bool <em id="i117561711437"><a name="i117561711437"></a><a name="i117561711437"></a>isHighPriority</em> = false, c10::DeviceIndex <em id="i66333254311"><a name="i66333254311"></a><a name="i66333254311"></a>device</em> = -1)</code>.</p>
</td>
</tr>
<tr id="row11822183742111"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p145511016377"><a name="p145511016377"></a><a name="p145511016377"></a><a href="(beta)c10_npu-getDefaultNPUStream.md">(beta) c10_npu::getDefaultNPUStream</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p245530143713"><a name="p245530143713"></a><a name="p245530143713"></a>Obtains the default NPU stream. The return value type is <code>NPUStream</code>. This function is identical to <code>c10::cuda::CUDAStream c10::cuda::getDefaultCUDAStream(c10::DeviceIndex <em id="i1692521216432"><a name="i1692521216432"></a><a name="i1692521216432"></a>device_index</em> = -1)</code>.</p>
</td>
</tr>
<tr id="row6822103772114"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p2455402378"><a name="p2455402378"></a><a name="p2455402378"></a><a href="(beta)c10_npu-getCurrentNPUStream.md">(beta) c10_npu::getCurrentNPUStream</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p104551505370"><a name="p104551505370"></a><a name="p104551505370"></a>Obtains the current NPU stream. The return value type is <code>NPUStream</code>. This function is identical to <code>c10::cuda::CUDAStream c10::cuda::getCurrentCUDAStream(c10::DeviceIndex <em id="i1444171610436"><a name="i1444171610436"></a><a name="i1444171610436"></a>device_index</em> = -1)</code>.</p>
</td>
</tr>
<tr id="row1324644442114"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p10455150143713"><a name="p10455150143713"></a><a name="p10455150143713"></a><a href="(beta)c10_npu-setCurrentNPUStream.md">(beta) c10_npu::setCurrentNPUStream</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p184550013377"><a name="p184550013377"></a><a name="p184550013377"></a>Sets the current NPU stream. This function is identical to <code>void c10::cuda::setCurrentCUDAStream(c10::cuda::CUDAStream <em id="i1428123154319"><a name="i1428123154319"></a><a name="i1428123154319"></a>stream</em>)</code>.</p>
</td>
</tr>
<tr id="row11246164452110"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p1364718411376"><a name="p1364718411376"></a><a name="p1364718411376"></a><a href="(beta)class-at_npu-native-OpCommand.md">(beta) class at_npu::native::OpCommand</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p11647446372"><a name="p11647446372"></a><a name="p11647446372"></a>Acts as a class encapsulating low-level operator calls and implements underlying operator execution on NPU devices.</p>
</td>
</tr>
<tr id="row20187152616544"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p1935915613234"><a name="p1935915613234"></a><a name="p1935915613234"></a><a href="(beta)struct-c10_npu-NPUHooksInterface.md">(beta) struct c10_npu::NPUHooksInterface</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p83135713258"><a name="p83135713258"></a><a name="p83135713258"></a>Provides NPU hook APIs as a hook interface class.</p>
</td>
</tr>
<tr id="row518051813244"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p6180191892414"><a name="p6180191892414"></a><a name="p6180191892414"></a><a href="(beta)struct-c10_npu-NPUHooksArgs.md">(beta) struct c10_npu::NPUHooksArgs</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p51808188246"><a name="p51808188246"></a><a name="p51808188246"></a>Provides parameters for NPU hooks as a hook parameter class.</p>
</td>
</tr>
<tr id="row1728493216243"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p1328483211244"><a name="p1328483211244"></a><a name="p1328483211244"></a><a href="(beta)c10_npu-device_count.md">(beta) c10_npu::device_count</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1028463217241"><a name="p1028463217241"></a><a name="p1028463217241"></a>Obtains the number of NPU devices. The return value type is <code>DeviceIndex</code>. This function is identical to <code>c10::DeviceIndex c10::cuda::device_count()</code>.</p>
</td>
</tr>
<tr id="row1848933022417"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p124901730162414"><a name="p124901730162414"></a><a name="p124901730162414"></a><a href="(beta)c10_npu-GetDevice.md">(beta) c10_npu::GetDevice</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p2490183092411"><a name="p2490183092411"></a><a name="p2490183092411"></a>Obtains the NPU device ID. The return value type is <code>aclError</code>. This function is identical to <code>cudaError_t c10::cuda::GetDevice(int *<em id="i5414123319436"><a name="i5414123319436"></a><a name="i5414123319436"></a>device</em>)</code>.</p>
</td>
</tr>
<tr id="row017092012245"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p101705205249"><a name="p101705205249"></a><a name="p101705205249"></a><a href="(beta)c10_npu-SetDevice.md">(beta) c10_npu::SetDevice</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p6171172042412"><a name="p6171172042412"></a><a name="p6171172042412"></a>Sets the NPU device. The return value type is <code>aclError</code>. This function is identical to <code>cudaError_t c10::cuda::SetDevice(int <em id="i32541536164317"><a name="i32541536164317"></a><a name="i32541536164317"></a>device</em>)</code>.</p>
</td>
</tr>
<tr id="row166762910245"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p186712293248"><a name="p186712293248"></a><a name="p186712293248"></a><a href="(beta)c10_npu-current_device.md">(beta) c10_npu::current_device</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p06702919248"><a name="p06702919248"></a><a name="p06702919248"></a>Obtains the NPU device ID. The return value type is <code>DeviceIndex</code>, which represents the obtained device ID. This function is identical to <code>c10::DeviceIndex c10::cuda::current_device()</code>. The main difference from <code>c10_npu::GetDevice</code> is that this function includes additional error checking.</p>
</td>
</tr>
<tr id="row193943155249"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p93941215142414"><a name="p93941215142414"></a><a name="p93941215142414"></a><a href="(beta)c10_npu-set_device.md">(beta) c10_npu::set_device</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1539413151242"><a name="p1539413151242"></a><a name="p1539413151242"></a>Sets the NPU device. This function is identical to <code>void c10::cuda::set_device(c10::DeviceIndex <em id="i13672341174314"><a name="i13672341174314"></a><a name="i13672341174314"></a>device</em>)</code>. The main difference from <code>c10_npu::SetDevice</code> is that this function includes additional error checking.</p>
</td>
</tr>
<tr id="row14242145315620"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p2024245305610"><a name="p2024245305610"></a><a name="p2024245305610"></a><a href="(beta)c10_npu-warning_state.md">(beta) c10_npu::warning_state</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p142421953125611"><a name="p142421953125611"></a><a name="p142421953125611"></a>Obtains the current synchronization warning level. The return type is the <code>WarningState</code> enum class, which can be <code>L_DISABLED</code> (no warning), <code>L_WARN</code> (warning), or <code>L_ERROR</code> (error). This function is identical to <code>WarningState&amp; c10::cuda::warning_state()</code>.</p>
</td>
</tr>
<tr id="row0199201317247"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p719931392415"><a name="p719931392415"></a><a name="p719931392415"></a><a href="(beta)c10_npu-warn_or_error_on_sync.md">(beta) c10_npu::warn_or_error_on_sync</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p4199313142416"><a name="p4199313142416"></a><a name="p4199313142416"></a>Raises a warning or error during NPU synchronization, with no return value. This function raises an error or logs a warning based on the current warning level, which is identical to <code>void c10::cuda::warn_or_error_on_sync()</code>.</p>
</td>
</tr>
<tr id="row36118212415"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p76215262419"><a name="p76215262419"></a><a name="p76215262419"></a><a href="(beta)at_npu-native-get_npu_format.md">(beta) at_npu::native::get_npu_format</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p49191458195"><a name="p49191458195"></a><a name="p49191458195"></a>Obtains the format information of an NPU tensor. The return value type is <code>int64_t</code>, which represents the obtained NPU tensor format data.</p>
</td>
</tr>
<tr id="row395820517240"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p1495814522412"><a name="p1495814522412"></a><a name="p1495814522412"></a><a href="(beta)at_npu-native-get_npu_storage_sizes.md">(beta) at_npu::native::get_npu_storage_sizes</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p17958115172419"><a name="p17958115172419"></a><a name="p17958115172419"></a>Obtains the memory size of an NPU tensor. The return value type is <code>vector&lt;int64_t&gt;</code>, which represents the allocated memory size of the NPU tensor.</p>
</td>
</tr>
<tr id="row978610882412"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p678619815247"><a name="p678619815247"></a><a name="p678619815247"></a><a href="(beta)at_npu-native-npu_format_cast.md">(beta) at_npu::native::npu_format_cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1478628132414"><a name="p1478628132414"></a><a name="p1478628132414"></a>Converts the format of an NPU tensor. The return value type is <code>Tensor</code>, which represents the converted tensor.</p>
</td>
</tr>
<tr id="row178951326102411"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p98951326142411"><a name="p98951326142411"></a><a name="p98951326142411"></a><a href="(beta)at_npu-native-empty_with_format.md">(beta) at_npu::native::empty_with_format</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p12895172619247"><a name="p12895172619247"></a><a name="p12895172619247"></a>Obtains an empty NPU tensor in a specified format. The return value type is <code>Tensor</code>, which represents the obtained empty tensor.</p>
</td>
</tr>
<tr id="row15172411122412"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p1317211162415"><a name="p1317211162415"></a><a name="p1317211162415"></a><a href="(beta)c10_npu-c10_npu_get_error_message.md">(beta) c10_npu::c10_npu_get_error_message</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p61725119244"><a name="p61725119244"></a><a name="p61725119244"></a>Obtains error messages. The return value type is <code>char *</code>, which represents the obtained error message string.</p>
</td>
</tr>
<tr id="row55131018125916"><td class="cellrowborder" valign="top" width="36.13%" headers="mcps1.2.3.1.1 "><p id="p75141718135919"><a name="p75141718135919"></a><a name="p75141718135919"></a><a href="(beta)at_npu-native-npu_dropout_gen_mask.md">(beta) at_npu::native::npu_dropout_gen_mask</a></p>
</td>
<td class="cellrowborder" valign="top" width="63.870000000000005%" headers="mcps1.2.3.1.2 "><p id="p1032517615010"><a name="p1032517615010"></a><a name="p1032517615010"></a>Generates a random mask based on the probability <code>p</code> during training, which is used to set elements to zero.</p>
</td>
</tr><tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="at_npu-native-empty_with_swapped_memory.md">at_npu.native-empty_with_swapped_memory</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Allocates a tensor whose device information is NPU while its actual memory resides on the host side.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10_npu-NPUStreamGuard.md">c10_npu::NPUStreamGuard</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Manages the NPU device stream within its execution scope. This guard is identical to <code>c10::cuda::CUDAStreamGuard</code>.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10_npu-NPUStreamGuard-current_device.md">c10_npu::NPUStreamGuard::current_device</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Returns the current device set by the guard.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10_npu-NPUStreamGuard-current_stream.md">c10_npu::NPUStreamGuard::current_stream</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Returns the stream currently managed by the guard.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10_npu-NPUStreamGuard-NPUStreamGuard.md">c10_npu::NPUStreamGuard::NPUStreamGuard</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Creates a stream guard.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10_npu-NPUStreamGuard-original_device.md">c10_npu::NPUStreamGuard::original_device</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Returns the NPU device captured during guard construction.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10_npu-NPUStreamGuard-original_stream.md">c10_npu::NPUStreamGuard::original_stream</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Returns the NPU stream set during guard construction.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10_npu-NPUStreamGuard-reset_stream.md">c10_npu::NPUStreamGuard::reset_stream</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Resets the guard to a new NPU stream.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10_npu-stream_synchronize.md">c10_npu::stream_synchronize</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Synchronizes an NPU stream. This function is identical to <code>c10::cuda::stream_synchronize</code>.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10d_npu-ProcessGroupHCCL.md">c10d_npu::ProcessGroupHCCL</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Inherits from <code>c10d::Backend</code> and implements the <code>HCCL</code> backend APIs for communication operators.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="38.61%" headers="mcps1.2.3.1.1 "><p><a href="c10d_npu-ProcessGroupHCCL-batch_isend_irecv.md">c10d_npu::ProcessGroupHCCL::batch_isend_irecv</a></p>
</td>
<td class="cellrowborder" valign="top" width="61.39%" headers="mcps1.2.3.1.2 "><p>Sends or receives a batch of tensors. Asynchronously processes each point-to-point (P2P) operation in the sequence. Returns the corresponding request handle.</p>
</td>
</tr>
</tbody>
</table>
