?	o??;?@o??;?@!o??;?@	??Dq8????Dq8??!??Dq8??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$o??;?@.Ui?k|??A?M?#@Y??9z???*bX9???@|?5^:Ʃ@2U
Iterator::Model::Prefetch::Map??8?~?@!?2?$|?R@)???^a?@1Q9?C?'R@:Preprocessing2?
_Iterator::Model::Prefetch::Map::Prefetch::BatchV2::Shuffle::ParallelMapV2::FlatMap[0]::TextLine??"?tu??!??/?;0@)??"?tu??1??/?;0@:Advanced file read2?
RIterator::Model::Prefetch::Map::Prefetch::BatchV2::Shuffle::ParallelMapV2::FlatMap#???SI??!}?ǅ1E3@)?+-#????1????K@:Preprocessing2q
:Iterator::Model::Prefetch::Map::Prefetch::BatchV2::ShuffleF???????!"GN??@)?l???B??1t?٘?7@:Preprocessing2h
1Iterator::Model::Prefetch::Map::Prefetch::BatchV2.c}???!?,??@)?'?$隱?1H?????:Preprocessing2?
IIterator::Model::Prefetch::Map::Prefetch::BatchV2::Shuffle::ParallelMapV2?o?DIH??!???_????)?o?DIH??1???_????:Preprocessing2F
Iterator::ModelF?~໭?!zʭ뵗??)\?=????1???H?v??:Preprocessing2P
Iterator::Model::Prefetch??h???!???ʃ??)??h???1???ʃ??:Preprocessing2_
(Iterator::Model::Prefetch::Map::Prefetch???x!??!????????)???x!??1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??Dq8??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	.Ui?k|??.Ui?k|??!.Ui?k|??      ??!       "      ??!       *      ??!       2	?M?#@?M?#@!?M?#@:      ??!       B      ??!       J	??9z?????9z???!??9z???R      ??!       Z	??9z?????9z???!??9z???JCPU_ONLYY??Dq8??b Y      Y@q&?+F?$??"?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 