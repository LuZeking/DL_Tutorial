?	???T?<@???T?<@!???T?<@	?N?͕?@?N?͕?@!?N?͕?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???T?<@?I??	???Aq?0'h/;@Y????M>??*	w???)?@2q
:Iterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2dQ???g3@!z+??tT@)Q???g3@1z+??tT@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2????&u6@!??Y???W@)5)?^?@1s?l	??$@:Preprocessing2F
Iterator::Model-?i?????!כ?UҚ@)?F?v??1?:8?h@:Preprocessing2b
+Iterator::Model::Prefetch::BatchV2::Shuffled?@J???3@!6?+?WU@)???????1????@:Preprocessing2z
CIterator::Model::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffled???B:??!??%?Q??)???B:??1??%?Q??:Preprocessing2P
Iterator::Model::Prefetch?F?????!+???je??)?F?????1+???je??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?N?͕?@#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?I??	????I??	???!?I??	???      ??!       "      ??!       *      ??!       2	q?0'h/;@q?0'h/;@!q?0'h/;@:      ??!       B      ??!       J	????M>??????M>??!????M>??R      ??!       Z	????M>??????M>??!????M>??JCPU_ONLYY?N?͕?@b Y      Y@qb?"?G???"?
device?Your program is NOT input-bound because only 4.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 