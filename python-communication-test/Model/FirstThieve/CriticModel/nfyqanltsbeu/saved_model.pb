ภ
พ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
พ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.0-dev202103312v1.12.1-54030-g23c3e3c4ad18มม

critic_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_namecritic_1/dense_6/kernel

+critic_1/dense_6/kernel/Read/ReadVariableOpReadVariableOpcritic_1/dense_6/kernel*
_output_shapes
:	*
dtype0

critic_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namecritic_1/dense_6/bias
|
)critic_1/dense_6/bias/Read/ReadVariableOpReadVariableOpcritic_1/dense_6/bias*
_output_shapes	
:*
dtype0

critic_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_namecritic_1/dense_7/kernel

+critic_1/dense_7/kernel/Read/ReadVariableOpReadVariableOpcritic_1/dense_7/kernel*
_output_shapes
:	*
dtype0

critic_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namecritic_1/dense_7/bias
{
)critic_1/dense_7/bias/Read/ReadVariableOpReadVariableOpcritic_1/dense_7/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ฝ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*๘

value๎
B๋
 Bไ

z

common

critic
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
	bias

regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 

0
	1
2
3

0
	1
2
3
ญ
layer_regularization_losses
regularization_losses
layer_metrics
metrics
trainable_variables

layers
	variables
non_trainable_variables
 
US
VARIABLE_VALUEcritic_1/dense_6/kernel(common/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcritic_1/dense_6/bias&common/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
	1

0
	1
ญ
layer_regularization_losses

regularization_losses
layer_metrics
metrics
trainable_variables

layers
	variables
non_trainable_variables
US
VARIABLE_VALUEcritic_1/dense_7/kernel(critic/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcritic_1/dense_7/bias&critic/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
ญ
layer_regularization_losses
regularization_losses
layer_metrics
 metrics
trainable_variables

!layers
	variables
"non_trainable_variables
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
ท
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1critic_1/dense_6/kernelcritic_1/dense_6/biascritic_1/dense_7/kernelcritic_1/dense_7/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *0
f+R)
'__inference_signature_wrapper_961373008
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
์
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+critic_1/dense_6/kernel/Read/ReadVariableOp)critic_1/dense_6/bias/Read/ReadVariableOp+critic_1/dense_7/kernel/Read/ReadVariableOp)critic_1/dense_7/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *+
f&R$
"__inference__traced_save_961373082

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecritic_1/dense_6/kernelcritic_1/dense_6/biascritic_1/dense_7/kernelcritic_1/dense_7/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *.
f)R'
%__inference__traced_restore_961373104จก

ห
"__inference__traced_save_961373082
file_prefix6
2savev2_critic_1_dense_6_kernel_read_readvariableop4
0savev2_critic_1_dense_6_bias_read_readvariableop6
2savev2_critic_1_dense_7_kernel_read_readvariableop4
0savev2_critic_1_dense_7_bias_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardฆ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameล
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ื
valueอBสB(common/kernel/.ATTRIBUTES/VARIABLE_VALUEB&common/bias/.ATTRIBUTES/VARIABLE_VALUEB(critic/kernel/.ATTRIBUTES/VARIABLE_VALUEB&critic/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_critic_1_dense_6_kernel_read_readvariableop0savev2_critic_1_dense_6_bias_read_readvariableop2savev2_critic_1_dense_7_kernel_read_readvariableop0savev2_critic_1_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2บ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesก
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?
NoOpNoOp^MergeV2Checkpoints*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*:
_input_shapes)
': :	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 
ด

%__inference__traced_restore_961373104
file_prefix;
(assignvariableop_critic_1_dense_6_kernel:	7
(assignvariableop_1_critic_1_dense_6_bias:	=
*assignvariableop_2_critic_1_dense_7_kernel:	6
(assignvariableop_3_critic_1_dense_7_bias:

identity_5ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_2ขAssignVariableOp_3ห
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ื
valueอBสB(common/kernel/.ATTRIBUTES/VARIABLE_VALUEB&common/bias/.ATTRIBUTES/VARIABLE_VALUEB(critic/kernel/.ATTRIBUTES/VARIABLE_VALUEB&critic/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesฤ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityง
AssignVariableOpAssignVariableOp(assignvariableop_critic_1_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ญ
AssignVariableOp_1AssignVariableOp(assignvariableop_1_critic_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ฏ
AssignVariableOp_2AssignVariableOp*assignvariableop_2_critic_1_dense_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ญ
AssignVariableOp_3AssignVariableOp(assignvariableop_3_critic_1_dense_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpบ

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4c

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_5
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
_output_shapes
 2
NoOp_1"!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

บ
G__inference_critic_1_layer_call_and_return_conditional_losses_961372959
input_1$
dense_6_961372937:	 
dense_6_961372939:	$
dense_7_961372953:	
dense_7_961372955:
identityขdense_6/StatefulPartitionedCallขdense_7/StatefulPartitionedCallด
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6_961372937dense_6_961372939*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_6_layer_call_and_return_conditional_losses_9613729362!
dense_6/StatefulPartitionedCallิ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_961372953dense_7_961372955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_7_layer_call_and_return_conditional_losses_9613729522!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityn
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
๓
$__inference__wrapped_model_961372921
input_1B
/critic_1_dense_6_matmul_readvariableop_resource:	?
0critic_1_dense_6_biasadd_readvariableop_resource:	B
/critic_1_dense_7_matmul_readvariableop_resource:	>
0critic_1_dense_7_biasadd_readvariableop_resource:
identityข'critic_1/dense_6/BiasAdd/ReadVariableOpข&critic_1/dense_6/MatMul/ReadVariableOpข'critic_1/dense_7/BiasAdd/ReadVariableOpข&critic_1/dense_7/MatMul/ReadVariableOpม
&critic_1/dense_6/MatMul/ReadVariableOpReadVariableOp/critic_1_dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&critic_1/dense_6/MatMul/ReadVariableOpจ
critic_1/dense_6/MatMulMatMulinput_1.critic_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
critic_1/dense_6/MatMulภ
'critic_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp0critic_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'critic_1/dense_6/BiasAdd/ReadVariableOpฦ
critic_1/dense_6/BiasAddBiasAdd!critic_1/dense_6/MatMul:product:0/critic_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
critic_1/dense_6/BiasAdd
critic_1/dense_6/ReluRelu!critic_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
critic_1/dense_6/Reluม
&critic_1/dense_7/MatMul/ReadVariableOpReadVariableOp/critic_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&critic_1/dense_7/MatMul/ReadVariableOpร
critic_1/dense_7/MatMulMatMul#critic_1/dense_6/Relu:activations:0.critic_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
critic_1/dense_7/MatMulฟ
'critic_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp0critic_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'critic_1/dense_7/BiasAdd/ReadVariableOpล
critic_1/dense_7/BiasAddBiasAdd!critic_1/dense_7/MatMul:product:0/critic_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
critic_1/dense_7/BiasAdd|
IdentityIdentity!critic_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityะ
NoOpNoOp(^critic_1/dense_6/BiasAdd/ReadVariableOp'^critic_1/dense_6/MatMul/ReadVariableOp(^critic_1/dense_7/BiasAdd/ReadVariableOp'^critic_1/dense_7/MatMul/ReadVariableOp*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2R
'critic_1/dense_6/BiasAdd/ReadVariableOp'critic_1/dense_6/BiasAdd/ReadVariableOp2P
&critic_1/dense_6/MatMul/ReadVariableOp&critic_1/dense_6/MatMul/ReadVariableOp2R
'critic_1/dense_7/BiasAdd/ReadVariableOp'critic_1/dense_7/BiasAdd/ReadVariableOp2P
&critic_1/dense_7/MatMul/ReadVariableOp&critic_1/dense_7/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1

ฮ
'__inference_signature_wrapper_961373008
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *-
f(R&
$__inference__wrapped_model_9613729212
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1


๘
F__inference_dense_7_layer_call_and_return_conditional_losses_961372952

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity[
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
่

๙
F__inference_dense_6_layer_call_and_return_conditional_losses_961372936

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????2

Identity[
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
่

๙
F__inference_dense_6_layer_call_and_return_conditional_losses_961373019

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????2

Identity[
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ว
ำ
,__inference_critic_1_layer_call_fn_961372973
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
identityขStatefulPartitionedCallฌ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *P
fKRI
G__inference_critic_1_layer_call_and_return_conditional_losses_9613729592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
๏

+__inference_dense_6_layer_call_fn_961373028

inputs
unknown:	
	unknown_0:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_6_layer_call_and_return_conditional_losses_9613729362
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs


๘
F__inference_dense_7_layer_call_and_return_conditional_losses_961373038

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity[
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๎

+__inference_dense_7_layer_call_fn_961373047

inputs
unknown:	
	unknown_0:
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_7_layer_call_and_return_conditional_losses_9613729522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

IdentityD
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs"ัL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ซ
serving_default
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ฺ?
่

common

critic
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*#&call_and_return_all_conditional_losses
$__call__
%_default_save_signature"
_tf_keras_model๚{"name": "critic_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Critic", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 13]}, "float32", "input_1"]}, "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Critic"}}
ส

kernel
	bias

regularization_losses
trainable_variables
	variables
	keras_api
*&&call_and_return_all_conditional_losses
'__call__"ฅ
_tf_keras_layer{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 13]}}
ฬ

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*(&call_and_return_all_conditional_losses
)__call__"ง
_tf_keras_layer{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
 "
trackable_list_wrapper
<
0
	1
2
3"
trackable_list_wrapper
<
0
	1
2
3"
trackable_list_wrapper
ส
layer_regularization_losses
regularization_losses
layer_metrics
metrics
trainable_variables

layers
	variables
non_trainable_variables
$__call__
%_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
,
*serving_default"
signature_map
*:(	2critic_1/dense_6/kernel
$:"2critic_1/dense_6/bias
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
ญ
layer_regularization_losses

regularization_losses
layer_metrics
metrics
trainable_variables

layers
	variables
non_trainable_variables
'__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
*:(	2critic_1/dense_7/kernel
#:!2critic_1/dense_7/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ญ
layer_regularization_losses
regularization_losses
layer_metrics
 metrics
trainable_variables

!layers
	variables
"non_trainable_variables
)__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
G__inference_critic_1_layer_call_and_return_conditional_losses_961372959ฦ
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *&ข#
!
input_1?????????
๚2๗
,__inference_critic_1_layer_call_fn_961372973ฦ
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *&ข#
!
input_1?????????
โ2฿
$__inference__wrapped_model_961372921ถ
ฒ
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *&ข#
!
input_1?????????
๐2ํ
F__inference_dense_6_layer_call_and_return_conditional_losses_961373019ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ี2า
+__inference_dense_6_layer_call_fn_961373028ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๐2ํ
F__inference_dense_7_layer_call_and_return_conditional_losses_961373038ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ี2า
+__inference_dense_7_layer_call_fn_961373047ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฮBห
'__inference_signature_wrapper_961373008input_1"
ฒ
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
$__inference__wrapped_model_961372921m	0ข-
&ข#
!
input_1?????????
ช "3ช0
.
output_1"
output_1?????????ช
G__inference_critic_1_layer_call_and_return_conditional_losses_961372959_	0ข-
&ข#
!
input_1?????????
ช "%ข"

0?????????
 
,__inference_critic_1_layer_call_fn_961372973R	0ข-
&ข#
!
input_1?????????
ช "?????????ง
F__inference_dense_6_layer_call_and_return_conditional_losses_961373019]	/ข,
%ข"
 
inputs?????????
ช "&ข#

0?????????
 
+__inference_dense_6_layer_call_fn_961373028P	/ข,
%ข"
 
inputs?????????
ช "?????????ง
F__inference_dense_7_layer_call_and_return_conditional_losses_961373038]0ข-
&ข#
!
inputs?????????
ช "%ข"

0?????????
 
+__inference_dense_7_layer_call_fn_961373047P0ข-
&ข#
!
inputs?????????
ช "?????????ฃ
'__inference_signature_wrapper_961373008x	;ข8
ข 
1ช.
,
input_1!
input_1?????????"3ช0
.
output_1"
output_1?????????