??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?

layer1

layer2
	actor
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

	kernel

bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*
	0

1
2
3
4
5
*
	0

1
2
3
4
5
 
?
	variables
trainable_variables
regularization_losses

layers
metrics
layer_metrics
non_trainable_variables
layer_regularization_losses
 
JH
VARIABLE_VALUEdense/kernel(layer1/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&layer1/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1

	0

1
 
?
	variables
trainable_variables
regularization_losses

 layers
!metrics
"layer_metrics
#non_trainable_variables
$layer_regularization_losses
LJ
VARIABLE_VALUEdense_1/kernel(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&layer2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
trainable_variables
regularization_losses

%layers
&metrics
'layer_metrics
(non_trainable_variables
)layer_regularization_losses
KI
VARIABLE_VALUEdense_2/kernel'actor/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEdense_2/bias%actor/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
trainable_variables
regularization_losses

*layers
+metrics
,layer_metrics
-non_trainable_variables
.layer_regularization_losses

0
1
2
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
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_5246
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *&
f!R
__inference__traced_save_5347
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *)
f$R"
 __inference__traced_restore_5375??
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_5165

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
__inference__wrapped_model_5123
input_1.
*actor_dense_matmul_readvariableop_resource/
+actor_dense_biasadd_readvariableop_resource0
,actor_dense_1_matmul_readvariableop_resource1
-actor_dense_1_biasadd_readvariableop_resource0
,actor_dense_2_matmul_readvariableop_resource1
-actor_dense_2_biasadd_readvariableop_resource
identity??"actor/dense/BiasAdd/ReadVariableOp?!actor/dense/MatMul/ReadVariableOp?$actor/dense_1/BiasAdd/ReadVariableOp?#actor/dense_1/MatMul/ReadVariableOp?$actor/dense_2/BiasAdd/ReadVariableOp?#actor/dense_2/MatMul/ReadVariableOp?
!actor/dense/MatMul/ReadVariableOpReadVariableOp*actor_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!actor/dense/MatMul/ReadVariableOp?
actor/dense/MatMulMatMulinput_1)actor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/dense/MatMul?
"actor/dense/BiasAdd/ReadVariableOpReadVariableOp+actor_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"actor/dense/BiasAdd/ReadVariableOp?
actor/dense/BiasAddBiasAddactor/dense/MatMul:product:0*actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/dense/BiasAdd|
actor/dense/ReluReluactor/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
actor/dense/Relu?
#actor/dense_1/MatMul/ReadVariableOpReadVariableOp,actor_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#actor/dense_1/MatMul/ReadVariableOp?
actor/dense_1/MatMulMatMulinput_1+actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/dense_1/MatMul?
$actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$actor/dense_1/BiasAdd/ReadVariableOp?
actor/dense_1/BiasAddBiasAddactor/dense_1/MatMul:product:0,actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/dense_1/BiasAdd?
actor/dense_1/ReluReluactor/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
actor/dense_1/Relu?
#actor/dense_2/MatMul/ReadVariableOpReadVariableOp,actor_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#actor/dense_2/MatMul/ReadVariableOp?
actor/dense_2/MatMulMatMul actor/dense_1/Relu:activations:0+actor/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/dense_2/MatMul?
$actor/dense_2/BiasAdd/ReadVariableOpReadVariableOp-actor_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$actor/dense_2/BiasAdd/ReadVariableOp?
actor/dense_2/BiasAddBiasAddactor/dense_2/MatMul:product:0,actor/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/dense_2/BiasAdd?
actor/dense_2/ReluReluactor/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
actor/dense_2/Relu?
IdentityIdentity actor/dense_2/Relu:activations:0#^actor/dense/BiasAdd/ReadVariableOp"^actor/dense/MatMul/ReadVariableOp%^actor/dense_1/BiasAdd/ReadVariableOp$^actor/dense_1/MatMul/ReadVariableOp%^actor/dense_2/BiasAdd/ReadVariableOp$^actor/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2H
"actor/dense/BiasAdd/ReadVariableOp"actor/dense/BiasAdd/ReadVariableOp2F
!actor/dense/MatMul/ReadVariableOp!actor/dense/MatMul/ReadVariableOp2L
$actor/dense_1/BiasAdd/ReadVariableOp$actor/dense_1/BiasAdd/ReadVariableOp2J
#actor/dense_1/MatMul/ReadVariableOp#actor/dense_1/MatMul/ReadVariableOp2L
$actor/dense_2/BiasAdd/ReadVariableOp$actor/dense_2/BiasAdd/ReadVariableOp2J
#actor/dense_2/MatMul/ReadVariableOp#actor/dense_2/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_5192

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_dense_1_layer_call_fn_5286

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_51652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
 __inference__traced_restore_5375
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B(layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB'actor/kernel/.ATTRIBUTES/VARIABLE_VALUEB%actor/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_5297

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_5246
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *(
f#R!
__inference__wrapped_model_51232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
y
$__inference_dense_layer_call_fn_5266

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_51382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_actor_layer_call_and_return_conditional_losses_5209
input_1

dense_5149

dense_5151
dense_1_5176
dense_1_5178
dense_2_5203
dense_2_5205
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1
dense_5149
dense_5151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_51382
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1_5176dense_1_5178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_51652!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_5203dense_2_5205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_51922!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_5138

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_dense_2_layer_call_fn_5306

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_51922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_5257

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_5347
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B(layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB'actor/kernel/.ATTRIBUTES/VARIABLE_VALUEB%actor/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: ::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_5277

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_actor_layer_call_fn_5227
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*J
config_proto:8

CPU

GPU

XLA_CPU

XLA_GPU2*0J 8? *H
fCRA
?__inference_actor_layer_call_and_return_conditional_losses_52092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?N
?

layer1

layer2
	actor
	variables
trainable_variables
regularization_losses
	keras_api

signatures
/__call__
*0&call_and_return_all_conditional_losses
1_default_save_signature"?
_tf_keras_model?{"class_name": "Actor", "name": "actor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Actor"}}
?

	kernel

bias
	variables
trainable_variables
regularization_losses
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
6__call__
*7&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
J
	0

1
2
3
4
5"
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
regularization_losses

layers
metrics
layer_metrics
non_trainable_variables
layer_regularization_losses
/__call__
1_default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
,
8serving_default"
signature_map
:2dense/kernel
:2
dense/bias
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
regularization_losses

 layers
!metrics
"layer_metrics
#non_trainable_variables
$layer_regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
regularization_losses

%layers
&metrics
'layer_metrics
(non_trainable_variables
)layer_regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
regularization_losses

*layers
+metrics
,layer_metrics
-non_trainable_variables
.layer_regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
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
?2?
$__inference_actor_layer_call_fn_5227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
?__inference_actor_layer_call_and_return_conditional_losses_5209?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
__inference__wrapped_model_5123?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
$__inference_dense_layer_call_fn_5266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_5257?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_5286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_5277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_5306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_5297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_5246input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_5123o	
0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
?__inference_actor_layer_call_and_return_conditional_losses_5209a	
0?-
&?#
!?
input_1?????????
? "%?"
?
0?????????
? |
$__inference_actor_layer_call_fn_5227T	
0?-
&?#
!?
input_1?????????
? "???????????
A__inference_dense_1_layer_call_and_return_conditional_losses_5277\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_dense_1_layer_call_fn_5286O/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense_2_layer_call_and_return_conditional_losses_5297\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_dense_2_layer_call_fn_5306O/?,
%?"
 ?
inputs?????????
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_5257\	
/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? w
$__inference_dense_layer_call_fn_5266O	
/?,
%?"
 ?
inputs?????????
? "???????????
"__inference_signature_wrapper_5246z	
;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????