þ
Ø
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
Á
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
executor_typestring ¨
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ß

Adam/d3qn/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/d3qn/dense_3/bias/v

,Adam/d3qn/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_3/bias/v*
_output_shapes
:*
dtype0

Adam/d3qn/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_nameAdam/d3qn/dense_3/kernel/v

.Adam/d3qn/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_3/kernel/v*
_output_shapes

:@*
dtype0

Adam/d3qn/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/d3qn/dense_2/bias/v

,Adam/d3qn/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/d3qn/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_nameAdam/d3qn/dense_2/kernel/v

.Adam/d3qn/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_2/kernel/v*
_output_shapes

:@*
dtype0

Adam/d3qn/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/d3qn/dense_1/bias/v

,Adam/d3qn/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_1/bias/v*
_output_shapes
:@*
dtype0

Adam/d3qn/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*+
shared_nameAdam/d3qn/dense_1/kernel/v

.Adam/d3qn/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_1/kernel/v*
_output_shapes

:@@*
dtype0

Adam/d3qn/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/d3qn/dense/bias/v
}
*Adam/d3qn/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense/bias/v*
_output_shapes
:@*
dtype0

Adam/d3qn/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_nameAdam/d3qn/dense/kernel/v

,Adam/d3qn/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense/kernel/v*
_output_shapes

:@*
dtype0

Adam/d3qn/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/d3qn/dense_3/bias/m

,Adam/d3qn/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_3/bias/m*
_output_shapes
:*
dtype0

Adam/d3qn/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_nameAdam/d3qn/dense_3/kernel/m

.Adam/d3qn/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_3/kernel/m*
_output_shapes

:@*
dtype0

Adam/d3qn/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/d3qn/dense_2/bias/m

,Adam/d3qn/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/d3qn/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_nameAdam/d3qn/dense_2/kernel/m

.Adam/d3qn/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_2/kernel/m*
_output_shapes

:@*
dtype0

Adam/d3qn/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/d3qn/dense_1/bias/m

,Adam/d3qn/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_1/bias/m*
_output_shapes
:@*
dtype0

Adam/d3qn/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*+
shared_nameAdam/d3qn/dense_1/kernel/m

.Adam/d3qn/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense_1/kernel/m*
_output_shapes

:@@*
dtype0

Adam/d3qn/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/d3qn/dense/bias/m
}
*Adam/d3qn/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense/bias/m*
_output_shapes
:@*
dtype0

Adam/d3qn/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_nameAdam/d3qn/dense/kernel/m

,Adam/d3qn/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/d3qn/dense/kernel/m*
_output_shapes

:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
z
d3qn/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_named3qn/dense_3/bias
s
%d3qn/dense_3/bias/Read/ReadVariableOpReadVariableOpd3qn/dense_3/bias*
_output_shapes
:*
dtype0

d3qn/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_named3qn/dense_3/kernel
{
'd3qn/dense_3/kernel/Read/ReadVariableOpReadVariableOpd3qn/dense_3/kernel*
_output_shapes

:@*
dtype0
z
d3qn/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_named3qn/dense_2/bias
s
%d3qn/dense_2/bias/Read/ReadVariableOpReadVariableOpd3qn/dense_2/bias*
_output_shapes
:*
dtype0

d3qn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_named3qn/dense_2/kernel
{
'd3qn/dense_2/kernel/Read/ReadVariableOpReadVariableOpd3qn/dense_2/kernel*
_output_shapes

:@*
dtype0
z
d3qn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_named3qn/dense_1/bias
s
%d3qn/dense_1/bias/Read/ReadVariableOpReadVariableOpd3qn/dense_1/bias*
_output_shapes
:@*
dtype0

d3qn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*$
shared_named3qn/dense_1/kernel
{
'd3qn/dense_1/kernel/Read/ReadVariableOpReadVariableOpd3qn/dense_1/kernel*
_output_shapes

:@@*
dtype0
v
d3qn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_named3qn/dense/bias
o
#d3qn/dense/bias/Read/ReadVariableOpReadVariableOpd3qn/dense/bias*
_output_shapes
:@*
dtype0
~
d3qn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_named3qn/dense/kernel
w
%d3qn/dense/kernel/Read/ReadVariableOpReadVariableOpd3qn/dense/kernel*
_output_shapes

:@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
÷
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1d3qn/dense/kerneld3qn/dense/biasd3qn/dense_1/kerneld3qn/dense_1/biasd3qn/dense_2/kerneld3qn/dense_2/biasd3qn/dense_3/kerneld3qn/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_52044224

NoOpNoOp
¬8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ç7
valueÝ7BÚ7 BÓ7
ò
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2

V
A
	optimizer

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*

0
1* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
 trace_3* 
6
!trace_0
"trace_1
#trace_2
$trace_3* 
* 
¦
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias*
¦
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias*
¦
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias*
¦
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

kernel
bias*
Á
=iter

>beta_1

?beta_2
	@decaymompmqmrmsmtmumvvwvxvyvzv{v|v}v~*

Aserving_default* 
QK
VARIABLE_VALUEd3qn/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEd3qn/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEd3qn/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEd3qn/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEd3qn/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEd3qn/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEd3qn/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEd3qn/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*

Btrace_0* 

Ctrace_0* 
* 
 
0
	1

2
3*

D0
E1
F2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
	
0* 

Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Ltrace_0* 

Mtrace_0* 

0
1*

0
1*
	
0* 

Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 

0
1*

0
1*
* 

Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 

0
1*

0
1*
* 

\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

atrace_0* 

btrace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
8
c	variables
d	keras_api
	etotal
	fcount*
8
g	variables
h	keras_api
	itotal
	jcount*
8
k	variables
l	keras_api
	mtotal
	ncount*
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

e0
f1*

c	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

i0
j1*

g	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

m0
n1*

k	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/d3qn/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/d3qn/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/d3qn/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/d3qn/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/d3qn/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/d3qn/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/d3qn/dense_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/d3qn/dense_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/d3qn/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/d3qn/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/d3qn/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/d3qn/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/d3qn/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/d3qn/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/d3qn/dense_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/d3qn/dense_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%d3qn/dense/kernel/Read/ReadVariableOp#d3qn/dense/bias/Read/ReadVariableOp'd3qn/dense_1/kernel/Read/ReadVariableOp%d3qn/dense_1/bias/Read/ReadVariableOp'd3qn/dense_2/kernel/Read/ReadVariableOp%d3qn/dense_2/bias/Read/ReadVariableOp'd3qn/dense_3/kernel/Read/ReadVariableOp%d3qn/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/d3qn/dense/kernel/m/Read/ReadVariableOp*Adam/d3qn/dense/bias/m/Read/ReadVariableOp.Adam/d3qn/dense_1/kernel/m/Read/ReadVariableOp,Adam/d3qn/dense_1/bias/m/Read/ReadVariableOp.Adam/d3qn/dense_2/kernel/m/Read/ReadVariableOp,Adam/d3qn/dense_2/bias/m/Read/ReadVariableOp.Adam/d3qn/dense_3/kernel/m/Read/ReadVariableOp,Adam/d3qn/dense_3/bias/m/Read/ReadVariableOp,Adam/d3qn/dense/kernel/v/Read/ReadVariableOp*Adam/d3qn/dense/bias/v/Read/ReadVariableOp.Adam/d3qn/dense_1/kernel/v/Read/ReadVariableOp,Adam/d3qn/dense_1/bias/v/Read/ReadVariableOp.Adam/d3qn/dense_2/kernel/v/Read/ReadVariableOp,Adam/d3qn/dense_2/bias/v/Read/ReadVariableOp.Adam/d3qn/dense_3/kernel/v/Read/ReadVariableOp,Adam/d3qn/dense_3/bias/v/Read/ReadVariableOpConst*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_52044594
ä
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamed3qn/dense/kerneld3qn/dense/biasd3qn/dense_1/kerneld3qn/dense_1/biasd3qn/dense_2/kerneld3qn/dense_2/biasd3qn/dense_3/kerneld3qn/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotal_2count_2total_1count_1totalcountAdam/d3qn/dense/kernel/mAdam/d3qn/dense/bias/mAdam/d3qn/dense_1/kernel/mAdam/d3qn/dense_1/bias/mAdam/d3qn/dense_2/kernel/mAdam/d3qn/dense_2/bias/mAdam/d3qn/dense_3/kernel/mAdam/d3qn/dense_3/bias/mAdam/d3qn/dense/kernel/vAdam/d3qn/dense/bias/vAdam/d3qn/dense_1/kernel/vAdam/d3qn/dense_1/bias/vAdam/d3qn/dense_2/kernel/vAdam/d3qn/dense_2/bias/vAdam/d3qn/dense_3/kernel/vAdam/d3qn/dense_3/bias/v*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_52044706ÂÊ
ª
ª
C__inference_dense_layer_call_and_return_conditional_losses_52044406

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú	
»
__inference_loss_fn_1_52044382P
>d3qn_dense_1_kernel_regularizer_l2loss_readvariableop_resource:@@
identity¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp´
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>d3qn_dense_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'd3qn/dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp
ú

Ç
'__inference_d3qn_layer_call_fn_52044113
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
identity

identity_1¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_d3qn_layer_call_and_return_conditional_losses_52044069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Â
®
E__inference_dense_1_layer_call_and_return_conditional_losses_52044430

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï2

B__inference_d3qn_layer_call_and_return_conditional_losses_52044313	
state6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:
identity

identity_1¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0t
dense/MatMulMatMulstate#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_3/MatMulMatMuldense_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMeandense_3/BiasAdd:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(e
subSubdense_3/BiasAdd:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
addAddV2dense_2/BiasAdd:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi

Identity_1Identitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
Ç

*__inference_dense_2_layer_call_fn_52044439

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_52043910o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ß'
ò
B__inference_d3qn_layer_call_and_return_conditional_losses_52044187
input_1 
dense_52044153:@
dense_52044155:@"
dense_1_52044158:@@
dense_1_52044160:@"
dense_2_52044163:@
dense_2_52044165:"
dense_3_52044168:@
dense_3_52044170:
identity

identity_1¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallî
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_52044153dense_52044155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_52043873
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_52044158dense_1_52044160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_52043894
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_52044163dense_2_52044165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_52043910
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_52044168dense_3_52044170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_52043926X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMean(dense_3/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(u
subSub(dense_3/StatefulPartitionedCall:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
addAddV2(dense_2/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_52044153*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_52044158*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ç

*__inference_dense_3_layer_call_fn_52044458

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_52043926o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ù'
ð
B__inference_d3qn_layer_call_and_return_conditional_losses_52043946	
state 
dense_52043874:@
dense_52043876:@"
dense_1_52043895:@@
dense_1_52043897:@"
dense_2_52043911:@
dense_2_52043913:"
dense_3_52043927:@
dense_3_52043929:
identity

identity_1¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallì
dense/StatefulPartitionedCallStatefulPartitionedCallstatedense_52043874dense_52043876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_52043873
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_52043895dense_1_52043897*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_52043894
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_52043911dense_2_52043913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_52043910
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_52043927dense_3_52043929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_52043926X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMean(dense_3/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(u
subSub(dense_3/StatefulPartitionedCall:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
addAddV2(dense_2/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_52043874*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_52043895*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
Ç

*__inference_dense_1_layer_call_fn_52044415

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_52043894o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È	
ö
E__inference_dense_2_layer_call_and_return_conditional_losses_52043910

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª
ª
C__inference_dense_layer_call_and_return_conditional_losses_52043873

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

(__inference_dense_layer_call_fn_52044391

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_52043873o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

Å
'__inference_d3qn_layer_call_fn_52044270	
state
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
identity

identity_1¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_d3qn_layer_call_and_return_conditional_losses_52044069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
ï2

B__inference_d3qn_layer_call_and_return_conditional_losses_52044356	
state6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:
identity

identity_1¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0t
dense/MatMulMatMulstate#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_3/MatMulMatMuldense_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMeandense_3/BiasAdd:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(e
subSubdense_3/BiasAdd:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
addAddV2dense_2/BiasAdd:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi

Identity_1Identitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
ß'
ò
B__inference_d3qn_layer_call_and_return_conditional_losses_52044150
input_1 
dense_52044116:@
dense_52044118:@"
dense_1_52044121:@@
dense_1_52044123:@"
dense_2_52044126:@
dense_2_52044128:"
dense_3_52044131:@
dense_3_52044133:
identity

identity_1¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallî
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_52044116dense_52044118*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_52043873
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_52044121dense_1_52044123*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_52043894
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_52044126dense_2_52044128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_52043910
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_52044131dense_3_52044133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_52043926X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMean(dense_3/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(u
subSub(dense_3/StatefulPartitionedCall:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
addAddV2(dense_2/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_52044116*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_52044121*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
È	
ö
E__inference_dense_3_layer_call_and_return_conditional_losses_52044468

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È	
ö
E__inference_dense_3_layer_call_and_return_conditional_losses_52043926

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ù'
ð
B__inference_d3qn_layer_call_and_return_conditional_losses_52044069	
state 
dense_52044035:@
dense_52044037:@"
dense_1_52044040:@@
dense_1_52044042:@"
dense_2_52044045:@
dense_2_52044047:"
dense_3_52044050:@
dense_3_52044052:
identity

identity_1¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallì
dense/StatefulPartitionedCallStatefulPartitionedCallstatedense_52044035dense_52044037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_52043873
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_52044040dense_1_52044042*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_52043894
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_52044045dense_2_52044047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_52043910
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_52044050dense_3_52044052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_52043926X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
MeanMean(dense_3/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(u
subSub(dense_3/StatefulPartitionedCall:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
addAddV2(dense_2/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_52044035*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_52044040*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
Â
®
E__inference_dense_1_layer_call_and_return_conditional_losses_52043894

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
&d3qn/dense_1/kernel/Regularizer/L2LossL2Loss=d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%d3qn/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¬
#d3qn/dense_1/kernel/Regularizer/mulMul.d3qn/dense_1/kernel/Regularizer/mul/x:output:0/d3qn/dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp5d3qn/dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú

Æ
&__inference_signature_wrapper_52044224
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_52043851o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
û
Ò
$__inference__traced_restore_52044706
file_prefix4
"assignvariableop_d3qn_dense_kernel:@0
"assignvariableop_1_d3qn_dense_bias:@8
&assignvariableop_2_d3qn_dense_1_kernel:@@2
$assignvariableop_3_d3qn_dense_1_bias:@8
&assignvariableop_4_d3qn_dense_2_kernel:@2
$assignvariableop_5_d3qn_dense_2_bias:8
&assignvariableop_6_d3qn_dense_3_kernel:@2
$assignvariableop_7_d3qn_dense_3_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: %
assignvariableop_12_total_2: %
assignvariableop_13_count_2: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: >
,assignvariableop_18_adam_d3qn_dense_kernel_m:@8
*assignvariableop_19_adam_d3qn_dense_bias_m:@@
.assignvariableop_20_adam_d3qn_dense_1_kernel_m:@@:
,assignvariableop_21_adam_d3qn_dense_1_bias_m:@@
.assignvariableop_22_adam_d3qn_dense_2_kernel_m:@:
,assignvariableop_23_adam_d3qn_dense_2_bias_m:@
.assignvariableop_24_adam_d3qn_dense_3_kernel_m:@:
,assignvariableop_25_adam_d3qn_dense_3_bias_m:>
,assignvariableop_26_adam_d3qn_dense_kernel_v:@8
*assignvariableop_27_adam_d3qn_dense_bias_v:@@
.assignvariableop_28_adam_d3qn_dense_1_kernel_v:@@:
,assignvariableop_29_adam_d3qn_dense_1_bias_v:@@
.assignvariableop_30_adam_d3qn_dense_2_kernel_v:@:
,assignvariableop_31_adam_d3qn_dense_2_bias_v:@
.assignvariableop_32_adam_d3qn_dense_3_kernel_v:@:
,assignvariableop_33_adam_d3qn_dense_3_bias_v:
identity_35¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*¨
valueB#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¶
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ð
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¢
_output_shapes
:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_d3qn_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_d3qn_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp&assignvariableop_2_d3qn_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_d3qn_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp&assignvariableop_4_d3qn_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_d3qn_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_d3qn_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_d3qn_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_d3qn_dense_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_d3qn_dense_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_d3qn_dense_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_d3qn_dense_1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_d3qn_dense_2_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_d3qn_dense_2_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_d3qn_dense_3_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_d3qn_dense_3_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_d3qn_dense_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_d3qn_dense_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_d3qn_dense_1_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_d3qn_dense_1_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_d3qn_dense_2_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_d3qn_dense_2_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp.assignvariableop_32_adam_d3qn_dense_3_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_d3qn_dense_3_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 »
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: ¨
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¾	
·
__inference_loss_fn_0_52044373N
<d3qn_dense_kernel_regularizer_l2loss_readvariableop_resource:@
identity¢3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp°
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<d3qn_dense_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@*
dtype0
$d3qn/dense/kernel/Regularizer/L2LossL2Loss;d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#d3qn/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;¦
!d3qn/dense/kernel/Regularizer/mulMul,d3qn/dense/kernel/Regularizer/mul/x:output:0-d3qn/dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%d3qn/dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp3d3qn/dense/kernel/Regularizer/L2Loss/ReadVariableOp
ÕE
³
!__inference__traced_save_52044594
file_prefix0
,savev2_d3qn_dense_kernel_read_readvariableop.
*savev2_d3qn_dense_bias_read_readvariableop2
.savev2_d3qn_dense_1_kernel_read_readvariableop0
,savev2_d3qn_dense_1_bias_read_readvariableop2
.savev2_d3qn_dense_2_kernel_read_readvariableop0
,savev2_d3qn_dense_2_bias_read_readvariableop2
.savev2_d3qn_dense_3_kernel_read_readvariableop0
,savev2_d3qn_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_d3qn_dense_kernel_m_read_readvariableop5
1savev2_adam_d3qn_dense_bias_m_read_readvariableop9
5savev2_adam_d3qn_dense_1_kernel_m_read_readvariableop7
3savev2_adam_d3qn_dense_1_bias_m_read_readvariableop9
5savev2_adam_d3qn_dense_2_kernel_m_read_readvariableop7
3savev2_adam_d3qn_dense_2_bias_m_read_readvariableop9
5savev2_adam_d3qn_dense_3_kernel_m_read_readvariableop7
3savev2_adam_d3qn_dense_3_bias_m_read_readvariableop7
3savev2_adam_d3qn_dense_kernel_v_read_readvariableop5
1savev2_adam_d3qn_dense_bias_v_read_readvariableop9
5savev2_adam_d3qn_dense_1_kernel_v_read_readvariableop7
3savev2_adam_d3qn_dense_1_bias_v_read_readvariableop9
5savev2_adam_d3qn_dense_2_kernel_v_read_readvariableop7
3savev2_adam_d3qn_dense_2_bias_v_read_readvariableop9
5savev2_adam_d3qn_dense_3_kernel_v_read_readvariableop7
3savev2_adam_d3qn_dense_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ÿ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*¨
valueB#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH³
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_d3qn_dense_kernel_read_readvariableop*savev2_d3qn_dense_bias_read_readvariableop.savev2_d3qn_dense_1_kernel_read_readvariableop,savev2_d3qn_dense_1_bias_read_readvariableop.savev2_d3qn_dense_2_kernel_read_readvariableop,savev2_d3qn_dense_2_bias_read_readvariableop.savev2_d3qn_dense_3_kernel_read_readvariableop,savev2_d3qn_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_d3qn_dense_kernel_m_read_readvariableop1savev2_adam_d3qn_dense_bias_m_read_readvariableop5savev2_adam_d3qn_dense_1_kernel_m_read_readvariableop3savev2_adam_d3qn_dense_1_bias_m_read_readvariableop5savev2_adam_d3qn_dense_2_kernel_m_read_readvariableop3savev2_adam_d3qn_dense_2_bias_m_read_readvariableop5savev2_adam_d3qn_dense_3_kernel_m_read_readvariableop3savev2_adam_d3qn_dense_3_bias_m_read_readvariableop3savev2_adam_d3qn_dense_kernel_v_read_readvariableop1savev2_adam_d3qn_dense_bias_v_read_readvariableop5savev2_adam_d3qn_dense_1_kernel_v_read_readvariableop3savev2_adam_d3qn_dense_1_bias_v_read_readvariableop5savev2_adam_d3qn_dense_2_kernel_v_read_readvariableop3savev2_adam_d3qn_dense_2_bias_v_read_readvariableop5savev2_adam_d3qn_dense_3_kernel_v_read_readvariableop3savev2_adam_d3qn_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*í
_input_shapesÛ
Ø: :@:@:@@:@:@::@:: : : : : : : : : : :@:@:@@:@:@::@::@:@:@@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@:  

_output_shapes
::$! 

_output_shapes

:@: "

_output_shapes
::#

_output_shapes
: 
ú

Ç
'__inference_d3qn_layer_call_fn_52043967
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
identity

identity_1¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_d3qn_layer_call_and_return_conditional_losses_52043946o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ô

Å
'__inference_d3qn_layer_call_fn_52044247	
state
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
identity

identity_1¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_d3qn_layer_call_and_return_conditional_losses_52043946o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
È	
ö
E__inference_dense_2_layer_call_and_return_conditional_losses_52044449

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ)
ã
#__inference__wrapped_model_52043851
input_1;
)d3qn_dense_matmul_readvariableop_resource:@8
*d3qn_dense_biasadd_readvariableop_resource:@=
+d3qn_dense_1_matmul_readvariableop_resource:@@:
,d3qn_dense_1_biasadd_readvariableop_resource:@=
+d3qn_dense_2_matmul_readvariableop_resource:@:
,d3qn_dense_2_biasadd_readvariableop_resource:=
+d3qn_dense_3_matmul_readvariableop_resource:@:
,d3qn_dense_3_biasadd_readvariableop_resource:
identity

identity_1¢!d3qn/dense/BiasAdd/ReadVariableOp¢ d3qn/dense/MatMul/ReadVariableOp¢#d3qn/dense_1/BiasAdd/ReadVariableOp¢"d3qn/dense_1/MatMul/ReadVariableOp¢#d3qn/dense_2/BiasAdd/ReadVariableOp¢"d3qn/dense_2/MatMul/ReadVariableOp¢#d3qn/dense_3/BiasAdd/ReadVariableOp¢"d3qn/dense_3/MatMul/ReadVariableOp
 d3qn/dense/MatMul/ReadVariableOpReadVariableOp)d3qn_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
d3qn/dense/MatMulMatMulinput_1(d3qn/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!d3qn/dense/BiasAdd/ReadVariableOpReadVariableOp*d3qn_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
d3qn/dense/BiasAddBiasAddd3qn/dense/MatMul:product:0)d3qn/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
d3qn/dense/ReluRelud3qn/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"d3qn/dense_1/MatMul/ReadVariableOpReadVariableOp+d3qn_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
d3qn/dense_1/MatMulMatMuld3qn/dense/Relu:activations:0*d3qn/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#d3qn/dense_1/BiasAdd/ReadVariableOpReadVariableOp,d3qn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
d3qn/dense_1/BiasAddBiasAddd3qn/dense_1/MatMul:product:0+d3qn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
d3qn/dense_1/ReluRelud3qn/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"d3qn/dense_2/MatMul/ReadVariableOpReadVariableOp+d3qn_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
d3qn/dense_2/MatMulMatMuld3qn/dense_1/Relu:activations:0*d3qn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#d3qn/dense_2/BiasAdd/ReadVariableOpReadVariableOp,d3qn_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
d3qn/dense_2/BiasAddBiasAddd3qn/dense_2/MatMul:product:0+d3qn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"d3qn/dense_3/MatMul/ReadVariableOpReadVariableOp+d3qn_dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
d3qn/dense_3/MatMulMatMuld3qn/dense_1/Relu:activations:0*d3qn/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#d3qn/dense_3/BiasAdd/ReadVariableOpReadVariableOp,d3qn_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
d3qn/dense_3/BiasAddBiasAddd3qn/dense_3/MatMul:product:0+d3qn/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
d3qn/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
	d3qn/MeanMeand3qn/dense_3/BiasAdd:output:0$d3qn/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(t
d3qn/subSubd3qn/dense_3/BiasAdd:output:0d3qn/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
d3qn/addAddV2d3qn/dense_2/BiasAdd:output:0d3qn/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityd3qn/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn

Identity_1Identityd3qn/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
NoOpNoOp"^d3qn/dense/BiasAdd/ReadVariableOp!^d3qn/dense/MatMul/ReadVariableOp$^d3qn/dense_1/BiasAdd/ReadVariableOp#^d3qn/dense_1/MatMul/ReadVariableOp$^d3qn/dense_2/BiasAdd/ReadVariableOp#^d3qn/dense_2/MatMul/ReadVariableOp$^d3qn/dense_3/BiasAdd/ReadVariableOp#^d3qn/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2F
!d3qn/dense/BiasAdd/ReadVariableOp!d3qn/dense/BiasAdd/ReadVariableOp2D
 d3qn/dense/MatMul/ReadVariableOp d3qn/dense/MatMul/ReadVariableOp2J
#d3qn/dense_1/BiasAdd/ReadVariableOp#d3qn/dense_1/BiasAdd/ReadVariableOp2H
"d3qn/dense_1/MatMul/ReadVariableOp"d3qn/dense_1/MatMul/ReadVariableOp2J
#d3qn/dense_2/BiasAdd/ReadVariableOp#d3qn/dense_2/BiasAdd/ReadVariableOp2H
"d3qn/dense_2/MatMul/ReadVariableOp"d3qn/dense_2/MatMul/ReadVariableOp2J
#d3qn/dense_3/BiasAdd/ReadVariableOp#d3qn/dense_3/BiasAdd/ReadVariableOp2H
"d3qn/dense_3/MatMul/ReadVariableOp"d3qn/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*é
serving_defaultÕ
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Þ

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2

V
A
	optimizer

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö
trace_0
trace_1
trace_2
 trace_32ë
'__inference_d3qn_layer_call_fn_52043967
'__inference_d3qn_layer_call_fn_52044247
'__inference_d3qn_layer_call_fn_52044270
'__inference_d3qn_layer_call_fn_52044113Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ztrace_0ztrace_1ztrace_2z trace_3
Â
!trace_0
"trace_1
#trace_2
$trace_32×
B__inference_d3qn_layer_call_and_return_conditional_losses_52044313
B__inference_d3qn_layer_call_and_return_conditional_losses_52044356
B__inference_d3qn_layer_call_and_return_conditional_losses_52044150
B__inference_d3qn_layer_call_and_return_conditional_losses_52044187Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 z!trace_0z"trace_1z#trace_2z$trace_3
ÎBË
#__inference__wrapped_model_52043851input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
»
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ð
=iter

>beta_1

?beta_2
	@decaymompmqmrmsmtmumvvwvxvyvzv{v|v}v~"
	optimizer
,
Aserving_default"
signature_map
#:!@2d3qn/dense/kernel
:@2d3qn/dense/bias
%:#@@2d3qn/dense_1/kernel
:@2d3qn/dense_1/bias
%:#@2d3qn/dense_2/kernel
:2d3qn/dense_2/bias
%:#@2d3qn/dense_3/kernel
:2d3qn/dense_3/bias
Ï
Btrace_02²
__inference_loss_fn_0_52044373
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ zBtrace_0
Ï
Ctrace_02²
__inference_loss_fn_1_52044382
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ zCtrace_0
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
'__inference_d3qn_layer_call_fn_52043967input_1"Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
üBù
'__inference_d3qn_layer_call_fn_52044247state"Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
üBù
'__inference_d3qn_layer_call_fn_52044270state"Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
þBû
'__inference_d3qn_layer_call_fn_52044113input_1"Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
B__inference_d3qn_layer_call_and_return_conditional_losses_52044313state"Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
B__inference_d3qn_layer_call_and_return_conditional_losses_52044356state"Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
B__inference_d3qn_layer_call_and_return_conditional_losses_52044150input_1"Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
B__inference_d3qn_layer_call_and_return_conditional_losses_52044187input_1"Ä
»²·
FullArgSpec
args
jself
jstate
varargs
 
varkwjkwargs
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ì
Ltrace_02Ï
(__inference_dense_layer_call_fn_52044391¢
²
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
annotationsª *
 zLtrace_0

Mtrace_02ê
C__inference_dense_layer_call_and_return_conditional_losses_52044406¢
²
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
annotationsª *
 zMtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
î
Strace_02Ñ
*__inference_dense_1_layer_call_fn_52044415¢
²
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
annotationsª *
 zStrace_0

Ttrace_02ì
E__inference_dense_1_layer_call_and_return_conditional_losses_52044430¢
²
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
annotationsª *
 zTtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
î
Ztrace_02Ñ
*__inference_dense_2_layer_call_fn_52044439¢
²
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
annotationsª *
 zZtrace_0

[trace_02ì
E__inference_dense_2_layer_call_and_return_conditional_losses_52044449¢
²
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
annotationsª *
 z[trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
î
atrace_02Ñ
*__inference_dense_3_layer_call_fn_52044458¢
²
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
annotationsª *
 zatrace_0

btrace_02ì
E__inference_dense_3_layer_call_and_return_conditional_losses_52044468¢
²
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
annotationsª *
 zbtrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
ÍBÊ
&__inference_signature_wrapper_52044224input_1"
²
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
annotationsª *
 
µB²
__inference_loss_fn_0_52044373"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µB²
__inference_loss_fn_1_52044382"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
N
c	variables
d	keras_api
	etotal
	fcount"
_tf_keras_metric
N
g	variables
h	keras_api
	itotal
	jcount"
_tf_keras_metric
N
k	variables
l	keras_api
	mtotal
	ncount"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_layer_call_fn_52044391inputs"¢
²
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
annotationsª *
 
÷Bô
C__inference_dense_layer_call_and_return_conditional_losses_52044406inputs"¢
²
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_1_layer_call_fn_52044415inputs"¢
²
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
annotationsª *
 
ùBö
E__inference_dense_1_layer_call_and_return_conditional_losses_52044430inputs"¢
²
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
annotationsª *
 
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
ÞBÛ
*__inference_dense_2_layer_call_fn_52044439inputs"¢
²
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
annotationsª *
 
ùBö
E__inference_dense_2_layer_call_and_return_conditional_losses_52044449inputs"¢
²
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
annotationsª *
 
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
ÞBÛ
*__inference_dense_3_layer_call_fn_52044458inputs"¢
²
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
annotationsª *
 
ùBö
E__inference_dense_3_layer_call_and_return_conditional_losses_52044468inputs"¢
²
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
annotationsª *
 
.
e0
f1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
:  (2total
:  (2count
.
m0
n1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
:  (2total
:  (2count
(:&@2Adam/d3qn/dense/kernel/m
": @2Adam/d3qn/dense/bias/m
*:(@@2Adam/d3qn/dense_1/kernel/m
$:"@2Adam/d3qn/dense_1/bias/m
*:(@2Adam/d3qn/dense_2/kernel/m
$:"2Adam/d3qn/dense_2/bias/m
*:(@2Adam/d3qn/dense_3/kernel/m
$:"2Adam/d3qn/dense_3/bias/m
(:&@2Adam/d3qn/dense/kernel/v
": @2Adam/d3qn/dense/bias/v
*:(@@2Adam/d3qn/dense_1/kernel/v
$:"@2Adam/d3qn/dense_1/bias/v
*:(@2Adam/d3qn/dense_2/kernel/v
$:"2Adam/d3qn/dense_2/bias/v
*:(@2Adam/d3qn/dense_3/kernel/v
$:"2Adam/d3qn/dense_3/bias/vÉ
#__inference__wrapped_model_52043851¡0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿà
B__inference_d3qn_layer_call_and_return_conditional_losses_52044150@¢=
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 à
B__inference_d3qn_layer_call_and_return_conditional_losses_52044187@¢=
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp"K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 Þ
B__inference_d3qn_layer_call_and_return_conditional_losses_52044313>¢;
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª

trainingp "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 Þ
B__inference_d3qn_layer_call_and_return_conditional_losses_52044356>¢;
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª

trainingp"K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ·
'__inference_d3qn_layer_call_fn_52043967@¢=
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ·
'__inference_d3qn_layer_call_fn_52044113@¢=
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp"=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿµ
'__inference_d3qn_layer_call_fn_52044247>¢;
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª

trainingp "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿµ
'__inference_d3qn_layer_call_fn_52044270>¢;
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª

trainingp"=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_1_layer_call_and_return_conditional_losses_52044430\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
*__inference_dense_1_layer_call_fn_52044415O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¥
E__inference_dense_2_layer_call_and_return_conditional_losses_52044449\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_2_layer_call_fn_52044439O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_3_layer_call_and_return_conditional_losses_52044468\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_3_layer_call_fn_52044458O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_layer_call_and_return_conditional_losses_52044406\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_layer_call_fn_52044391O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@=
__inference_loss_fn_0_52044373¢

¢ 
ª " =
__inference_loss_fn_1_52044382¢

¢ 
ª " ×
&__inference_signature_wrapper_52044224¬;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ