#!/bin/bash
cd /n/data2/hms/dbmi/kyu/lab/cl355/Bone_marrow_cytology/0codes/
module load gcc/9.2.0 cuda/11.7
source activate tf210
export LD_LIBRARY_PATH=/home/cl355/.conda/envs/tf210/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/app/cuda/11.7-gcc-9.2.0

$@