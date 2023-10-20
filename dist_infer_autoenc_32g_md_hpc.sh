#!/bin/bash
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH -J 3d-infer-autoenc
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=10:00:00

module load anaconda3/23.1.0
conda activate zj-py39

cd /scratch/zjiang/generative_ai/3d_ldm_cnh

echo 'Start------------'
date
echo '-----------------'

NUM_GPUS_PER_NODE=2
gpu_memory=32
NUM_NODES=1

export NCCL_P2P_LEVEL=NVL
echo "CUDA:${CUDA_VISIBLE_DEVICES}"

torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --master_addr=localhost --master_port=1234 \
    inference_autoencoder.py -c ./config/config_infer_32g_ldmor.json -e ./config/environment_ngc_32g_new.json \
    -g ${NUM_GPUS_PER_NODE} --amp

echo 'End--------------'
date
echo '-----------------'

