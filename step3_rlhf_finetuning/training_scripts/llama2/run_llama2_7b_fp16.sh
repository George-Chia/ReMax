#!/bin/bash

set -e
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# DeepSpeed Team
DATA_PATH="/home/zhaiyuanzhao/llm/dataset/rm-static/data"
ACTOR_MODEL_PATH="/home/zhaiyuanzhao/llm/hh/output_step1_Llama2_7b_hh/epoch1"
REWARD_MODEL_PATH="/home/zhaiyuanzhao/llm/hh/output_step2_Llama2_7b_hh/epoch0"
ACTOR_ZERO_STAGE=2
REWARD_ZERO_STAGE=3
REFERENCE_ZERO_STAGE=3
OUTPUT=$1
SEED=1234

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="./log/step3_remax-meta_llama_Llama_2_7b_hf-$TIME_STEP-$SEED"
fi

mkdir -p $OUTPUT


ACTOR_LR=1e-6


deepspeed --master_port 12346 --include localhost:6,7  main.py \
   --algo "remax" \
   --data_path $DATA_PATH \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --reward_model_name_or_path $REWARD_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${ACTOR_LR} \
   --actor_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --disable_reward_dropout \
   --num_warmup_steps 0 \
   --kl_ctl 0.05 \
   --gamma 0.99 \
   --deepspeed \
   --offload \
   --offload_reward_model \
   --offload_reference_model \
   --seed $SEED \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --reward_zero_stage $REWARD_ZERO_STAGE \
   --reference_zero_stage $REFERENCE_ZERO_STAGE \
   --enable_hybrid_engine \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_answers \
   --save_answers \
   --save_at_final \
   &> $OUTPUT/training.log
