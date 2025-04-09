export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="muse-unlearning"




# "gpt2-small": "openai-community/gpt2",
# "gpt2-medium": "openai-community/gpt2-medium",
# "gpt2-large": "openai-community/gpt2-large",
# "gpt2-xl": "openai-community/gpt2-xl",
# "t5-small": "google-t5/t5-small",
# "t5-base": "google-t5/t5-base",
# "t5-large": "google-t5/t5-large",
# "t5-xl": "google/t5-v1_1-xl"

CUDA_VISIBLE_DEVICES=1 python3 model_finetune.py \
    --corpus news \
    --model_name t5-xl \
    --output_dir ./target_model \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --save_total_limit 3 \
    --save_steps 500 \
    --logging_steps 1 \
    --use_wandb # require pip install wandb
