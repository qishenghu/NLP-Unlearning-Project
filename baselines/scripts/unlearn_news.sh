export CUDA_VISIBLE_DEVICES=2

CORPUS='news'

FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

# TARGET_DIR='muse-bench/MUSE-News_target'
# LLAMA_DIR='meta-llama/Llama-2-7b-hf'
TARGET_DIR='openai-community/gpt2-large'
GPT_DIR='openai-community/gpt2-large'

MAX_LEN=1024
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=2 # 8 GPUs
FT_EPOCHS=10
FT_LR='1e-5'


algo='ga'

python unlearn.py \
    --algo $algo \
    --model_dir $TARGET_DIR --tokenizer_dir $GPT_DIR \
    --data_file $FORGET --retain_data_file $RETAIN \
    --out_dir "./ckpt/$CORPUS/$algo" \
    --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE

# for algo in 'ga' 'ga_gdr' 'ga_klr' 'npo' 'npo_gdr' 'npo_klr'; do
#     python unlearn.py \
#         --algo $algo \
#         --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
#         --data_file $FORGET --retain_data_file $RETAIN \
#         --out_dir "./ckpt/$CORPUS/$algo" \
#         --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
#         --per_device_batch_size $PER_DEVICE_BATCH_SIZE
# done


# python unlearn.py \
#     --algo 'tv' \
#     --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
#     --data_file $FORGET --retain_data_file $RETAIN \
#     --out_dir "./ckpt/$CORPUS/tv" \
#     --max_len $MAX_LEN --epochs $FT_EPOCHS --lr $FT_LR \
#     --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
#     --alpha 5.0
