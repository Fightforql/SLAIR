CONFIG_PATH=${1:-tokenizer/configs/vavae_f16d32.yaml}
CSV_PATH=${2:-/openbayes/input/input0/5-task_train.csv}
OUTPUT_PATH=${3:-/openbayes/input/input0/latents_with_origin_data}
DATA_SPLIT=${4:-5-task_train}
IMAGE_SIZE=${IMAGE_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
PRECISION=${PRECISION:-bf16}

accelerate launch \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --machine_rank "$NODE_RANK" \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines "$NNODES" \
    --mixed_precision "$PRECISION" \
    extract_paired_features.py \
    --config "$CONFIG_PATH" \
    --csv_path "$CSV_PATH" \
    --output_path "$OUTPUT_PATH" \
    --data_split "$DATA_SPLIT" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"
