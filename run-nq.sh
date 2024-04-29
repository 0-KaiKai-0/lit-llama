DATASET=narrativeqa16
OUTPUT_DIR=out/lora/$DATASET/epoch3-r16-lr1e-4-targets1-warmup50

mkdir -p $OUTPUT_DIR

cp finetune/lora-nq.py $OUTPUT_DIR/lora.py
cp run.sh $OUTPUT_DIR/run.sh

NCCL_P2P_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6,7 python finetune/lora-nq.py \
--out_dir $OUTPUT_DIR \
--data_dir data/$DATASET \
| tee -a $OUTPUT_DIR/train.log