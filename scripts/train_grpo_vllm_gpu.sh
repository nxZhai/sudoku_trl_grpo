export CUDA_VISIBLE_DEVICES=1,4,5,6

accelerate launch \
    --num_processes 4 \
    --config_file ./configs/deepspeed_zero3.yaml \
    train_grpo.py \
    --config ./configs/grpo_qwen2.5-3b-it_lora-gpu.yaml
