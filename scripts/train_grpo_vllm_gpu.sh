export CUDA_VISIBLE_DEVICES=1,2,3,4

accelerate launch \
    --num_processes 4 \
    --config_file ./configs/deepspeed_zero3.yaml \
    train_grpo.py \
    --config ./configs/nicy_grpo_qwen2.5-7b-it_lora-gpu.yaml
