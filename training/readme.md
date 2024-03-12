# training logbook

# Distributed example:

on a 2x RTX 3090 with deepspeed
```
accelerate launch train.py --run_name test_wood_2  --config config_test.json --lora_r 0 --base_model pythia-big --bf16 1 --gradient
_steps 4 --distributed 1 --batch_size 1
```


## DDP vs FSDP
https://vitalflux.com/distributed-llm-training-explained-with-examples/


# Experiment 1

from FinGpT "Multi-task Instruction Tuning"
<!-- CUDA_VISIBLE_DEVICES=0 -->
```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --run_name "pythia_s_mt_0" \
    --base_model pythia-small \
    --prompt_loss_weight 0 \
    --max_length 1024 \
    --config config_mt.json \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --eval_accumulation_steps 8 \
>train.log 2>&1 &;

CUDA_VISIBLE_DEVICES=0 python train.py \
    --run_name "pythia_s_mt_1" \
    --base_model pythia-small \
    --prompt_loss_weight 0.1 \
    --max_length 1024 \
    --config config_mt.json \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --eval_accumulation_steps 8 \
>train.log 2>&1 &;

CUDA_VISIBLE_DEVICES=0 python train.py \
    --run_name "pythia_s_mt_2" \
    --base_model pythia-small \
    --prompt_loss_weight 1 \
    --max_length 1024 \
    --config config_mt.json \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --eval_accumulation_steps 8 \
>train.log 2>&1 &;


CUDA_VISIBLE_DEVICES=1 python train.py \
    --run_name "mamba_s_mt_0" \
    --base_model mamba-small \
    --prompt_loss_weight 0 \
    --max_length 1024 \
    --config config_mt.json \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --eval_accumulation_steps 8 \
>train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train.py \
    --run_name "mamba_s_mt_1" \
    --base_model mamba-small \
    --prompt_loss_weight 0.1 \
    --max_length 1024 \
    --config config_mt.json \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --eval_accumulation_steps 8 \
>train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train.py \
    --run_name "mamba_s_mt_2" \
    --base_model mamba-small \
    --prompt_loss_weight 1 \
    --max_length 1024 \
    --config config_mt.json \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --eval_accumulation_steps 8 \
>train.log 2>&1 &
```

# trash
    --distributed 1 \


trying to replicate mamba_chat runs

OOM on wood
```
python train.py --run_name pythia_big_paged_ad  --config config_test.json --lora_r 0 --base_model pythia-big --gradient_steps 4 --optim paged_adamw_8bit --batch_size 1
```

```sh
python train.py --run_name dataset_warmup \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --max_length 512  \
    --config config_test.json \
    --num_epochs 1
 ```

 ```
accelerate launch train.py --run_name dataset_warmup \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --max_length 512  \
    --config config_test.json \
    --num_epochs 1 \
    --distributed 1
 ```


<!-- sinfo -o "%50N  %10c  %20m  %30G " -->
<!-- srun --gres=gpu:a6000:1 --pty bash -->

<!-- sbatch emnist_single_gpu_tutorial.sh -->
<!-- smap -->
<!-- squeue -->
<!-- sinfo -->  