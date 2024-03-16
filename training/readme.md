# training logbook

# Distributed example:

on a 2x RTX 3090 with deepspeed
```
accelerate launch train.py --run_name test_wood_2  --config config_test.json --lora_r 0 --base_model pythia-big --bf16 1 --gradient_steps 4 --distributed 1 --batch_size 1
```


## DDP vs FSDP
https://vitalflux.com/distributed-llm-training-explained-with-examples/


# Experiment 1

from FinGpT "Multi-task Instruction Tuning"
<!-- CUDA_VISIBLE_DEVICES=0 -->
```sh
CUDA_VISIBLE_DEVICES=0 python train.py \
--run_name "mamba_s_mt_1" \
--base_model mamba-small \
--prompt_loss_weight 0.1 \
--max_length 512 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--num_epochs 2 \
--batch_size 16 \
--eval_steps 0.02 \
--eval_accumulation_steps 8 2>&1 | tee train1.log

CUDA_VISIBLE_DEVICES=0 python train.py \
--run_name "pythia_s_mt_2" \
--base_model pythia-small \
--prompt_loss_weight 1 \
--max_length 512 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--num_epochs 2 \
--batch_size 16 \
--eval_steps 0.02 \
--eval_accumulation_steps 8 2>&1 | tee train0.log

CUDA_VISIBLE_DEVICES=0 python train.py \
--run_name "pythia_s_mt_1" \
--base_model pythia-small \
--prompt_loss_weight 0.1 \
--max_length 512 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--num_epochs 2 \
--batch_size 16 \
--eval_steps 0.02 \
--eval_accumulation_steps 8 2>&1 | tee train0.log


CUDA_VISIBLE_DEVICES=1 python train.py \
--run_name "mamba_s_mt_0" \
--base_model mamba-small \
--prompt_loss_weight 0 \
--max_length 512 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--num_epochs 2 \
--batch_size 16 \
--eval_steps 0.02 \
--eval_accumulation_steps 8 2>&1 | tee train1.log

CUDA_VISIBLE_DEVICES=1 python train.py \
--run_name "mamba_s_mt_2" \
--base_model mamba-small \
--prompt_loss_weight 1 \
--max_length 512 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--num_epochs 2 \
--batch_size 16 \
--eval_steps 0.02 \
--eval_accumulation_steps 8 2>&1 | tee train1.log

CUDA_VISIBLE_DEVICES=1 python train.py \
--run_name "pythia_s_mt_0" \
--base_model pythia-small \
--prompt_loss_weight 0 \
--max_length 512 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--num_epochs 2 \
--batch_size 16 \
--eval_steps 0.02 \
--eval_accumulation_steps 8 2>&1 | tee train0.log
```


# Experiment 2

I was able to make this run with context_length=2048 on 2x24GB 3090
```sh
accelerate launch train.py --run_name mamba_m_mt_2 --base_model mamba-medium --num_epochs 2 --eval_steps 0.05 --lora 8 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --eval_accumulation_steps 4 --gradient_steps 4 --distributed --batch_size 16
```

ZeRO stage 2 2x A6000 48GB, usage: 46.5GB and 40GB 
```sh
accelerate launch train.py --run_name mamba_m_mt_0 --base_model pythia-medium --num_epochs 2 --eval_steps 0.05 --lora 8 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --eval_accumulation_steps 4 --gradient_steps 4 --distributed --shared_dir ~/shared --batch_size 16
```

Crazy big chonker
```sh
accelerate launch train.py --run_name mamba_m_mt_0 --base_model mamba-big --num_epochs 2 --eval_steps 0.05 --lora 8 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --eval_accumulation_steps 4 --gradient_steps 4 --distributed --batch_size 32 --bf16
```

# Can I train a 3B model on a single 80GB card?

Yes! This one takes about 24h on a single H100:

## Mamba
```sh
python train.py --run_name test_big_chungus --base_model mamba-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2
```

Adding `--bf16` takes it down to ~22h
```sh
python train.py --run_name test_big_chungus --base_model mamba-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2 --bf16
```
Adding `--eval_accumulation_steps 4 --gradient_steps 4` takes it down to ~15h
```sh
python train.py --run_name test_big_chungus --base_model mamba-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2 --bf16 --eval_accumulation_steps 4 --gradient_steps 4 
```
Adding `--lora 8` takes it down to ~15h, again we see reduced memory/computation footprint
```sh
python train.py --run_name test_big_chungus --base_model mamba-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2 --bf16 --eval_accumulation_steps 4 --gradient_steps 4 --lora 8
```
Increasing the batch size to `4` yields about 8h and to `8` about 6h


## Pythia
Now going back to the transformer, this gets an OOM otherwise it was looking like somewhere around 20h
```sh
python train.py --run_name test_big_chungus_pythia --base_model pythia-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 2048 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2
```

So I go back to 512 seq_len and get also around 20h
```sh
python train.py --run_name test_big_chungus_pythia --base_model pythia-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2
```

Now what about half-precision? Bf16 takes it down to about 15 hours and fp16 about 18
```sh
python train.py --run_name test_big_chungus_pythia --base_model pythia-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2 --bf16
```

Adding grad step slashes it down to 9h
```sh
python train.py --run_name test_big_chungus_pythia --base_model pythia-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2 --bf16 --eval_accumulation_steps 4 --gradient_steps 4 
```

Assing `--lora 8` and we're down to about 6.5h, in particular I see we are using only about 23GB memory and half the compute
```sh
python train.py --run_name test_big_chungus_pythia --base_model pythia-big --num_epochs 2 --eval_steps 0.05 --prompt_loss_weight 0 --max_length 512 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --num_workers all --batch_size 2 --bf16 --eval_accumulation_steps 4 --gradient_steps 4 
```

Increasing the batch size to `4` yields about 4h 15min and to `8` about 2h 30min



# about 24h of just 

mamba_m_mt_0

pythia_m_mt_0

CUDA_VISIBLE_DEVICES=0,1,2,3


from FinGpT "Multi-task Instruction Tuning"
<!-- CUDA_VISIBLE_DEVICES=0 -->
```sh
CUDA_VISIBLE_DEVICES=0 python train.py \
--run_name "pythia_m_mt_1" \
--base_model pythia-medium \
--lora 8 \
--prompt_loss_weight 0.1 \
--max_length 1024 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--eval_accumulation_steps 8 \
--shared_dir ~/shared \
2>&1 | tee train1.log

CUDA_VISIBLE_DEVICES=0 python train.py \
--run_name "pythia_m_mt_0" \
--base_model pythia-medium \
--lora 8 \
--prompt_loss_weight 0 \
--max_length 1024 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--eval_accumulation_steps 8 \
>train2.log 2>&1

CUDA_VISIBLE_DEVICES=0 python train.py \
--run_name "pythia_m_mt_2" \
--base_model pythia-medium \
--lora 8 \
--prompt_loss_weight 1 \
--max_length 1024 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--eval_accumulation_steps 8 \
>train3.log 2>&1


ç \
>train4.log 2>&1

CUDA_VISIBLE_DEVICES=1 python train.py \
--run_name "mamba_m_mt_1" \
--base_model mamba-medium \
--lora 8 \
--prompt_loss_weight 0.1 \
--max_length 1024 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--eval_accumulation_steps 8 \
>train5.log 2>&1

CUDA_VISIBLE_DEVICES=1 python train.py \
--run_name "mamba_m_mt_2" \
--base_model mamba-medium \
--lora 8 \
--prompt_loss_weight 1 \
--max_length 1024 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--eval_accumulation_steps 8 \
>train6.log 2>&1
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


## example of local model

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --run_name test_saving_model  --config config_test.json --batch_size 64 --base_model "finetuned_models/test_saving_model_2024_03_12_1415/checkpoint-24" --model_from_local
```


# 20VR VRAM
CUDA_VISIBLE_DEVICES=1 python train.py --run_name mamba_s_mt_0_bf16 --base_model mamba-small  --prompt_loss_weight 0 --max_length 512 --config config_mt.json  --dataset "sentiment-train,headline,finred*3,ner*15" --eval_accumulation_steps 8 --bf16 --eval_steps 0.01 --batch_size 32

CUDA_VISIBLE_DEVICES=1 python train.py --run_name "mamba_s_mt_0" --base_model mamba-small  --prompt_loss_weight 0 --max_length 1024 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --eval_accumulation_steps 8 --shared_dir ~/shared


bf16 1K 1.4B - 38GB VRAM - 91h a6000


CUDA_VISIBLE_DEVICES=0 python train.py \
    --run_name "test" \
    --base_model pythia-small \
    --prompt_loss_weight 1 \
    --max_length 1024 \
    --config config_mt.json \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --eval_accumulation_steps 8





accelerate launch train.py --run_name "pythia_m_mt_0" --base_model pythia-medium     --lora 8 --prompt_loss_weight 0 --max_length 2048 --config config_mt.json --dataset "sentiment-train,headline,finred*3,ner*15" --eval_accumulation_steps 4 --gradient_steps 4 --distributed

CUDA_VISIBLE_DEVICES=0 python train.py \
--run_name "pythia_s_mt_1" \
--base_model pythia-small \
--prompt_loss_weight 0.1 \
--max_length 512 \
--config config_mt.json \
--dataset "sentiment-train,headline,finred*3,ner*15" \
--num_epochs 2 \
--batch_size 32 \
--eval_steps 0.02 \
--eval_accumulation_steps 8
--shared_dir ~/shared