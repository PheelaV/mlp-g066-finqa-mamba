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

```
python train.py \
--run_name \ 
--dataset sentiment-train,headline,finred*3,ner*15 \
--max_length 512 \
>train.log 2>&1 &
```
tail -f train.log
to check the training log


 python train.py --run_name --dataset "sentiment-train,headline,finred*3,ner*15" --max_length 512
# trash


trying to replicate mamba_chat runs

OOM on wood
```
python train.py --run_name pythia_big_paged_ad  --config config_test.json --lora_r 0 --base_model pythia-big --gradient_steps 4 --optim paged_adamw_8bit --batch_size 1
```

```sh
python train.py --run_name local_test \
    --dataset "sentiment-train,headline,finred*3,ner*15" \
    --max_length 512 

 ```