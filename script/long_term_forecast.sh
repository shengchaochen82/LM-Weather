
export CUDA_VISIBLE_DEVICES=0
# 
seq_len=192
model=GPT4TS
rank=32

for pred_len in 96 192 336 720
do

nohup python -u main.py \
    -lbs 256 \
    -nc 15 \
    -jr 0.1 \
    -ls 5 \
    -lr 0.005 \
    -eg 5 \
    -gr 50 \
    -go long_term_forecast \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --base_model GPT2 \
    --freeze_part 2 \
    --gpt_layers 1 \
    --lradj type1 \
    --features MS \
    --freq h \
    --target rh \
    --dataset Weather-Tiny \
    --model $model \
    --is_peft 1 \
    --peft lora \
    --rank $rank \
    --is_gpt 1 > logs/LTF_GPT_lora4_fre2_rk_{$rank}_sl192_pl$pred_len.out 2>&1 &

wait

done