if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=MWD_MFIC
root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
random_seed=2021

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --random_seed $random_seed\
    --is_training 1\
    --root_path $root_path_name \
    --data_path $data_path_name\
    --model_id  $model_id_name'_'$seq_len'_'$pred_len \
    --model  $model_name\
    --data custom\
    --features M\
    --seq_len  $data_name\
    --pred_len  $pred_len \
    --enc_in 862\
    --head_d ropout 0.3\
    --des 'Exp'\
    --train_epochs 100\
    --patience 10\
    --itr 1\
    --batch_size 256\
    --learning_rate 0.001\
    --wavelet_layers 4\
    --wavelet_type db5\
    --wavelet_mode periodization\
    --wavelet_dim 128\
    --hidden_wavelet_dim 256\
    --SmoothL1Loss_beta 1.0\
#      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done