if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=MWD_MFIC
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
random_seed=2021

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1\
      --root_path $root_path_name\
      --data_path  $data_path_name\
      --model_id  $model_id_name'_'$seq_len'_'$pred_len0\
      --model  $model_name\
      --data  $data_name\
      --features M\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321\
      --head_dropout 0.1\
      --des 'Exp'\
      --train_epochs 100\
      --patience 10\
      --itr 1\
      --batch_size 32\
      --learning_rate  0.001\
      --wavelet_layers 6\
      --wavelet_type haar\
      --wavelet_mode periodization\
      --wavelet_dim 256\
      --hidden_wavelet_dim 64\
      --SmoothL1Loss_beta 1.0\
#      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done