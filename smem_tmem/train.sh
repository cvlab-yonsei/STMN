TRAIN_TXT=../database/MARS_database/train_path.txt
TRAIN_INFO=../database/MARS_database/train_info.npy
TEST_TXT=../database/MARS_database/test_path.txt
TEST_INFO=../database/MARS_database/test_info.npy
QUERY_INFO=../database/MARS_database/query_IDX.npy

CKPT=./log
python3 main.py \
    --smem_size 10 --smem_margin 0.3 \
    --tmem_size 5 --tmem_margin 0.3 \
    --train_txt $TRAIN_TXT --train_info $TRAIN_INFO  --test_batch 128 \
    --test_txt $TEST_TXT  --test_info $TEST_INFO --query_info $QUERY_INFO \
    --n_epochs 200 --lr 0.0001 --lr_step_size 50 --optimizer adam \
    --ckpt $CKPT --log_path loss.txt \
    --class_per_batch 8 --track_per_class 4 --seq_len 6 \
    --feat_dim 2048 --stride 1 --gpu_id '0,1,2' ;
