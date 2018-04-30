export CUDA_VISIBLE_DEVICES=1

    
python3 -m nrm.nrm \
    --seg_len=5\
    --seg_embed_dim=128 \
    --num_units=256 \
    --embed_dim=512 \
    --attention_architecture=char_standard \
    --residual_cnn_layer_type=concat \
    --width_strides=3 \
    --cnn_min_window_size=1 \
    --cnn_max_window_size=5 \
    --num_gpus=2 \
    --share_vocab=False \
    --src_max_len=30 \
    --tgt_max_len=30 \
    --batch_size=256 \
    --encoder_type=bi \
    --attention=luong\
    --src=message --tgt=response \
    --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/charlevel2/vocab.separate   \
    --train_prefix=/ldev/tensorflow/nmt2/nmt/data/charlevel2/train\
    --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/charlevel2/dev \
    --test_prefix=/ldev/tensorflow/nmt2/nmt/data/charlevel2/test \
    --out_dir=models/cnnEncDecM2 \
    --num_train_steps=1000000 \
    --metrics=bleu-2@char2word,bleu-4@char2char,rouge,accuracy  \
    --steps_per_stats=100 >> logs/cnnEncDecM2.txt 
    

   