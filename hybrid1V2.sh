export CUDA_VISIBLE_DEVICES=2


python3 -m nrm.nrm \
    --src_embed_type=cnn_segment \
    --seg_len=5\
    --seg_embed_dim=128 \
    --num_units=256 \
    --embed_dim=512 \
    --num_gpus=2 \
    --share_vocab=False \
    --src_max_len=20 \
    --tgt_max_len=30 \
    --charcnn_min_window_size=1 \
    --charcnn_max_window_size=3 \
    --charcnn_max_k=1 \
    --charcnn_high_way_layer=4 \
    --charcnn_filters_per_windows=300\
    --charcnn_relu=relu \
    --batch_size=256 \
    --encoder_type=bi \
    --attention=luong\
    --src=message --tgt=response \
    --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hybrid2/vocab.40000  \
    --train_prefix=/ldev/tensorflow/nmt2/nmt/data/hybrid2/train.40000\
    --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/hybrid2/dev.40000 \
    --test_prefix=/ldev/tensorflow/nmt2/nmt/data/hybrid2/test.40000 \
    --out_dir=models/NRL1V2_charCNN_hybrid_4W \
    --num_train_steps=1000000 \
    --metrics=rouge@hybrid \
    --steps_per_stats=100 >> logs/NRL1V2_charCNN_hybrid_4W.txt 

export CUDA_VISIBLE_DEVICES=2
