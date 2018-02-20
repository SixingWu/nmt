export CUDA_VISIBLE_DEVICES=1
python3 -m nrm.nrm \
    --num_units=256 \
    --embed_dim=512 \
    --num_gpus=2 \
    --share_vocab=False \
    --src_max_len=20 \
    --tgt_max_len=20 \
    --batch_size=256 \
    --encoder_type=bi \
    --attention=luong\
    --src=message --tgt=response \
    --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.20000.separate  \
    --train_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/train \
    --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/dev \
    --test_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/test \
    --out_dir=models/hybrid2WN \
    --num_train_steps=1000000 \
    --metrics=bleu-2@hybrid,bleu-4@char,rouge,accuracy \
    --steps_per_stats=100 >> logs/hybrid2WN.txt