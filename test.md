mkdir tmp/nrm_model
python3 -m nrm.nrm \
    --attention=luong\
    --src=message --tgt=response \
    --vocab_prefix=/ldev/tensorflow/seq2seq/data/CharLevel2/vocab  \
    --train_prefix=/ldev/tensorflow/seq2seq/data/CharLevel2/train ls\
    --dev_prefix=/ldev/tensorflow/seq2seq/data/CharLevel2/dev \
    --test_prefix=/ldev/tensorflow/seq2seq/data/CharLevel2/test \
    --out_dir=tmp/nrm_tmp_model \
    --num_train_steps=30000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \