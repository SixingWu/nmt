python3 -m nrm.nrm      --steps_per_stats=100      --attention=luong      --metrics=rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid      --embed_dim=512      --vocab_prefix=/home/mebiuw/nmt/data/enhllevel/vocab.40000      --charcnn_high_way_layer=4      --charcnn_max_window_size=8      --charcnn_high_way_type=uniform      --tgt_max_len=35      --charcnn_filters_per_windows=100      --src_embed_type=rl1_rnn_segment_attention      --unit_type=lstm      --seg_len=10      --src=message      --seg_embed_dim=160      --out_dir=models/en_rnn_hlat      --infer_batch_size=10      --train_prefix=/home/mebiuw/nmt/data/enhllevel/train.40000      --batch_size=256      --num_train_steps=1000000      --tgt=response      --encoder_type=bi      --test_prefix=/home/mebiuw/nmt/data/enhllevel/test.40000      --seg_embed_mode=separate      --dev_prefix=/home/mebiuw/nmt/data/enhllevel/dev.40000      --charcnn_min_window_size=1      --num_units=512      --num_layers=2      --src_max_len=35      --high_way_layer=4      --charcnn_relu=relu      --share_vocab=False     >> logs/en_rnn_hlat.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/en_rnn_hlat --vocab_prefix=/home/mebiuw/nmt/data/enhllevel/vocab.40000 --inference_input_file=/home/mebiuw/nmt/data/enhllevel/test.40000.message --inference_output_file=infer_test/en_rnn_hlat.test.txt >> infer_test/log/en_rnn_hlat.test.txt
    
python3 -m nrm.utils.evaluation_utils en_rnn_hlat /home/mebiuw/nmt/data/enhllevel/test.40000.response infer_test/en_rnn_hlat.test.txt infer_test/scores/en_rnn_hlat.test.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
    
