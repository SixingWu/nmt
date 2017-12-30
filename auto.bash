export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_1_test_f.inf.response >> decoded/log/decoded/char_cnn_strides3_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides3_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides3_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original2_char_level   --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_1_test_f.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original2_char_level /best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original2_char_level /best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/real_separate_word_40000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_1_test_f.inf.response >> decoded/log/decoded/word_rnn_separate_4w_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/real_separate_word_40000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_1_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_separate_4w_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/real_separate_word_40000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_1_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_separate_4w_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_word_40000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_1_test_f.inf.response >> decoded/log/decoded/word_rnn_shared_4W_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_word_40000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_1_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_shared_4W_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_word_40000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_1_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_shared_4W_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_nopooling_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_1_test_f.inf.response >> decoded/log/decoded/char_cnn_strides1_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_nopooling_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides1_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_nopooling_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides1_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original3_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_1_test_f.inf.response >> decoded/log/decoded/char_cnn_strides2_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original3_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides2_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original3_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides2_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_nopoolingres_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_1_test_f.inf.response >> decoded/log/decoded/char_cnn_concat_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_nopoolingres_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_concat_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_nopoolingres_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_concat_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/shared_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_1_test_f.inf.response >> decoded/log/decoded/char_rnn_shared_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/shared_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_rnn_shared_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/shared_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_rnn_shared_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_1_test_f.inf.response >> decoded/log/decoded/char_cnn_strides5_original_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides5_original_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides5_original_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_sub_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_1_test_f.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_sub_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_1_test_fbest_rouge.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_sub_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_1_test_fbest_bleu.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_1_test_f.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/cnn_separate_original_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_sub2_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_1_test_f.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_sub2_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_1_test_fbest_rouge.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_sub2_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_1_test_fbest_bleu.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/shared_word_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_1_test_f.inf.response >> decoded/log/decoded/word_rnn_separate_8w_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/shared_word_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_1_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_separate_8w_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/shared_word_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_1_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_separate_8w_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_1_test_f.inf.response >> decoded/log/decoded/char_rnn_separate_1_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_1_test_fbest_rouge.inf.response >> decoded/log/decoded/char_rnn_separate_1_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=1 --out_dir=models/separate_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_1_test_fbest_bleu.inf.response >> decoded/log/decoded/char_rnn_separate_1_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_5_test_f.inf.response >> decoded/log/decoded/char_cnn_strides3_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides3_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides3_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original2_char_level   --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_5_test_f.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original2_char_level /best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original2_char_level /best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/real_separate_word_40000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_5_test_f.inf.response >> decoded/log/decoded/word_rnn_separate_4w_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/real_separate_word_40000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_5_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_separate_4w_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/real_separate_word_40000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_5_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_separate_4w_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_word_40000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_5_test_f.inf.response >> decoded/log/decoded/word_rnn_shared_4W_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_word_40000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_5_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_shared_4W_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_word_40000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_5_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_shared_4W_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_nopooling_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_5_test_f.inf.response >> decoded/log/decoded/char_cnn_strides1_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_nopooling_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides1_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_nopooling_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides1_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original3_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_5_test_f.inf.response >> decoded/log/decoded/char_cnn_strides2_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original3_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides2_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original3_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides2_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_nopoolingres_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_5_test_f.inf.response >> decoded/log/decoded/char_cnn_concat_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_nopoolingres_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_concat_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_nopoolingres_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_concat_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/shared_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_5_test_f.inf.response >> decoded/log/decoded/char_rnn_shared_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/shared_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_rnn_shared_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/shared_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_rnn_shared_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_5_test_f.inf.response >> decoded/log/decoded/char_cnn_strides5_original_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides5_original_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides5_original_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_sub_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_5_test_f.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_sub_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_5_test_fbest_rouge.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_sub_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_5_test_fbest_bleu.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_5_test_f.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/cnn_separate_original_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_sub2_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_5_test_f.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_sub2_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_5_test_fbest_rouge.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_sub2_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_5_test_fbest_bleu.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/shared_word_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_5_test_f.inf.response >> decoded/log/decoded/word_rnn_separate_8w_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/shared_word_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_5_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_separate_8w_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/shared_word_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_5_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_separate_8w_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_5_test_f.inf.response >> decoded/log/decoded/char_rnn_separate_5_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_5_test_fbest_rouge.inf.response >> decoded/log/decoded/char_rnn_separate_5_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=5 --out_dir=models/separate_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_5_test_fbest_bleu.inf.response >> decoded/log/decoded/char_rnn_separate_5_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_10_test_f.inf.response >> decoded/log/decoded/char_cnn_strides3_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides3_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides3_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original2_char_level   --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_10_test_f.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original2_char_level /best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original2_char_level /best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_max8_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides5_max8_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/real_separate_word_40000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_10_test_f.inf.response >> decoded/log/decoded/word_rnn_separate_4w_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/real_separate_word_40000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_10_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_separate_4w_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/real_separate_word_40000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_4w_10_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_separate_4w_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_word_40000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_10_test_f.inf.response >> decoded/log/decoded/word_rnn_shared_4W_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_word_40000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_10_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_shared_4W_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_word_40000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_shared_4W_10_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_shared_4W_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_nopooling_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_10_test_f.inf.response >> decoded/log/decoded/char_cnn_strides1_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_nopooling_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides1_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_nopooling_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides1_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides1_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original3_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_10_test_f.inf.response >> decoded/log/decoded/char_cnn_strides2_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original3_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides2_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original3_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides2_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides2_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_nopoolingres_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_10_test_f.inf.response >> decoded/log/decoded/char_cnn_concat_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_nopoolingres_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_concat_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_nopoolingres_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_concat_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_concat_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/shared_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_10_test_f.inf.response >> decoded/log/decoded/char_rnn_shared_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/shared_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_rnn_shared_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/shared_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_shared_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_rnn_shared_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_beam_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_10_test_f.inf.response >> decoded/log/decoded/char_cnn_strides5_original_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_beam_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides5_original_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_beam_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides5_original_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides5_original_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_sub_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_10_test_f.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_sub_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_10_test_fbest_rouge.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_sub_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel/test.message  --inference_output_file=decoded/sub1_rnn_separate_4w_10_test_fbest_bleu.inf.response >> decoded/log/decoded/sub1_rnn_separate_4w_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_10_test_f.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/cnn_separate_original_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_cnn_strides3_nobeam_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_cnn_strides3_nobeam_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_sub2_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_10_test_f.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_sub2_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_10_test_fbest_rouge.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_sub2_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/sublevel2/test.message  --inference_output_file=decoded/sub2_rnn_separate_8w_10_test_fbest_bleu.inf.response >> decoded/log/decoded/sub2_rnn_separate_8w_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/shared_word_80000_level  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_10_test_f.inf.response >> decoded/log/decoded/word_rnn_separate_8w_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/shared_word_80000_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_10_test_fbest_rouge.inf.response >> decoded/log/decoded/word_rnn_separate_8w_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/shared_word_80000_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/wordlevel/test.message  --inference_output_file=decoded/word_rnn_separate_8w_10_test_fbest_bleu.inf.response >> decoded/log/decoded/word_rnn_separate_8w_10_test_fbest_bleu.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_char_level  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_10_test_f.inf.response >> decoded/log/decoded/char_rnn_separate_10_test_f.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_char_level/best_rouge  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_10_test_fbest_rouge.inf.response >> decoded/log/decoded/char_rnn_separate_10_test_fbest_rouge.log
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --beam_width=10 --out_dir=models/separate_char_level/best_bleu  --inference_input_file=/ldev/tensorflow/nmt/data/charlevel/test.message  --inference_output_file=decoded/char_rnn_separate_10_test_fbest_bleu.inf.response >> decoded/log/decoded/char_rnn_separate_10_test_fbest_bleu.log
