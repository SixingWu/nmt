# Configs including a end
seg_len = 8
seg_end = '<#>'
seg_pad = '<_>'
seg_separator = '\t'
seg_inter_separator = ' '

# read the vocabs and add all characters in the vocab
def add_chars_to_vocab(vocab_path, word_num = -1):
    with open(vocab_path,'r+',encoding='utf-8') as fin:
        with open(vocab_path+'_seg','w+',encoding='utf-8') as fout:
            lines = fin.readlines()
            print('original num: %d' % len(lines))
            fout.write(''.join(lines[0:3]))
            vocab = set()
            vocab.add(seg_end)
            vocab.add(seg_pad)
            # first 3 lines are special tokens
            if word_num != -1:
                lines = lines[3:3+word_num]
            else:
                lines = lines[3:]
            for line in lines:
                word = line.strip('\n')
                #vocab.add(word)
                for char in word:
                    vocab.add(char)
            for token in vocab:
                fout.write(token+'\n')
            print('new num: %d' % len(vocab))

add_chars_to_vocab(r'/Users/mebiuw/PycharmProjects/nmt/nmt_data/vocab.en')
add_chars_to_vocab(r'/Users/mebiuw/PycharmProjects/nmt/nmt_data/vocab.vi')


def convert_to_seg_file(vocab_path, seg_len):
    with open(vocab_path, 'r+', encoding='utf-8') as fin:
        with open(vocab_path + '_seg', 'w+', encoding='utf-8') as fout,open(vocab_path + '_seg_len', 'w+', encoding='utf-8') as flout:
            lines = fin.readlines()
            for line in lines:
                items = line.strip('\n').split(' ')
                seg_items = []
                for item in items:
                    item = list(item)
                    item = item[0:seg_len - 1]
                    item.append(seg_end)
                    while len(item) != seg_len:
                        item.append(seg_pad)
                    seg_items.append(seg_inter_separator.join(item))
                flout.write(' '.join([str(min(seg_len,len(x)+1)) for x in items]) + '\n')
                fout.write(seg_separator.join(seg_items) + '\n')


convert_to_seg_file(r'/Users/mebiuw/PycharmProjects/nmt/nmt_data/train.en', seg_len)
convert_to_seg_file(r'/Users/mebiuw/PycharmProjects/nmt/nmt_data/train.vi', seg_len)
convert_to_seg_file(r'/Users/mebiuw/PycharmProjects/nmt/nmt_data/tst2012.en', seg_len)
convert_to_seg_file(r'/Users/mebiuw/PycharmProjects/nmt/nmt_data/tst2012.vi', seg_len)
convert_to_seg_file(r'/Users/mebiuw/PycharmProjects/nmt/nmt_data/tst2013.en', seg_len)
convert_to_seg_file(r'/Users/mebiuw/PycharmProjects/nmt/nmt_data/tst2013.vi', seg_len)

