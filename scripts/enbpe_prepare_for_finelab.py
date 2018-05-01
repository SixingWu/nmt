
# coding: utf-8

# In[16]:

import os
from collections import defaultdict

def build_vocab(path,out_path):
    counter = defaultdict(int)
    total = 0
    with open(path,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        token_set = set()
        total = len(lines)
        for line in lines:
            items = line.strip('\n').split()
            counter[len(items)] += 1
            for item in items:
                token_set.add(item)
    sorted_items = sorted(counter.items(), key = lambda x:x[0])
    sums = 0;
    for item in sorted_items:
        sums += item[1]
        print("%d/%f" % (item[0], sums/total))
    with open(out_path,'w+',encoding='utf-8') as fout:
        for token in token_set:
            fout.write('%s\n' % token)

def count(path):
    with open(path,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        token_set = set()
        for line in lines:
            items = line.strip('\n').split()
            for item in items:
                token_set.add(item)
        return len(token_set)
def worker(input_path,output_path,bpe_path,bpe_symbols=40000,target_symbols=40000):
    while True:
        print('python learn_bpe.py -i %s -o %s -s %d' % (input_path,bpe_path,bpe_symbols))
        print(os.system('python learn_bpe.py -i %s -o %s -s %d' % (input_path,bpe_path,bpe_symbols)))
        print('python3 apply_bpe.py -i %s -c %s -o %s' % (input_path,bpe_path,output_path))
        print(os.system('python3 apply_bpe.py -i %s -c %s -o %s' % (input_path,bpe_path,output_path)))
        now_count = count(output_path)
        print('%d/%d' % (now_count,target_symbols))
        if now_count == target_symbols:
            print('Done')
            break
        elif now_count < target_symbols:
            print('error')
            break
        else:
            bpe_symbols -= (now_count - target_symbols)
            print('reset to %d' % bpe_symbols)
    


# In[24]:

#worker('../data/wordlevel/train.message','../data/bpelevel/train.message','../data/bpelevel/train.message.bpe',34075)
#worker('../data/wordlevel/train.response','../data/bpelevel/train.response','../data/bpelevel/train.resonse.bpe',25300)
#worker('../data/enwordlevel/train.response','../data/enbpelevel/train.response','../data/enbpelevel/train.resonse.bpe',36246)
#worker('../data/enwordlevel/train.message','../data/enbpelevel/train.message','../data/enbpelevel/train.message.bpe',35081)


# In[27]:
os.system('mkdir ../data/enbpelevel')

os.system('python3 apply_bpe.py -i ../data/enwordlevel/dev.message -c ../data/enbpelevel/train.message.bpe -o ../data/enbpelevel/dev.message')
os.system('python3 apply_bpe.py -i ../data/enwordlevel/test.message -c ../data/enbpelevel/train.message.bpe -o ../data/enbpelevel/test.message')
os.system('python3 apply_bpe.py -i ../data/enwordlevel/dev.response -c ../data/enbpelevel/train.response.bpe -o ../data/enbpelevel/dev.response')
os.system('python3 apply_bpe.py -i ../data/enwordlevel/test.response -c ../data/enbpelevel/train.response.bpe -o ../data/enbpelevel/test.response')

build_vocab('../data/enbpelevel/train.response','../data/enbpelevel/vocab.40000.separate.response')
build_vocab('../data/enbpelevel/train.message','../data/enbpelevel/vocab.40000.separate.message')


# In[ ]:

