import random
print('这是一个对话测试，人工标注的程序，本程序有如下特点：')
print("""
1、本程序自带随机，对话每次都会被打乱，保证你不会因为选择特定模型而给分！、
2、自带基准培训教程，需要完全通过基准培训教程测试后，才能进行评分！

程序运行节奏：
#1 首先进行基准培训教程，以及测试
#2 通过#1后可以进行打分

""")
def mark(message, responses,title='1/1',ground_truth=None):
    n = len(responses)
    
    print('%s：#############################' % (title))
    flag = False
    orders = random.sample(range(n),n)
    if ground_truth is not None:
        answer = ''.join([str(orders[i]) for i in range(n)])
        print("教学模式：当前的答案应该是，请记住这次评分标准：%s" % answer)
    while(flag is False):
        print('当前的上一句是：【%s】' % message)
        print('当前的回复是，请仔细阅读后打分：')
        for i,index in enumerate(orders):
            print('%d:\t【%s】' % (i, responses[index]))
        line = input('请输入你的分数，0分1分2分，格式如同：0101010101（长度一直就可以）:')
        print('读入：【%s】' % line)
        flag = True
        if len(line) != n:
            print('!!! 长度不合法，次样例数据将会重新进行')
            flag = False
        for digit in line:
            if digit not in ['0','1','2']:
                print('!!! 只能输入012，0分最低，2分最高')
                flag = False
                break
        if ground_truth is not None and line!= answer:
             print("教学模式：输入错误：当前的答案应该是，请记住这次评分标准：%s" % answer)
             flag = False
    scores = [0] * n
    for i,index in enumerate(orders):
        scores[index] = int(line[i])
    return''.join([str(x) for x in scores])


def read_lines(path):
    with open(path,'r+',encoding='utf-8') as fin:
        lines = [ line.strip('\n') for line in fin.readlines()]
    return lines

scores = []
total_scores = [0]*6
messages = read_lines('dataset/message.txt')
responses = []
for i in range(6):
    responses.append(read_lines('dataset/response%d.txt' % i))
for i in range(50):
    score_line = mark(messages[i], [responses[j][i] for j in range(6)],'%d/50' % i)
    scores.append(score_line)
    for i in range(6):
        total_scores[i] += int(score_line[i])

with open('dataset/scores.txt','w+') as fout:
    fout.write('\n'.join(scores))
    fout.write('\n'+total_scores)
print('测试完成！ 十分感谢你的支持！ 接下来请把程序当前目录下的scores.txt发送给我吧 wusixing@pku.edu.cn,只需要这个文件哦！')
