# 摸鱼划水是要遭报应的
# By Kirin
# https://docs.python.org/zh-cn/3/library/csv.html
# https://sraf.nd.edu/textual-analysis/resources/
# https://stackoverflow.com/questions/33543446/what-is-the-formula-of-sentiment-calculation
# https://provenclei.github.io/2020/02/01/TF-IDF.html#23-python
# http://www.ruanyifeng.com/blog/2013/03/tf-idf.html

import csv
import re
# import math

# 定义全局变量
senti_dict = {}
idf_dict = {}
test_x = []
test_y = []
train = []
custom_dict = {}
merged_dict = {}
accuCount = 0.0

class handleData:

    def __init__(self, senti_dict_path, test_x_path, test_y_path, train_path=None, custom_dict_path=None):
        # 执行处理
        if not senti_dict_path:
            print('请输入 senti_dict_path')
            quit()
        else:
            self.handle_senti_dict(senti_dict_path)
        if not test_x_path:
            print('请输入 test_x_path')
            quit()
        else:
            self.handle_test_x(test_x_path)
        if not test_y_path:
            print('请输入 test_y_path')
            quit()
        if train_path:
            self.handle_train(train_path)
        if custom_dict_path:
            self.handle_custom_dict(custom_dict_path)
            merged_dict.update(custom_dict)
        merged_dict.update(senti_dict)

    def handle_senti_dict(self, path):
        global senti_dict
        # 预处理 senti_dict 类型为 dictionary
        with open(path) as path:
            csv_reader = csv.reader(path)
            for line in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                senti_dict[line[1]] = float(line[2])
            print('senti_dict 库共有 %s 行' %len(senti_dict))

    def handle_test_x(self, path):
        global test_x
        # 预处理 test_x 类型为 二维数组
        with open(path) as path:
            csv_reader = csv.reader(path)
            for line in csv_reader:
                line[0] = int(line[0])
                test_x.append(line)
            #test_x = sorted(test_x)
            print('test_x 库共有 %s 行' %len(test_x))

    def handle_train(self, path):
        global train
        # 预处理 train 类型为 二维数组
        with open(path) as path:
            csv_reader = csv.reader(path)
            for line in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                line[0] = int(line[0])
                train.append(line)
            #train = sorted(train)
            print('train 库共有 %s 行' %len(train))

    def handle_custom_dict(self, path):
        global custom_dict
        # 预处理 senti_dict 类型为 dictionary
        with open(path) as path:
            csv_reader = csv.reader(path)
            for line in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                custom_dict[line[0]] = float(line[1])
                idf_dict[line[0]] = float(line[2])
            print('custom_dict 库共有 %s 行' %len(custom_dict))

# 返回一个 list
def getWords(text):
    # 用正则表达式取出符合规范的部分
    text = re.sub("[^a-zA-Z]", " ", text)
    # 小写化所有的词，并转成词list
    words = text.lower().split()
    # 返回words
    return words

# 返回一个 Integer
def countWords(text):
    return len(getWords(text))

def sentimentAnalysis(data, num):
    global accuCount
    # -= 以行为单位提取 =-
    # 对于 test_x，line[0] 为 id，line[1] 为 sentence
    # 对于 train，line[0] 为 id，line[1] 为 句子的情感分析结果，line[2] 为 sentence
    # 因此在 main() 中有对两种数据源的判断语句
    # index 为该句子的从 0 开始的计数
    for index, line in enumerate(data):
        positiveCount = 0
        negativeCount = 0
        sum = 0
        diff = 0
        sentiPoint = 0.0
        max_tf_idf = 0
        max_word = ''
        sentiType = 'neutral'
        # -= 以单词为单位执行判断 =-
        # 分别统计正面与负面词汇的数量
        # line[num] 为 string，为整个句子的字符串值
        for word in getWords(line[num]):
            # word 命中字典词库
            # merged_dict[word] 为极端分，idf_dict[word] 为该词语的 idf 值
            if word in merged_dict:
                tf_idf_val = tf_idf(word, line[num], idf_dict[word])
                if tf_idf_val > max_tf_idf:
                    max_tf_idf = tf_idf_val
                    max_word = word
        if max_word != '':
            if merged_dict[max_word] > 0.01:
                sentiType = 'positive'
            elif merged_dict[max_word] < -0.01:
                sentiType = 'negative'
            else:
                sentiType = 'neutral'
        else:
            sentiType = 'neutral'
        test_y.append([line[0], sentiType])
        if sentiType == line[1]:
            accuCount += 1
    # 根据 train.txt 计算准确率
    if num == 2:
        print('精确度为', accuCount / len(data))
    return test_y

def listToCSV(list, path):
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list)
        print('已将结果保存于 %s' %path)

# 句子 sentence 中词 word 出现的频率
# 传入 word, sentence 为 string，idf 为 float
def tf_idf(word, sentence, idf):
    tf = getWords(sentence).count(word) / countWords(sentence)
    return tf * idf

def main():
    print('Python for Finance Project: Financial Statement Sentiment Analysis')
    handleData('./senti_dict.csv', './test_x.txt', './test_y.txt', './train.txt', './sentiment_words_dict.txt')
    if train != []:
        listToCSV(sentimentAnalysis(train, 2), './test_train.txt')
    else:
        listToCSV(sentimentAnalysis(test_x, 1), './test_y.txt')

if __name__ == '__main__':
    main()