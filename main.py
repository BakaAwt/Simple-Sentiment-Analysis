# Don't touch fish! Or you will get an F
# By Kirin
# https://docs.python.org/zh-cn/3/library/csv.html
# https://sraf.nd.edu/textual-analysis/resources/
# https://stackoverflow.com/questions/33543446/what-is-the-formula-of-sentiment-calculation
# https://provenclei.github.io/2020/02/01/TF-IDF.html#23-python
# http://www.ruanyifeng.com/blog/2013/03/tf-idf.html

import csv
import re

# Define global variables
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
        # Handle data
        if not senti_dict_path:
            print('Please input senti_dict_path')
            quit()
        else:
            self.handle_senti_dict(senti_dict_path)
        if not test_x_path:
            print('Please input test_x_path')
            quit()
        else:
            self.handle_test_x(test_x_path)
        if not test_y_path:
            print('Please input test_y_path')
            quit()
        if train_path:
            self.handle_train(train_path)
        if custom_dict_path:
            self.handle_custom_dict(custom_dict_path)
            merged_dict.update(custom_dict)
        merged_dict.update(senti_dict)

    def handle_senti_dict(self, path):
        global senti_dict
        # Preprocessing senti_dict type of dictionary
        with open(path) as path:
            csv_reader = csv.reader(path)
            for line in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                senti_dict[line[1]] = float(line[2])
            print('senti_dict library has %s line(s).' %len(senti_dict))

    def handle_test_x(self, path):
        global test_x
        # Processing test_x type of two-dimensional array
        with open(path) as path:
            csv_reader = csv.reader(path)
            for line in csv_reader:
                line[0] = int(line[0])
                test_x.append(line)
            # test_x = sorted(test_x)
            print('test_x library has %s line(s).' %len(test_x))

    def handle_train(self, path):
        global train
        # Processing train type of two-dimensional array
        with open(path) as path:
            csv_reader = csv.reader(path)
            for line in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                line[0] = int(line[0])
                train.append(line)
            # train = sorted(train)
            print('train library has %s line(s).' %len(train))

    def handle_custom_dict(self, path):
        global custom_dict
        # Preprocessing custom_dict type of dictionary
        with open(path) as path:
            csv_reader = csv.reader(path)
            for line in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                custom_dict[line[0]] = float(line[1])
                idf_dict[line[0]] = float(line[2])
            print('custom_dict library has %s line(s).' %len(custom_dict))

# return a list
def getWords(text):
    # Use RegEx to extract the part that meets the specification
    text = re.sub("[^a-zA-Z]", " ", text)
    # Lowercase all words and turn them into word list
    words = text.lower().split()
    # return words
    return words

# return an integer
def countWords(text):
    return len(getWords(text))

def sentimentAnalysis(data, num):
    global accuCount
    # -= Extract by line =-
    # For test_x, line[0] is id，line[1] is sentence
    # For train，line[0] is id，line[1] is given result, line[2] is sentence
    # There are judgment statements for two data sources in main()
    # index is the 0-based count of the sentence
    for index, line in enumerate(data):
        positiveCount = 0
        negativeCount = 0
        sum = 0
        diff = 0
        sentiPoint = 0.0
        max_tf_idf = 0
        max_word = ''
        sentiType = 'neutral'
        # -= Perform judgment in units of words =-
        # Count the number of positive and negative words separately
        # line[num] as string
        for word in getWords(line[num]):
            # find word in dictionary
            # merged_dict[word] idf_dict[word]
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
    # Calculate accuracy using train.txt 
    if num == 2:
        print('Accuracy is %s.' %(accuCount / len(data)))
    return test_y

def listToCSV(list, path):
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list)
        print('Results saved in %s' %path)

# Count the frequency of the word word in the sentence
# Input word, sentence as string，idf as float
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