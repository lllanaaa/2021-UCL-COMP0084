import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import contractions
import re
import string
import unidecode
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def loadData():
    file_path = "passage-collection.txt"
    data = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            data.append(line)
    return data


# normalisation not removing stop words
def processData(data):
    processedData = []
    for line in data:
        # should not remove stop words

        line = re.sub(' +', ' ', line)

        line = line.lower()

        line = contractions.fix(line)

        punctuation_table = str.maketrans('', '', string.punctuation)
        line = line.translate(punctuation_table)

        line = unidecode.unidecode(line)

        line = word_tokenize(line)

        for i in range(len(line)):
            line[i] = lemmatizer.lemmatize(line[i])
            line[i] = stemmer.stem(line[i])

        processedData.append(line)

    return processedData


def createInvertedIndex(data):
    dict = {}
    for i in range(len(data)):
        line = data[i]
        for item in line:
            if item not in dict:
                dict[item] = 1
            else:
                dict[item] += 1
    size = len(dict)
    print(size)
    return dict


def writeInCsv(data, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in data.items():
            writer.writerow(i)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def plot():
    # count the number of occurrences of terms in the provided dataset
    # plot their probability of occurrence (normalised frequency) against their frequency ranking
    # qualitatively justify that these terms follow Zipf's law
    # use Eq. to explain where their difference is coming from and also compare the two in a log-log plot

    df = pd.read_csv('task1_vocal.csv', names=['word', 'count'], encoding="gbk")
    size_term = df.shape[0]
    all_size = df['count'].sum()
    print("all size of term: ", all_size)

    df['frequency'] = df['count'] / df['count'].sum()
    df['rank'] = df['count'].rank(ascending=False, method='min')
    df['rank'] = df['rank'].astype(int)
    df['rank*frequency'] = df['rank'] * df['frequency']
    df.sort_values(by="rank", inplace=True, ascending=True)

    # all terms
    N = size_term
    # EQ1
    df["Zipf's law"] = 1 / (df['rank'] * sum([n ** (-1) for n in range(1, N+1)]))
    df.plot(x='rank', y=['frequency', "Zipf's law"], style=['-', '-.'], xlabel='Term frequency ranking',
            ylabel='Term prob. of occurrence', title='all terms')
    plt.show()

    # top-1000 most frequent term
    df1000 = df.head(1000)
    df1000.plot(x='rank', y=['frequency', "Zipf's law"], style=['-', '-.'], xlabel='Term frequency ranking',
                ylabel='Term prob. of occurrence', title='top-1000 most frequent terms')
    plt.show()

    # log-log
    df.plot(x='rank', y=['frequency', "Zipf's law"], style=['-', '-.'], xlabel='Term frequency ranking(log)',
            ylabel='Term prob. of occurrence(log)', title='all terms', logx=True, logy=True)
    plt.show()


print("task1")
data = loadData()
processed_data = processData(data)
inverted_index_dict = createInvertedIndex(processed_data)
writeInCsv(inverted_index_dict, "task1_vocal.csv")
plot()
