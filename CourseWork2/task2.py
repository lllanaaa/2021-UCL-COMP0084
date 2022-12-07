import pandas as pd
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from collections import Counter
import contractions
import re
import string
import unidecode
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score, dcg_score
from numpy import log, dot, exp, shape
import math
import matplotlib.pyplot as plt


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def processDataRemoveStopword(data):
    stop_words = set(stopwords.words('english'))
    processedData = []
    for line in data:

        # whitespace: replace more than one space with a single space
        line = re.sub(' +', ' ', line)

        # convert to lowercase
        line = line.lower()

        # expand contractions
        line = contractions.fix(line)

        # remove punctuation
        punctuation_table = str.maketrans('', '', string.punctuation)
        line = line.translate(punctuation_table)

        # remove unicode characters
        line = unidecode.unidecode(line)

        # tokenization
        line = word_tokenize(line)

        # Lemmatisation, Stemming
        # remove stop words
        arr = []
        for i in range(len(line)):
            if line[i] not in stop_words:
                processed = lemmatizer.lemmatize(line[i])
                arr.append(processed)

        processedData.append(arr)
    return processedData



class LogisticRegression1():
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize_weights(self, x, y):
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

    def fit(self, x, y):
        self.initialize_weights(x, y)

        cost_list = np.zeros(self.n_iterations,)
        # gradient descent
        for i in range(self.n_iterations):
            self.update_weight()
            cost_list[i] = self.cost()
        return cost_list

    def update_weight(self):
        # gradient descent
        h = self.sigmoid(self.x.dot(self.w) + self.b)
        tmp = np.reshape(h - self.y.T, self.m)
        dw = np.dot(self.x.T, tmp) / self.m
        db = np.sum(tmp) / self.m
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def cost(self):
        h = self.sigmoid(self.x.dot(self.w) + self.b)
        cost0 = self.y.T.dot(log(h))
        cost1 = (1 - self.y).T.dot(log(1 - h))
        cost = -((cost1 + cost0)) / len(self.y)
        return cost

    def predict(self, x):
        res = self.sigmoid(x.dot(self.w) + self.b)
        return res



def word_embedding():
    embedding_dict = {}
    with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embedding_dict[word] = vector
    return embedding_dict


def load_train_validation():
    df_train = pd.read_csv('train_data.tsv', sep='\t', header=0)
    df_validation = pd.read_csv('validation_data.tsv', sep='\t', header=0)
    return df_train, df_validation


def get_mean_vector(sentence, embedding_dict):
    terms_in_sentence = processDataRemoveStopword([sentence])[0]

    words = [word for word in terms_in_sentence if word in embedding_dict]
    arr = []
    for word in words:
        arr.append(embedding_dict[word])

    return np.mean(arr, axis=0)


def generate_embedding(df_train, df_validation, embedding_dict):
    dic = {}

    # train
    df_train_query = df_train[['qid', 'queries']]
    df_train_query.drop_duplicates(subset=['qid'], keep='first', inplace=True)
    df_train_passage = df_train[['pid', 'passage']]
    df_train_passage.drop_duplicates(subset=['pid'], keep='first', inplace=True)

    print(df_train_query.shape[0])
    num = 0
    for index, row in df_train_query.iterrows():
        if num % 10000 == 0:
            print(num)
        qid = row['qid']
        query = row['queries']
        vec = get_mean_vector(query, embedding_dict)
        dic[qid] = vec
        num += 1

    print(df_train_passage.shape[0])
    num = 0
    for index, row in df_train_passage.iterrows():
        if num % 10000 == 0:
            print(num)
        pid = row['pid']
        passage = row['passage']
        vec = get_mean_vector(passage, embedding_dict)
        dic[pid] = vec
        num += 1

    # validation
    df_validation_query = df_validation[['qid', 'queries']]
    df_validation_query.drop_duplicates(subset=['queries'], keep='first', inplace=True)
    df_validation_passage = df_validation[['pid', 'passage']]
    df_validation_passage.drop_duplicates(subset=['passage'], keep='first', inplace=True)

    print(df_validation_query.shape[0])
    num = 0
    for index, row in df_validation_query.iterrows():
        if num % 10000 == 0:
            print(num)
        qid = row['qid']
        if qid in dic:
            continue
        query = row['queries']
        vec = get_mean_vector(query, embedding_dict)
        dic[qid] = vec
        num += 1

    print(df_validation_passage.shape[0])
    num = 0
    for index, row in df_validation_passage.iterrows():
        if num % 10000 == 0:
            print(num)
        pid = row['pid']
        if pid in dic:
            continue
        passage = row['passage']
        vec = get_mean_vector(passage, embedding_dict)
        dic[pid] = vec
        num += 1

    with open('sentence_embedding_task2.pkl', 'wb') as file:
        pickle.dump(dic, file, protocol=-1)


def load_embedding_pickle():
    with open('sentence_embedding_task2.pkl', 'rb') as f:
        embedding_dict = pickle.load(f)
    return embedding_dict


def process_model_input(df_train, df_test, embedding_dict):

    print(df_train.shape[0])
    x_train = []
    y_train = []
    drop_row_train = []
    for index, row in df_train.iterrows():
        if index % 10000 == 0:
            print(index)
        qid = row['qid']
        pid = row['pid']
        relevancy = row['relevancy']
        if qid in embedding_dict and pid in embedding_dict:
            q = embedding_dict[qid]
            p = embedding_dict[pid]
            if q.shape == (50,) and p.shape == (50,):
                qp = np.hstack((q, p))
                x_train.append(qp)
                y_train.append([relevancy])
            else:
                drop_row_train.append(index)
        else:
            drop_row_train.append(index)

    print(df_test.shape[0])
    x_test = []
    y_test = []
    drop_row_test = []
    for index, row in df_test.iterrows():
        if index % 10000 == 0:
            print(index)
        qid = row['qid']
        pid = row['pid']
        relevancy = row['relevancy']
        if qid in embedding_dict and pid in embedding_dict:
            q = embedding_dict[qid]
            p = embedding_dict[pid]
            if q.shape == (50,) and p.shape == (50,):
                qp = np.hstack((q, p))
                x_test.append(qp)
                y_test.append([relevancy])
            else:
                drop_row_test.append(index)
        else:
            drop_row_test.append(index)

    np.save("x_train_task2.npy", x_train)
    np.save("y_train_task2.npy", y_train)
    np.save("x_test_task2.npy", x_test)
    np.save("y_test_task2.npy", y_test)

    df_train = df_train.drop(drop_row_train)
    df_test = df_test.drop(drop_row_test)
    df_train.to_csv('df_train_task2.csv')
    df_test.to_csv('df_test_task2.csv')


def evaluate_model(df, y_pred):
    df['predict_score'] = y_pred.tolist()

    df['relevancy'] = pd.to_numeric(df['relevancy'], downcast='integer')

    # calculate mAP
    map3_sum = 0
    map10_sum = 0
    map100_sum = 0
    map1000_sum = 0
    num = 0
    for qid, group in df.groupby('qid'):
        group3 = group.head(3)
        group10 = group.head(10)
        group100 = group.head(100)
        group1000 = group.head(1000)
        ap3 = average_precision_score(group3['relevancy'].values, group3['predict_score'].values)
        ap10 = average_precision_score(group10['relevancy'].values, group10['predict_score'].values)
        ap100 = average_precision_score(group100['relevancy'].values, group100['predict_score'].values)
        ap1000 = average_precision_score(group1000['relevancy'].values, group1000['predict_score'].values)

        map3_sum += ap3
        map10_sum += ap10
        map100_sum += ap100
        map1000_sum += ap1000
        num += 1
    map3 = map3_sum / num
    map10 = map10_sum / num
    map100 = map100_sum / num
    map1000 = map1000_sum / num

    # calculate NDCG
    ndcg3_sum = 0
    ndcg10_sum = 0
    ndcg100_sum = 0
    ndcg1000_sum = 0
    num = 0
    for qid, group in df.groupby('qid'):
        group3 = group.head(3)
        group10 = group.head(10)
        group100 = group.head(100)
        group1000 = group.head(1000)

        ndcg3 = ndcg_score(np.asarray([group3['relevancy'].values]), np.asarray([group3['predict_score'].values]))
        ndcg10 = ndcg_score(np.asarray([group10['relevancy'].values]), np.asarray([group10['predict_score'].values]))
        ndcg100 = ndcg_score(np.asarray([group100['relevancy'].values]), np.asarray([group100['predict_score'].values]))
        ndcg1000 = ndcg_score(np.asarray([group1000['relevancy'].values]), np.asarray([group1000['predict_score'].values]))
        ndcg3_sum += ndcg3
        ndcg10_sum += ndcg10
        ndcg100_sum += ndcg100
        ndcg1000_sum += ndcg1000
        num += 1
    mndcg3 = ndcg3_sum / num
    mndcg10 = ndcg10_sum / num
    mndcg100 = ndcg100_sum / num
    mndcg1000 = ndcg1000_sum / num

    print("mAP@3:", map3, " mAP@10:", map10, " mAP@100:", map100, " map@1000", map1000)
    print("NDCG@3:", mndcg3, " NDCG@10:", mndcg10, " NDCG@100:", mndcg100, "  NDCG@1000:", mndcg1000)


def main():
    # 1. load training/validation dataset
    df_train, df_validation = load_train_validation()
    embedding_dict = word_embedding()

    # 2. get query/passage embedding
    generate_embedding(df_train, df_validation, embedding_dict)

    # 3. load query/passage embedding file    query + passage embedding as x input,  relevancy as y input
    embedding_dict = load_embedding_pickle()
    df_train = df_train[['qid', 'pid', 'relevancy']]
    df_validation = df_validation[['qid', 'pid', 'relevancy']]
    process_model_input(df_train, df_validation, embedding_dict)

    # 4. load x/y_train as model input,  train model
    x_train = np.load("x_train_task2.npy", allow_pickle=True)
    print(x_train.shape)
    y_train = np.load("y_train_task2.npy", allow_pickle=True)
    print(y_train.shape)
    x_test = np.load("x_test_task2.npy", allow_pickle=True)
    print(x_test.shape)
    y_test = np.load("y_test_task2.npy", allow_pickle=True)
    print(y_test.shape)

    df_test = pd.read_csv('df_test_task2.csv')

    # lr=2
    print("start training")
    model6 = LogisticRegression1(learning_rate=2, n_iterations=1000)
    y_cost6 = model6.fit(x_train, y_train)
    cost6 = y_cost6.tolist()
    x6 = []
    y6 = []
    num = 0
    for i in cost6:
        if not math.isnan(i):
            num += 1
            y6.append(i)
            x6.append(num)
    print("start predicting")
    y_pred = model6.predict(x_test)
    print("start evaluating")
    print("learning rate = 2")
    evaluate_model(df_test, y_pred)

    plt.plot(x6, y6, label='lr = 2')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title("Model loss - Learning rate")
    plt.legend()
    plt.show()

    # lr=1
    print("start training")
    model1 = LogisticRegression1(learning_rate=1, n_iterations=1000)
    y_cost1 = model1.fit(x_train, y_train)
    cost1 = y_cost1.tolist()
    x1 = []
    y1 = []
    num = 0
    for i in cost1:
        if not math.isnan(i):
            num += 1
            y1.append(i)
            x1.append(num)
    print("start predicting")
    y_pred = model1.predict(x_test)
    print("start evaluating")
    print("learning rate = 1")
    evaluate_model(df_test, y_pred)

    # lr=0.1
    print("start training")
    model2 = LogisticRegression1(learning_rate=0.1, n_iterations=1000)
    y_cost2 = model2.fit(x_train, y_train)
    cost2 = y_cost2.tolist()
    y2 = []
    x2 = []
    num = 0
    for i in cost2:
        if not math.isnan(i):
            num += 1
            y2.append(i)
            x2.append(num)
    print("start predicting")
    y_pred = model2.predict(x_test)
    print("start evaluating")
    print("learning rate = 0.1")
    evaluate_model(df_test, y_pred)

    # lr=0.01
    print("start training")
    model3 = LogisticRegression1(learning_rate=0.01, n_iterations=1000)
    y_cost3 = model3.fit(x_train, y_train)
    cost3 = y_cost3.tolist()
    y3 = []
    x3 = []
    num = 0
    for i in cost3:
        if not math.isnan(i):
            num += 1
            y3.append(i)
            x3.append(num)
    print("start predicting")
    y_pred = model3.predict(x_test)
    print("start evaluating")
    print("learning rate = 0.01")
    evaluate_model(df_test, y_pred)

    # lr=0.001
    print("start training")
    model4 = LogisticRegression1(learning_rate=0.001, n_iterations=1000)
    y_cost4 = model4.fit(x_train, y_train)
    cost4 = y_cost4.tolist()
    y4 = []
    x4 = []
    num = 0
    for i in cost4:
        if not math.isnan(i):
            num += 1
            y4.append(i)
            x4.append(num)
    print("start predicting")
    y_pred = model4.predict(x_test)
    print("start evaluating")
    print("learning rate = 0.001")
    evaluate_model(df_test, y_pred)

    # lr=0.0001
    print("start training")
    model5 = LogisticRegression1(learning_rate=0.0001, n_iterations=1000)
    y_cost5 = model5.fit(x_train, y_train)
    cost5 = y_cost5.tolist()
    y5 = []
    x5 = []
    num = 0
    for i in cost5:
        if not math.isnan(i):
            num += 1
            y5.append(i)
            x5.append(num)
    print("start predicting")
    y_pred = model5.predict(x_test)
    print("start evaluating")
    print("learning rate = 0.0001")
    evaluate_model(df_test, y_pred)

    plt.plot(x1, y1, label='lr = 1')
    plt.plot(x2, y2, label='lr = 0.1')
    plt.plot(x3, y3, label='lr = 0.01')
    plt.plot(x4, y4, label='lr = 0.001')
    plt.plot(x5, y5, label='lr = 0.0001')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title("Model loss - Learning rate")
    plt.legend()
    plt.show()


main()