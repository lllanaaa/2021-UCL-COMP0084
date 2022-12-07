import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from collections import Counter
import contractions
import re
import string
import unidecode
import math
import matplotlib.pyplot as plt
import numpy as np


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def load_dataset():
    # df_train = pd.read_csv('dataset/train_data.tsv', names=['qid', 'pid', 'query', 'passage', 'relevancy'])
    df_validation_query_passage = pd.read_csv('validation_data.tsv', sep='\t', header=0)
    df_validation_query = df_validation_query_passage.loc[:, ['qid', 'queries']]
    df_validation_query.drop_duplicates(subset=None, keep='first', inplace=True)
    df_validation_passage = df_validation_query_passage.loc[:, ['pid', 'passage']]
    df_validation_passage.drop_duplicates(subset=None, keep='first', inplace=True)
    return df_validation_query_passage, df_validation_query, df_validation_passage


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


def inverted_index(all_query, all_passage):
    vocal_dict = {}
    sum = 0
    for index, row in all_passage.iterrows():
        passage = row['passage']
        terms_in_passage = processDataRemoveStopword([passage])[0]
        term_num = len(terms_in_passage)
        sum += term_num
        terms_in_passage = list(set(terms_in_passage))
        for term in terms_in_passage:
            if term in vocal_dict:
                vocal_dict[term][0] += 1
            else:
                vocal_dict[term] = [1]

    for index, row in all_query.iterrows():
        query = row['queries']
        terms_in_query = processDataRemoveStopword([query])[0]
        for term in terms_in_query:
            if term not in vocal_dict:
                vocal_dict[term] = [0]

    N = all_passage.shape[0]
    R = 0
    r = 0
    average_passage_len = sum / N

    for key, value in vocal_dict.items():
        ni = value[0]
        bim = math.log(((r + 0.5) / (R - r + 0.5)) / ((ni - r + 0.5) / (N - ni - R + r + 0.5)))
        value.append(bim)

    # dict {'term': [ni, bim]}
    return vocal_dict, average_passage_len


def calculate_bm25_score(query, passage, inverted_index_dict, k1, k2, b, avdl):
    query_terms = processDataRemoveStopword([query])[0]
    passage_terms = processDataRemoveStopword([passage])[0]
    query_terms_counter = Counter(query_terms)
    passage_terms_counter = Counter(passage_terms)
    dl = len(passage_terms)
    K = k1 * ((1 - b) + (b * dl / avdl))

    value = 0
    for x in query_terms_counter:
        term = x
        qf = query_terms_counter[term]
        f = passage_terms_counter[term]
        v1 = inverted_index_dict[term][1]  # BIM log()
        v2 = ((k1 + 1) * f) / (K + f)
        v3 = ((k2 + 1) * qf) / (k2 + qf)
        value += v1 * v2 * v3
    return value


def load_bm25_model(df_validation_query_passage, df_validation_query, inverted_index_dict, average_passage_len):
    k1 = 1.2
    k2 = 100
    b = 0.75

    average_precision_array = []
    NDCG_array = []

    for index, row in df_validation_query.iterrows():
        res = []
        qid = row['qid']
        query = row['queries']
        passages = df_validation_query_passage.loc[df_validation_query_passage['qid'] == qid, ['pid', 'passage', 'relevancy']]
        for i, r in passages.iterrows():
            pid = r['pid']
            passage = r['passage']
            relevancy = r['relevancy']
            score = calculate_bm25_score(query, passage, inverted_index_dict, k1, k2, b, average_passage_len)
            row['score'] = score
            res.append([qid, pid, score, relevancy])

        bm_df = pd.DataFrame(res, columns=['qid', 'pid', 'score', 'relevancy']).sort_values(by='score', ascending=False, ignore_index=True)

        # rows = bm_df.shape[0]
        # arr_ap = []
        # for i in range(rows):
        #     arr_ap.append(calculate_average_precision(bm_df, i+1))
        # average_precision_array.append(arr_ap)

        precision3 = calculate_average_precision(bm_df, 3)
        precision10 = calculate_average_precision(bm_df, 10)
        precision100 = calculate_average_precision(bm_df, 100)
        average_precision_array.append([precision3, precision10, precision100])

        # arr_ndcg = []
        # for i in range(rows):
        #     arr_ndcg.append(calculate_NDCG(bm_df, i+1))
        # NDCG_array.append(arr_ndcg)

        NDCG3 = calculate_NDCG(bm_df, 3)
        NDCG10 = calculate_NDCG(bm_df, 10)
        NDCG100 = calculate_NDCG(bm_df, 100)
        NDCG_array.append([NDCG3, NDCG10, NDCG100])

    return average_precision_array, NDCG_array


def calculate_average_precision(bm_df, k):
    df = bm_df.iloc[:k]
    df_relevant = df.loc[df['relevancy'] == 1.0]
    sum = 0
    num = 0
    for index, row in df_relevant.iterrows():
        num += 1
        sum += num / (index + 1)
    if num == 0:
        return 0
    ap = sum / num
    return ap


def calculate_NDCG(bm_df, k):
    df = bm_df.iloc[:k]
    dcg = 0
    for index, row in df.iterrows():
        dcg += (pow(2, row['relevancy']) - 1) / math.log2(1 + (index + 1))

    idcg_df = bm_df.sort_values(by='relevancy', ascending=False, ignore_index=True).iloc[:k]
    idcg = 0
    for index, row in idcg_df.iterrows():
        value = (pow(2, row['relevancy']) - 1) / math.log2(1 + (index + 1))
        if value > 0:
            idcg += value

    return dcg / idcg



def main():
    df_validation_query_passage, df_validation_query, df_validation_passage = load_dataset()
    inverted_index_dict, average_passage_len = inverted_index(df_validation_query, df_validation_passage)

    average_precision_array, NDCG_array = load_bm25_model(df_validation_query_passage, df_validation_query, inverted_index_dict, average_passage_len)

    # # plot
    # x_k = np.array(range(1, 1001))
    # map = []
    # for i in range(1000):
    #     sum = 0
    #     num = 0
    #     for item in average_precision_array:
    #         if i <= len(item)-1:
    #             sum += item[i]
    #             num += 1
    #     value = sum / num
    #     map.append(value)
    # y_map = np.array(map)
    # mndcg = []
    # for i in range(1000):
    #     sum = 0
    #     num = 0
    #     for item in NDCG_array:
    #         if i <= len(item)-1:
    #             sum += item[i]
    #             num += 1
    #     value = sum / num
    #     mndcg.append(value)
    # y_mndcg = np.array(mndcg)
    #
    # plt.title("BM25 Evaluation using AP and NDCG metrics")
    # plt.plot(x_k, y_map, label='mean average precision')
    # plt.plot(x_k, y_mndcg, label='mean NDCG')
    # plt.xlabel('k')
    # plt.ylabel('value')
    # plt.legend()
    # plt.show()

    ap3 = 0
    ap10 = 0
    ap100 = 0
    size = len(average_precision_array)
    for ap in average_precision_array:
        ap3 += ap[0]
        ap10 += ap[1]
        ap100 += ap[2]
    map3 = ap3 / size
    map10 = ap10 / size
    map100 = ap100 / size

    dcg3 = 0
    dcg10 = 0
    dcg100 = 0
    for ndcg in NDCG_array:
        dcg3 += ndcg[0]
        dcg10 += ndcg[1]
        dcg100 += ndcg[2]
    ndcg3 = dcg3 / size
    ndcg10 = dcg10 / size
    ndcg100 = dcg100 / size

    print("mAP@3:", map3, " mAP@10:", map10, " mAP@100:", map100)
    print("NDCG@3:", ndcg3, " NDCG@10:", ndcg10, " NDCG@100:", ndcg100)


pd.set_option('display.width', None)
main()