import pandas as pd
import math
from numpy import dot
from numpy.linalg import norm
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


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


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
                processed = stemmer.stem(processed)
                arr.append(processed)

        processedData.append(arr)
    return processedData


def loadQueryPassage():
    file_path = "candidate-passages-top1000.tsv"
    query_passage_data = pd.read_csv(file_path, sep='\t', names=['qid', 'pid', 'query', 'passage'])

    all_passage = query_passage_data[['pid', 'passage']]
    all_passage.drop_duplicates(subset=['passage'], keep='first', inplace=True)
    all_passage_num = all_passage.shape[0]

    return query_passage_data, all_passage, all_passage_num


def loadAllQuery():
    file_path = "test-queries.tsv"
    all_query = pd.read_csv(file_path, sep='\t', names=['qid', 'query'])
    return all_query


def loadinvertedindex():
    file_path = "task2_invertedindex.csv"
    tsv_data = pd.read_csv(file_path, names=['term', 'value'], keep_default_na=False)
    tsv_data.set_index('term', drop=False, inplace=True)
    return tsv_data


def processValue(str):
    if str == '[]':
        return []
    return str[1: -1].split(',')


def calArrLen(arr):
    arr = list(set(arr))
    return len(arr)


def calItemIDF(nt, num):
    return math.log10(num / float(nt))


def calIDF(inverted_index, num):
    inverted_index['value1'] = inverted_index.apply(lambda row: processValue(row['value']), axis=1)
    inverted_index['nt'] = inverted_index.apply(lambda row: calArrLen(row['value1']), axis=1)
    inverted_index['idf'] = inverted_index.apply(lambda row: calItemIDF(row['nt']+1, num), axis=1)
    dict = inverted_index[['term', 'idf']].to_dict(orient='index')
    return dict


def calStringDict(s):
    terms = processDataRemoveStopword([s])[0]
    term_dict = {i: terms.count(i) for i in terms}
    return term_dict


def cosine_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def getAllTermsInQueryPassage(query, passage):
    terms = processDataRemoveStopword([passage, query])
    term_passage = terms[0]
    term_query = terms[1]
    all_terms = term_passage + term_query
    all_terms = list(set(all_terms))
    return all_terms


def calArr(query, passage, idf_dict):
    query_terms = processDataRemoveStopword([query])[0]
    query_term_num = len(query_terms)
    query_term_dict = {i: query_terms.count(i) for i in query_terms}
    passage_terms = processDataRemoveStopword([passage])[0]
    passage_term_num = len(passage_terms)
    passage_term_dict = {i: passage_terms.count(i) for i in passage_terms}

    query_arr = []
    passage_arr = []
    all_terms = list(set(query_terms+passage_terms))
    for term in all_terms:
        if term in query_term_dict:
            query_arr.append((query_term_dict[term] / query_term_num) * idf_dict[term]['idf'])
        else:
            query_arr.append(0)
        if term in passage_term_dict:
            passage_arr.append((passage_term_dict[term] / passage_term_num) * idf_dict[term]['idf'])
        else:
            passage_arr.append(0)
    return query_arr, passage_arr


def calItemBim(ni, N, R, r):
    value = math.log(((r + 0.5) / (R - r + 0.5)) / ((ni - r + 0.5) / (N - ni - R + r + 0.5)))
    return value


def calBim(inverted_index, N, R, r):
    inverted_index['value2'] = inverted_index.apply(lambda row: processValue(row['value']), axis=1)
    inverted_index['ni'] = inverted_index.apply(lambda row: calArrLen(row['value2']), axis=1)
    inverted_index['bim'] = inverted_index.apply(lambda row: calItemBim(row['ni'], N, R, r), axis=1)
    dict = inverted_index[['term', 'bim']].to_dict(orient='index')
    return dict


def processLen(passage):
    passage_terms = processDataRemoveStopword([passage])[0]
    return len(passage_terms)


def calAvePassageLen(all_passage):
    all_passage['len'] = all_passage.apply(lambda row: processLen(row['passage']), axis=1)
    num = all_passage.shape[0]
    sum = all_passage['len'].sum()
    return sum / num


def calBM25(query, passage, inverted_index_bim_dict, k1, k2, b, avdl):
    query_terms = processDataRemoveStopword([query])[0]
    query_terms_counter = Counter(query_terms)
    passage_terms = processDataRemoveStopword([passage])[0]
    passage_terms_counter = Counter(passage_terms)
    dl = len(passage_terms)
    K = k1 * ((1 - b) + (b * dl / avdl))

    value = 0
    for x in query_terms_counter:
        term = x
        qf = query_terms_counter[term]
        f = passage_terms_counter[term]
        v1 = inverted_index_bim_dict[term]['bim']
        v2 = ((k1 + 1) * f) / (K + f)
        v3 = ((k2 + 1) * qf) / (k2 + qf)
        value += v1 * v2 * v3
    return value


def tfidf(inverted_index, passage_num, all_query, query_passage_data):
    inverted_index_df = inverted_index
    inverted_index_idf_dict = calIDF(inverted_index_df, passage_num)

    tfidf_df = pd.DataFrame()
    for index, row in all_query.iterrows():
        res = []
        qid = row['qid']
        query = row['query']
        passages = query_passage_data[query_passage_data['qid'] == qid][['pid', 'passage']]
        for i, r in passages.iterrows():
            pid = r['pid']
            passage = r['passage']
            query_arr, passage_arr = calArr(query, passage, inverted_index_idf_dict)
            sim = cosine_similarity(query_arr, passage_arr)
            row['similarity'] = sim
            res.append([qid, pid, sim])

        sim_df = pd.DataFrame(res).sort_values(by=2, ascending=False).iloc[:100]
        tfidf_df = tfidf_df.append(sim_df)

    tfidf_df.to_csv('tfidf.csv', header=False, index=False)


def bm25(inverted_index, passage_num, all_query, query_passage_data, all_passage):
    R = 0
    r = 0
    N = passage_num
    k1 = 1.2
    k2 = 100
    b = 0.75

    inverted_index_df = inverted_index
    inverted_index_bim_dict = calBim(inverted_index_df, N, R, r)
    avePassageLen = calAvePassageLen(all_passage)

    bm25_df = pd.DataFrame()
    for index, row in all_query.iterrows():
        res = []
        qid = row['qid']
        query = row['query']
        passages = query_passage_data.loc[query_passage_data['qid'] == qid, ['pid', 'passage']]
        for i, r in passages.iterrows():
            pid = r['pid']
            passage = r['passage']
            value = calBM25(query, passage, inverted_index_bim_dict, k1, k2, b, avePassageLen)
            row['bm25'] = value
            res.append([qid, pid, value])

        bm_df = pd.DataFrame(res).sort_values(by=2, ascending=False).iloc[:100]
        bm25_df = bm25_df.append(bm_df)

    bm25_df.to_csv('bm25.csv', header=False, index=False)


print("task3")
query_passage_data, all_passage, passage_num = loadQueryPassage()
all_query = loadAllQuery()
inverted_index = loadinvertedindex()

# tf idf
tfidf(inverted_index, passage_num, all_query, query_passage_data)

# BM25
bm25(inverted_index, passage_num, all_query, query_passage_data, all_passage)
