import pandas as pd
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
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

    all_query = query_passage_data[['qid', 'query']]
    all_query.drop_duplicates(subset=['query'], keep='first', inplace=True)

    return query_passage_data, all_passage, all_passage_num


def loadAllQuery():
    file_path = "test-queries.tsv"
    all_query = pd.read_csv(file_path, sep='\t', names=['qid', 'query'])
    return all_query


def loadinvertedindex():
    file_path = "task2_invertedindex.csv"
    inverted_index = pd.read_csv(file_path, names=['term', 'value'], keep_default_na=False)
    inverted_index.set_index('term', drop=False, inplace=True)
    inverted_index = calTermOccurrence(inverted_index)
    return inverted_index


def processEachTerm(str):
    if str == '[]':
        arr = []
    else:
        arr = str[1: -1].split(',')
    return len(arr)


def calTermOccurrence(inverted_index):
    inverted_index['count'] = inverted_index.apply(lambda row: processEachTerm(row['value']), axis=1)
    return inverted_index


def calLaplaceSmoothing(query, passage, inverted_index):
    query_terms = processDataRemoveStopword([query])[0]
    query_terms = list(set(query_terms))
    passage_terms = processDataRemoveStopword([passage])[0]
    d_len = len(passage_terms)
    vocal_size = inverted_index.shape[0]
    value = 0
    for term in query_terms:
        f = 0
        for t in passage_terms:
            if t == term:
                f += 1
        v = math.log((f + 1) / (d_len + vocal_size))
        value += v
    return value


def laplaceSmoothing(all_query, query_passage_data, inverted_index):

    laplace_smoothing_df = pd.DataFrame()
    for index, row in all_query.iterrows():
        res = []
        qid = row['qid']
        query = row['query']
        passages = query_passage_data.loc[query_passage_data['qid'] == qid, ['pid', 'passage']]
        for i, r in passages.iterrows():
            pid = r['pid']
            passage = r['passage']
            value = calLaplaceSmoothing(query, passage, inverted_index)
            res.append([qid, pid, value])

        one_query_df = pd.DataFrame(res).sort_values(by=2, ascending=False).iloc[:100]
        laplace_smoothing_df = laplace_smoothing_df.append(one_query_df)

    laplace_smoothing_df.to_csv('laplace.csv', header=False, index=False)


def calLidstoneCorrection(query, passage, inverted_index, x):
    query_terms = processDataRemoveStopword([query])[0]
    query_terms = list(set(query_terms))
    passage_terms = processDataRemoveStopword([passage])[0]
    d_len = len(passage_terms)
    vocal_size = inverted_index.shape[0]
    value = 0
    for term in query_terms:
        f = 0
        for t in passage_terms:
            if t == term:
                f += 1
        v = math.log((f + x) / (d_len + vocal_size * x))
        value += v
    return value


def lidstoneCorrection(all_query, query_passage_data, inverted_index):
    x = 0.1
    lidstone_correction_df = pd.DataFrame()
    for index, row in all_query.iterrows():
        res = []
        qid = row['qid']
        query = row['query']
        passages = query_passage_data.loc[query_passage_data['qid'] == qid, ['pid', 'passage']]
        for i, r in passages.iterrows():
            pid = r['pid']
            passage = r['passage']
            value = calLidstoneCorrection(query, passage, inverted_index, x)
            res.append([qid, pid, value])

        one_query_df = pd.DataFrame(res).sort_values(by=2, ascending=False).iloc[:100]
        lidstone_correction_df = lidstone_correction_df.append(one_query_df)

    lidstone_correction_df.to_csv('lidstone.csv', header=False, index=False)


def processLen(passage):
    passage_terms = processDataRemoveStopword([passage])[0]
    return len(passage_terms)


def calAllTermNum(all_passage):
    all_passage['len'] = all_passage.apply(lambda row: processLen(row['passage']), axis=1)
    sum = all_passage['len'].sum()
    return sum


def calDirichletSmoothing(query, passage, inverted_index_dict, x, all_terms_num):  #
    query_terms = processDataRemoveStopword([query])[0]
    query_terms = list(set(query_terms))
    passage_terms = processDataRemoveStopword([passage])[0]
    d_len = len(passage_terms)
    value = 0
    for term in query_terms:
        f = 0
        for t in passage_terms:
            if t == term:
                f += 1
        c = inverted_index_dict[term]['count']
        if c == 0 and f == 0:
            return None
        v = math.log( (d_len / (d_len + x)) * (f / d_len) + (x / (d_len + x)) * (c / all_terms_num) )
        value += v
    return value


def dirichletSmoothing(all_query, all_passage, query_passage_data, inverted_index):
    x = 50
    inverted_index_dict = inverted_index[['term', 'count']].to_dict(orient='index')
    all_terms_num = calAllTermNum(all_passage)
    dirichlet_smoothing_df = pd.DataFrame()
    for index, row in all_query.iterrows():
        res = []
        qid = row['qid']
        query = row['query']
        passages = query_passage_data.loc[query_passage_data['qid'] == qid, ['pid', 'passage']]
        for i, r in passages.iterrows():
            pid = r['pid']
            passage = r['passage']
            value = calDirichletSmoothing(query, passage, inverted_index_dict, x, all_terms_num)
            res.append([qid, pid, value])

        one_query_df = pd.DataFrame(res).sort_values(by=2, ascending=False).iloc[:100]
        dirichlet_smoothing_df = dirichlet_smoothing_df.append(one_query_df)

    dirichlet_smoothing_df.to_csv('dirichlet.csv', header=False, index=False)


print('task4')
query_passage_data, all_passage, passage_num = loadQueryPassage()
all_query = loadAllQuery()
inverted_index = loadinvertedindex()

laplaceSmoothing(all_query, query_passage_data, inverted_index)

lidstoneCorrection(all_query, query_passage_data, inverted_index)

dirichletSmoothing(all_query, all_passage, query_passage_data, inverted_index)







