import csv
import pandas as pd
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


def loadQueryPassageData():
    file_path = "candidate-passages-top1000.tsv"
    tsv_data = pd.read_csv(file_path, sep='\t', names=['qid', 'pid', 'query', 'passage'])
    return tsv_data


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


def processData(all_passage, all_query):
    vocal_dict = {}

    for index, row in all_passage.iterrows():
        pid = row['pid']
        passage = row['passage']
        terms_in_passage = processDataRemoveStopword([passage])[0]
        for term in terms_in_passage:
            if term in vocal_dict:
                vocal_dict[term].append(pid)
            else:
                vocal_dict[term] = [pid]

    for index, row in all_query.iterrows():
        qid = row['qid']
        query = row['query']
        terms_in_query = processDataRemoveStopword([query])[0]
        for term in terms_in_query:
            if term not in vocal_dict:
                vocal_dict[term] = []

    with open('task2_invertedindex.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in vocal_dict.items():
            writer.writerow(i)


print("task2")
query_passage = loadQueryPassageData()
all_passage = query_passage[['pid', 'passage']]
all_passage.drop_duplicates(subset=['passage'], keep='first', inplace=True)
all_query = query_passage[['qid', 'query']]
all_query.drop_duplicates(subset=['query'], keep='first', inplace=True)

processData(all_passage, all_query)
