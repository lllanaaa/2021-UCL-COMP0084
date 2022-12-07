import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RankModel(nn.Module):
    def __init__(self, num_features):
        super(RankModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.output = nn.Sigmoid()

    def forward(self, x1, x2):
        s1 = self.model(x1)
        s2 = self.model(x2)
        prob = self.output(s1 - s2)
        return prob

    def predict(self, input_):
        return self.model(input_)


def train_model():

    x1_train = np.load("x1_train_task4.npy", allow_pickle=True)
    x2_train = np.load("x2_train_task4.npy", allow_pickle=True)
    y_train = np.load("y_train_task4.npy", allow_pickle=True)
    x1_test = np.load("x1_test_task4.npy", allow_pickle=True)
    x2_test = np.load("x2_test_task4.npy", allow_pickle=True)
    y_test = np.load("y_test_task4.npy", allow_pickle=True)

    x1 = torch.from_numpy(x1_train)
    x2 = torch.from_numpy(x2_train)
    y = torch.from_numpy(y_train).float()

    ranknet = RankModel(num_features=384)
    optimizer = torch.optim.Adam(ranknet.parameters())
    criterion = nn.BCELoss()
    ranknet = ranknet.to(device)
    x1 = x1.to(device)
    x2 = x2.to(device)
    y = y.to(device)
    losses = []

    for epoch in range(500):
        outputs = ranknet(x1, x2)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plt.figure(figsize=(12, 7))
    plt.plot(range(len(losses)), losses, linewidth=2.5)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()


def load_embedding_pickle():
    with open('sentence_embedding_task3.pkl', 'rb') as f:
        embedding_dict = pickle.load(f)
    return embedding_dict


def process_model_input(df_train, df_test, embedding_dict):
    print(df_train.shape[0])
    x1_train = []
    x2_train = []
    y_train = []
    drop_row_train = []
    for index, row in df_train.iterrows():
        if index % 10000 == 0:
            print(index)
        qid = row['qid']
        pid = row['pid']
        relevancy = float(row['relevancy'])
        if qid in embedding_dict and pid in embedding_dict:
            q = embedding_dict[qid]
            p = embedding_dict[pid]
            if q.shape == (384,) and p.shape == (384,):
                re = [relevancy]
                x1_train.append(q)
                x2_train.append(q)
                y_train.append(re)
            else:
                drop_row_train.append(index)
        else:
            drop_row_train.append(index)

    print(df_test.shape[0])
    x1_test = []
    x2_test = []
    y_test = []
    drop_row_test = []
    for index, row in df_test.iterrows():
        if index % 10000 == 0:
            print(index)
        qid = row['qid']
        pid = row['pid']
        relevancy = float(row['relevancy'])
        if str(int(qid)) in embedding_dict and str(int(pid)) in embedding_dict:
            q = embedding_dict[str(int(qid))]
            p = embedding_dict[str(int(pid))]
            if q.shape == (384,) and p.shape == (384,):
                re = [relevancy]
                x1_test.append(q)
                x2_test.append(p)
                y_test.append(re)
            else:
                drop_row_test.append(index)
        else:
            drop_row_test.append(index)

    np.save("x1_train_task4.npy", x1_train)
    np.save("x2_train_task4.npy", x2_train)
    np.save("y_train_task4.npy", y_train)
    np.save("x1_test_task4.npy", x1_test)
    np.save("x2_test_task4.npy", x2_test)
    np.save("y_test_task4.npy", y_test)

    df_train = df_train.drop(drop_row_train)
    df_test = df_test.drop(drop_row_test)
    df_train.to_csv('df_train_task4.csv')
    df_test.to_csv('df_test_task4.csv')


df_train = pd.read_csv('train_data.tsv', sep='\t', header=0)
df_test = pd.read_csv('validation_data.tsv', sep='\t', header=0)
df_train = df_train[['qid', 'pid', 'relevancy']]
df_test = df_test[['qid', 'pid', 'relevancy']]
embedding_dict = load_embedding_pickle()
process_model_input(df_train, df_test, embedding_dict)


train_model()