import xgboost as xgb
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score, dcg_score
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle


def load_dataset():

    df_train = pd.read_csv('train_data.tsv', sep='\t', header=0)
    df_test = pd.read_csv('validation_data.tsv', sep='\t', header=0)

    return df_train, df_test


def sentence_embedding(training_data, testing_data):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    print(training_data.shape[0])
    sentence_dict = {}
    num = 0
    for index, row in training_data.iterrows():
        if num % 1000 == 0:
            print(num)
        qid = row['qid']
        pid = row['pid']
        query = row['queries']
        passage = row['passage']
        if qid not in sentence_dict:
            query_embeddings = model.encode(query)
            sentence_dict[qid] = query_embeddings
        if pid not in sentence_dict:
            passage_embeddings = model.encode(passage)
            sentence_dict[pid] = passage_embeddings
        num += 1

    print(testing_data.shape[0])
    num = 0
    for index, row in testing_data.iterrows():
        if num % 1000 == 0:
            print(num)
        qid = row['qid']
        pid = row['pid']
        query = row['queries']
        passage = row['passage']
        if qid not in sentence_dict:
            query_embeddings = model.encode(query)
            sentence_dict[qid] = query_embeddings
        if pid not in sentence_dict:
            passage_embeddings = model.encode(passage)
            sentence_dict[pid] = passage_embeddings
        num += 1

    with open('sentence_embedding_task3.pkl', 'wb') as file:
        pickle.dump(sentence_dict, file, protocol=-1)


def load_embedding_pickle():
    with open('sentence_embedding_task3.pkl', 'rb') as f:
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
        relevancy = float(row['relevancy'])
        if qid in embedding_dict and pid in embedding_dict:
            q = embedding_dict[qid]
            p = embedding_dict[pid]
            if q.shape == (384,) and p.shape == (384,):
                qp = np.hstack((q, p))
                re = [relevancy]
                x_train.append(qp)
                y_train.append(re)

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
        relevancy = float(row['relevancy'])
        if str(int(qid)) in embedding_dict and str(int(pid)) in embedding_dict:
            q = embedding_dict[str(int(qid))]
            p = embedding_dict[str(int(pid))]
            if q.shape == (384,) and p.shape == (384,):
                qp = np.hstack((q, p))
                re = [relevancy]
                x_test.append(qp)
                y_test.append(re)
            else:
                drop_row_test.append(index)
        else:
            drop_row_test.append(index)

    np.save("x_train_task3.npy", x_train)
    np.save("y_train_task3.npy", y_train)
    np.save("x_test_task3.npy", x_test)
    np.save("y_test_task3.npy", y_test)

    df_train = df_train.drop(drop_row_train)
    df_test = df_test.drop(drop_row_test)
    df_train.to_csv('df_train_task3.csv')
    df_test.to_csv('df_test_task3.csv')


def process_data():

    x_train = np.load("x_train_task3.npy", allow_pickle=True)
    y_train = np.load("y_train_task3.npy", allow_pickle=True)
    x_test = np.load("x_test_task3.npy", allow_pickle=True)
    y_test = np.load("y_test_task3.npy", allow_pickle=True)

    training_data = xgb.DMatrix(x_train, label=y_train)
    testing_data = xgb.DMatrix(x_test, label=y_test)
    return training_data, testing_data


def train_model(training_data, testing_data):

    # model parameters
    params = {
        # Parameters that we are going to tune.
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.3,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'rank:pairwise',
        'eval_metric': ['ndcg', 'map']
    }
    num_round = 44

    model = xgb.train(params, training_data, num_round)
    return model


def predict(model, testing_data):
    # predict
    preds = model.predict(testing_data)
    print(preds)

    return preds


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
        ndcg1000 = ndcg_score(np.asarray([group1000['relevancy'].values]),
                              np.asarray([group1000['predict_score'].values]))
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


def tune_parameter(dtrain, dtest):
    # param = {
    #     'max_depth': 6,
    #     'eta': 0.3,
    #     'objective': 'rank:pairwise',
    # }
    # num_round = 10

    # early_stopping_rounds
    params = {
        # Parameters that we are going to tune.
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.3,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'rank:pairwise',
        'eval_metric': ['ndcg', 'map']
    }
    num_boost_round = 99
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10
    )

    print(model.best_score, model.best_iteration+1)

    # max_depth  min_child_weight
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(6, 9)
        for min_child_weight in range(1, 4)
    ]
    max_ndcg = float("-Inf")
    max_map = float("-Inf")
    best_params = None

    for max_depth, min_child_weight in gridsearch_params:
        print("max_depth={}, min_child_weight={}".format(
            max_depth,
            min_child_weight))
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['ndcg', 'map'],
            early_stopping_rounds=10
        )
        mean_ndcg = cv_results['test-ndcg-mean'].max()
        boost_rounds = cv_results['test-ndcg-mean'].argmax()
        mean_map = cv_results['test-map-mean'].max()
        boost_rounds2 = cv_results['test-map-mean'].argmax()
        print("\tNDCG {} for {} rounds".format(mean_ndcg, boost_rounds))
        print("\tMAP {} for {} rounds".format(mean_map, boost_rounds2))
        if mean_ndcg > max_ndcg and mean_map > max_map:
            max_ndcg = mean_ndcg
            max_map = mean_map
            best_params = (max_depth, min_child_weight)

    print("{}, {}, NDCG: {}, MAP: {}".format(best_params[0], best_params[1], max_ndcg, max_map))


    # tune subsample, colsample
    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i / 10. for i in range(7, 11)]
        for colsample in [i / 10. for i in range(7, 11)]
    ]

    max_ndcg = float("-Inf")
    max_map = float("-Inf")
    best_params = None
    for subsample, colsample in reversed(gridsearch_params):
        print("subsample={}, colsample={}".format(
            subsample,
            colsample))
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['ndcg', 'map'],
            early_stopping_rounds=10
        )
        mean_ndcg = cv_results['test-ndcg-mean'].max()
        boost_rounds = cv_results['test-ndcg-mean'].argmax()
        mean_map = cv_results['test-map-mean'].max()
        boost_rounds2 = cv_results['test-map-mean'].argmax()
        print("\tNDCG {} for {} rounds".format(mean_ndcg, boost_rounds))
        print("\tMAP {} for {} rounds".format(mean_map, boost_rounds2))
        if mean_ndcg > max_ndcg and mean_map > max_map:
            max_ndcg = mean_ndcg
            max_map = mean_map
            best_params = (subsample, colsample)
    print("{}, {}, NDCG: {}, MAP: {}".format(best_params[0], best_params[1], max_ndcg, max_map))


    # tune eta
    max_ndcg = float("-Inf")
    max_map = float("-Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("eta={}".format(eta))
        params['eta'] = eta
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['ndcg', 'map'],
            early_stopping_rounds=10
        )
        mean_ndcg = cv_results['test-ndcg-mean'].max()
        boost_rounds = cv_results['test-ndcg-mean'].argmax()
        mean_map = cv_results['test-map-mean'].max()
        boost_rounds2 = cv_results['test-map-mean'].argmax()
        print("\tNDCG {} for {} rounds".format(mean_ndcg, boost_rounds))
        print("\tMAP {} for {} rounds".format(mean_map, boost_rounds2))
        if mean_ndcg > max_ndcg and mean_map > max_map:
            max_ndcg = mean_ndcg
            max_map = mean_map
            best_params = eta
    print("{}, NDCG: {}, MAP: {}".format(best_params, max_ndcg, max_map))


    # num
    params = {
        'max_depth': 6,
        'min_child_weight': 3,
        'eta': 0.3,
        'subsample': 0.9,
        'colsample_bytree': 1,
        'objective': 'rank:pairwise',
        'eval_metric': ['ndcg', 'map']
    }
    num_boost_round = 99
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10
    )

    print(model)
    print(model.best_score, model.best_iteration+1)



def main():
    training_data, testing_data = load_dataset()

    sentence_embedding(training_data, testing_data)

    embedding_dict = load_embedding_pickle()
    df_train = training_data[['qid', 'pid', 'relevancy']]
    df_test = testing_data[['qid', 'pid', 'relevancy']]
    process_model_input(df_train, df_test, embedding_dict)

    training_data, testing_data = process_data()

    tune_parameter(training_data, testing_data)

    model = train_model(training_data, testing_data)

    preds = predict(model, testing_data)

    df_test = pd.read_csv('df_test_task3.csv')
    evaluate_model(df_test, preds)


main()