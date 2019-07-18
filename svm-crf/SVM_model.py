import json, pickle
import random
from sklearn.model_selection import *
from sklearn.metrics import classification_report
from sklearn import svm

train_folder = './sighan8'
test_folder = './CLP14'

test_datafile = test_folder + "/test_transform.json"
test_result = test_folder + "/test_result.json"
train_datafile = train_folder + "/train_transform.json"

cluster_file = train_folder + "/cluster_dict.json"
embedding_file = train_folder + "/embedding.json"
model_file = train_folder + "/svm.pkl"

window = 0
NEGATIVE = 1
KMEANS_n = 512
UNK_CLASS = KMEANS_n+1


def getData(mode, ratio=0.8):

    def getfeature_w(sid, wid):
        word = data_dict[sid]['seg'][wid]
        c = cluster_dict[word] if word in cluster_dict else UNK_CLASS
        return [data_dict[sid]['pos'][wid],
                data_dict[sid]['len'][wid],
                c]

    def getfeature_s(sid, w):
        lens = len(data_dict[sid]['seg'])
        sample = []
        windows = [w+i for i in range(-window, window+1)]
        for j in windows:
            if j >= 0 and j < lens:
                sample += getfeature_w(id, j)
            else:
                sample += [0, 0, UNK_CLASS]
        sample += getfeature_w(id, w)
        return sample

    with open(cluster_file, 'r') as load_f:
        cluster_dict = json.load(load_f)

    if mode == 'Train':
        with open(train_datafile, 'r') as load_f:
            data_dict = json.load(load_f)

        train_x, train_y, test_x, test_y = [], [], [], []
        all = list(data_dict.keys())
        train_n = int(len(all)*ratio)
        train_set = all[:train_n] if ratio != 0 else []
        test_set = all[train_n:]

        for id in train_set:
            lens = len(data_dict[id]['seg'])
            for w in data_dict[id]['label']:
                train_x.append(getfeature_s(id, w))
                train_y.append(1)
            negative = []
            for i in range(int(len(data_dict[id]['label'])*NEGATIVE)):
                idx = random.randint(0, lens-1)
                if idx not in negative and idx not in data_dict[id]['label']:
                    train_x.append(getfeature_s(id, idx))
                    train_y.append(0)

        for id in test_set:
            lens = len(data_dict[id]['seg'])
            for w in range(lens):
                test_x.append(getfeature_s(id, w))
                if w in data_dict[id]['label']:
                    test_y.append(1)
                else:
                    test_y.append(0)
        return train_x, train_y, test_x, test_y

    elif mode == 'Test':
        with open(test_datafile, 'r') as load_f:
            data_dict = json.load(load_f)

        test_x, test_y, test_seq = [], [], {}
        for id in data_dict.keys():
            lens = len(data_dict[id]['seg'])
            test_seq[id] = lens
            for w in range(lens):
                test_x.append(getfeature_s(id, w))
                if w in data_dict[id]['label']:
                    test_y.append(1)
                else:
                    test_y.append(0)
        return test_x, test_y, test_seq


def train():
    X, Y, test_x, test_y = getData(mode='Train', ratio = 1)

    x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=1, train_size=0.9)
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_val)
    print(classification_report(y_val, y_pred, labels=[0, 1], target_names=['correct', 'incorrect']))

    # y_pred = clf.predict(test_x)
    # print(classification_report(test_y, y_pred, labels=[0, 1], target_names=['correct', 'incorrect']))

    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)


def test():
    test_x, test_y, seq_test = getData(mode='Test', ratio=0)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(test_x)

    n = len(seq_test)
    print(classification_report(test_y, y_pred, labels=[0, 1], target_names=['correct', 'incorrect']))

    results = dict()
    j = 0
    for id in seq_test:
        detect = []
        for i in range(seq_test[id]):
            if y_pred[j+i] == 1:
                detect.append(i)
        j = j+seq_test[id]
        results[id] = detect

    print('ave:', sum(y_pred) / n)

    with open(test_datafile, 'r') as load_f:
        data_dict = json.load(load_f)

    i = 0
    correct = 0
    for id in results:
        i += 1
        # print(results[id], data_dict[id]['label'])
        if set(data_dict[id]['label']) < set(results[id]):
            correct += 1
        # if i > 30:
        #     break
    print(correct/len(results))

    with open(test_result, "w") as f:
        json.dump(results, f)

    return results

# train()
result = test()
# print(result)