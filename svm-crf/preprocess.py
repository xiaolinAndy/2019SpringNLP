import json

TRAIN = False
train_folder = './sighan8'
test_folder = './CLP14'

vocab_file = train_folder + "/word_dict.json"
pos_vocab_file = train_folder + "/pos_dict.json"

if TRAIN:
    transform_data_file = train_folder + "/train_transform.json"
    datafile = "./data/" + train_folder.replace('./', '') + "_seg_simple.json"
else:
    transform_data_file = test_folder + "/test_transform.json"
    if 'sighan8' in test_folder:
        datafile = "./data/" + test_folder.replace('./', '') + "_dev_seg_simple.json"
    else:
        datafile = "./data/"+test_folder.replace('./', '') + "_seg_simple.json"

embedding_file = train_folder + "/embedding.json"
cluster_file = train_folder + "/cluster_dict.json"

original_emb = "./sgns.renmin.bigram-char/sgns.renmin.bigram-char"
KMEANS_n = 512


'''
word_dict = {word:freq}
word_embed = {word:[embed]}
pos_dict = {pos:idx} 
word_class = {word:class}
'''


def preprocess(data):
    pos_dict = {}
    word_dict = {}
    transform_data = data.copy()
    idx = 1
    for id in data.keys():
        for i in range(len(data[id]['pos'])):
            p = data[id]['pos'][i]
            if p not in pos_dict:
                pos_dict[p] = idx
                idx += 1
            transform_data[id]['pos'][i] = pos_dict[p]
        for w in data[id]['seg']:
            if w in word_dict:
                word_dict[w] += 1
            else:
                word_dict[w] = 1
    with open(transform_data_file, "w") as f:
        json.dump(transform_data, f)
    with open(pos_vocab_file, "w") as f:
        json.dump(pos_dict, f)
        print('pos:', len(pos_dict))
    with open(vocab_file, "w") as f:
        json.dump(word_dict, f)
        print('word:',len(word_dict))

    word_emb = {}
    with open(original_emb, "rb") as f:
        line = f.readline()
        total = int(line.decode('utf-8').strip().replace('\n', '').split(' ')[0])
        print('original embedding file:', total)
        i = 0
        while line:
            i += 1
            tmp = line.decode('utf-8').strip().replace('\n', '').split(' ')
            if tmp[0] in word_dict:
                word_emb[tmp[0]] = [float(e) for e in tmp[1:]]
            if i % 50000 == 0:
                print(i/total)
            line = f.readline()
        print('uncovered:', 1-len(word_emb)/len(word_dict))
    with open(embedding_file, "w") as f:
        json.dump(word_emb, f)


def train_Kmeans():
    from sklearn.cluster import KMeans
    import numpy as np
    with open(embedding_file, "rb") as f:
        emb_dict = json.load(f)
    emb_train = []
    for w in emb_dict:
        emb_train.append(emb_dict[w])
    emb_train = np.array(emb_train)
    model = KMeans(n_clusters=KMEANS_n, random_state=0).fit(emb_train)
    print(model.labels_)
    cluster_dict = {}
    for c, w in zip(model.labels_.tolist(), emb_dict.keys()):
        cluster_dict[w] = c
    print(cluster_dict)
    with open(cluster_file, "w") as f:
        json.dump(cluster_dict, f)


def preprocess_test(data):
    with open(pos_vocab_file, 'r') as load_f:
        pos_dict = json.load(load_f)
    transform_data = data.copy()
    for id in data.keys():
        for i in range(len(data[id]['pos'])):
            p = data[id]['pos'][i]
            tag = 0 if p not in pos_dict else pos_dict[p]
            transform_data[id]['pos'][i] = tag
    with open(transform_data_file, "w") as f:
        json.dump(transform_data, f)


if __name__ == '__main__':
    with open(datafile, 'r') as load_f:
        load_data1 = json.load(load_f)
        print(len(load_data1))

    if TRAIN:
        preprocess(load_data1)
        train_Kmeans()
    else:
        preprocess_test(load_data1)

    # with open("./sighan8/embedding.json", 'r') as f:
    #     data = json.load(f)
    #     for id in data:
    #         print(len(data[id]))
    #         break
    # print(data)
