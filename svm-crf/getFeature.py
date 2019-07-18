import json
import numpy as np

folder = './CLP14'
cluster_file = folder + "/cluster_dict.json"
transform_data_file = folder + "/train_transform.json"
model_file = folder + '/CRF.pkl'

MAXLEN = 150     # backpropagation through time 的time_steps
BATCH_SIZE = 50
INPUT_SIZE = 3   # x数据输入size
LR = 0.05        # learning rate
num_tags = 2
KMEANS_n = 512
UNK_CLASS = KMEANS_n+1


def get_feature(is_CUT = True):
    with open(transform_data_file, 'r') as load_f:
        data_dict = json.load(load_f)
    with open(cluster_file, 'r') as load_f:
        cluster_dict = json.load(load_f)

    n = len(data_dict)
    sequence_length = []
    x_features = []   # [length, pos, class]
    y_label = []

    for id in data_dict.keys():

        seg = data_dict[id]['seg']
        pos_seq = np.array(data_dict[id]['pos'])
        len_seq = np.array(data_dict[id]['len'], dtype=int)
        class_seq = np.array([cluster_dict[w] if w in cluster_dict else UNK_CLASS for w in seg], dtype=int)
        y = [0] * len(seg)

        if len(data_dict[id]['label']) != 0:
            for idx in data_dict[id]['label']:
                y[idx] = 1

            wrong_begin = min(data_dict[id]['label'])
            wrong_end = wrong_begin+1
            if is_CUT and wrong_end > wrong_begin:
                wrong_end += 3
                wrong_begin -= 2
                wrong_begin = 0 if wrong_begin<0 else wrong_begin
                wrong_end = len(seg)-1 if wrong_end>= len(seg) else wrong_end
                len_seq, pos_seq, class_seq = len_seq[wrong_begin:wrong_end], \
                                              pos_seq[wrong_begin:wrong_end], \
                                              class_seq[wrong_begin:wrong_end]
                y = y[wrong_begin:wrong_end]
                sequence_length.append(wrong_end-wrong_begin)
            else:
                sequence_length.append(len(seg))
        else:
            sequence_length.append(len(seg))

        f = np.stack((len_seq, pos_seq, class_seq), axis=1)
        x_features.append(f.tolist())
        y_label.append(y)

    x_features_transform = np.zeros((n, MAXLEN, INPUT_SIZE), dtype='int')
    y_label_transform = np.zeros((n, MAXLEN), dtype='int')
    idx = list(range(n))
    for id, feature, res, l in zip(idx, x_features, y_label, sequence_length):
        x_features_transform[id, :l, :] = feature
        y_label_transform[id,:l] = res
    return x_features_transform, y_label_transform, sequence_length, n


'''
Test:
    x_features_transform, y_label,seq_len, n = get_feature()
    x = np.array(x_features_transform)
    y = np.array(y_label)
    seq_len = np.array(seq_len)
    print(x.shape)
    print(y.shape)
    print(seq_len.shape)
'''