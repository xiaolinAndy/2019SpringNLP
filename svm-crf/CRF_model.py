import tensorflow as tf
import pickle
from getFeature import *

Mode = 'Train'


class CRF(object):
    def __init__(self, n_steps, input_size, num_tags, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.num_tags = num_tags
        self.batch_size = batch_size

        self.xs = tf.placeholder(tf.float32, [None, self.n_steps, self.input_size], name='xs')
        self.ys = tf.placeholder(tf.int32, [self.batch_size, self.n_steps], name='ys')
        self.sequence_lengths = tf.placeholder(tf.int32, [self.batch_size], name='len')
        # 将输入 batch_size x seq_length x input_size   映射到 batch_size x seq_length x num_tags

        weights = tf.get_variable("weights", [self.input_size, self.num_tags])
        matricized_x_t = tf.reshape(self.xs, [-1, self.input_size])
        matricized_unary_scores = tf.matmul(matricized_x_t, weights)
        unary_scores = tf.reshape(matricized_unary_scores, [self.batch_size, self.n_steps, self.num_tags])

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, self.ys, self.sequence_lengths)

        self.cost = tf.reduce_mean(-log_likelihood)
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

        self.pred, self.viterbi_score = tf.contrib.crf.crf_decode(unary_scores, transition_params,
                                                            self.sequence_lengths)


def get_batch(is_CUT = True):
    x_features_transform, y_label, seq_len, n = get_feature(is_CUT)
    return [np.array(x_features_transform), np.array(y_label), np.array(seq_len), n]


# 训练 CRF
if __name__ == '__main__':

    if Mode == 'Train':
        xs, res, seq_len, n = get_batch(False)  # 提取 batch data
        batch_num = int(n/BATCH_SIZE)

        xs_all, res_all, seq_len_all, n = get_batch(False)
        val_xs, val_res, val_seqlen = xs_all[-BATCH_SIZE:], res_all[-BATCH_SIZE:], seq_len_all[-BATCH_SIZE:]

        model = CRF(MAXLEN, INPUT_SIZE, num_tags, BATCH_SIZE)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # 训练多次
        for i in range(50):
            for j in range(batch_num-1):
                feed_dict = {
                    model.xs: xs[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                    model.ys: res[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                    model.sequence_lengths: seq_len[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                }
                _, cost, pred = sess.run(
                    [model.train_op, model.cost, model.pred],
                    feed_dict=feed_dict)

            if i % 20 == 0:
                print('cost: ', round(cost, 4))

        print('validate: ')
        feed_dict = {
            model.xs: val_xs,
            model.ys: val_res,
            model.sequence_lengths: val_seqlen
        }

        _, cost, pred = sess.run(
            [model.train_op, model.cost, model.pred],
            feed_dict=feed_dict)

        pred_y = [e for l, p in zip(val_seqlen, pred) for e in p[:l]]
        label_y = [e for l, p in zip(val_seqlen, val_res) for e in p[:l]]
        from sklearn.metrics import classification_report
        print(classification_report(label_y, pred_y, labels=[0, 1], target_names=['correct', 'incorrect']))

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

    if Mode == 'Test':
        '''
        Test code 
        '''