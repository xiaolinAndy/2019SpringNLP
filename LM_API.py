import pickle

S_gram_file = "./LM_results/S_gram.pkl"
Bi_gram_file = "./LM_results/Bi_gram.pkl"
Tri_gram_file = "./LM_results/Tri_gram.pkl"
STOP_TAG = '</s>'
ngram = 3


if ngram == 3:
    with open(Bi_gram_file, 'rb') as f:
        down_model = pickle.load(f)
    with open(Tri_gram_file, 'rb') as f:
        up_model = pickle.load(f)
elif ngram == 2:
    with open(Bi_gram_file, 'rb') as f:
        up_model = pickle.load(f)
    with open(S_gram_file, 'rb') as f:
        down_model = pickle.load(f)


def LM_score(x):
    if not isinstance(x, list):
        x = [x]
    scores = []
    for s in x:
        score = 1
        if STOP_TAG in s:
            s = s.replace(STOP_TAG, '')
            s = list(s) + [STOP_TAG]
        else:
            s = list(s)
        for i in range(len(s) - ngram + 1):
            up_word = ''.join(s[i:i + ngram])
            down_word = ''.join(up_word[1:])
            p = (up_model[up_word]+1)/(len(down_model)+down_model[down_word])
            score *= p
            # print(up_word, down_word, p)
        scores.append(score)
    return scores

