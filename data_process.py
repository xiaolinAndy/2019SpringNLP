import re
import jieba
#import thulac
import json
import jieba.posseg as pseg

from snownlp import SnowNLP
from argparse import ArgumentParser



def is_chinese(char):
    if char >= '\u4e00' and char <= '\u9fa5':
            return True
    else:
            return False

# format: dict {'id' : {'text': str, 'answer': [[pos, char], [pos, char]]}}
def process_data(text_name, gt_name, save_name, simple):
    data = {}
    with open(text_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            pattern = r'(NID=[0-9]+)'
            id = re.search(pattern, line).group(0)[-5:]
            text = ''.join(line.split()[1:])
            if simple:
                if len(text) != len(SnowNLP(text).han):
                    text = "想一想，台北也是我们每天在待的地方，每天在这个大都市来回穿梭，自由走动，但我们可能也不会到公车站牌旁的小巷子里有些什么，或许有一窝狗、猫，在天桥下，又有什么，或许有着一群游名，或者是一群热爱极限运动的舞者，脚踏车、滑板玩家，这些东西都等着我们去发现。"
                    print(text)
                else:
                    text = SnowNLP(text).han
            data[id] = {'text': text}
        assert len(data.items()) == 1000
    with open(gt_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(', ')
            answer = []
            # extract wrong word id and replaced word
            for i in range(1, len(line), 2):
                if simple:
                    line[i + 1] = SnowNLP(line[i+1]).han
                answer.append([int(line[i]), line[i+1]])
            data[line[0]]['answer'] = answer
        assert len(data.items()) == 1000
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(data, f)

# format: list [word1, word2, ...]
def process_dict(dict_path, save_path, simple):
    dict = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            if line[0][0] != '#' and line[0][0] != '%':
                if simple:
                    dict.append(line[1])
                else:
                    dict.append(line[0])
        print(len(dict))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f)

# format: dict {'char1': [char1-1, char1-2, ...], 'char2': [char2-1, ...], ...}
def process_cfs(cfs_pro_path, cfs_shape_path, save_path, simple):
    dict = {}
    with open(cfs_pro_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            key = line[0]
            value = ''.join(line[1:])
            try:
                if simple:
                    value = SnowNLP(value).han
            except:
                pass
            value = list(value)
            dict[key] = value
    with open(cfs_shape_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            key = line[0]
            value = ''.join(line[1:])
            if simple:
                value = SnowNLP(value).han
            value = list(value)
            if key in dict.keys():
                dict[key] += value
            else:
                dict[key] = value
        print(len(dict.items()))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f)

# format: dict {'id' : {'text': str, 'answer': [[pos, char], [pos, char]], 'seg': [a, b, c], 'pos': [n, v, adj], 'len': [2, 1, 2], 'label': [12, 23]}}
def data_seg(data_json, save_json):
    with open(data_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    near_count = 0
    for k,v in data.items():
        seg_list = jieba.lcut(v['text'], HMM=False)
        len_list = [len(word) for word in seg_list]
        words = pseg.cut(v['text'], HMM=False)
        pos = [word.flag for word in words]
        answer_index = [p[0] for p in v['answer']]
        answer_index = sorted(answer_index)
        for i, index in enumerate(answer_index):
            if i > 0 and answer_index[i] - answer_index[i-1] < 3:
                print(seg_list, v['answer'])
                near_count += 1
        label_index = []
        index = 1
        answer_p = 0
        for i, word in enumerate(seg_list):
            if index <= answer_index[answer_p] and index + len(word) > answer_index[answer_p]:
                label_index.append(i)
                answer_p += 1
                if answer_p == len(answer_index):
                    break
            index += len(word)

        data[k]['seg'] = seg_list
        data[k]['pos'] = pos
        data[k]['len'] = len_list
        data[k]['label'] = label_index
        assert len(seg_list) == len(pos)
        assert len(label_index) == len(answer_index)
    print(near_count)
    '''with open(save_json, 'w', encoding='utf-8') as f:
        json.dump(data, f)'''

def make_candidate(data, vocab_dict, cfs_dict, save_path, config):
    total_count, hit_count, cand_count, total_chars = 0., 0., 0., 0.
    # thu1 = thulac.thulac(seg_only=True)  # 默认模式
    # test for segment feasibility
    total_cand_count = 0
    for k, v in data.items():
        sample = v
        total_chars += len(sample['text'])
        if config.parser == 'jieba':
            seg_list = jieba.lcut(sample['text'], HMM=False)
            # seg_list = jieba.lcut(sample['text'])
        elif config.parser == 'thu':
            seg_list = thu1.cut(sample['text'], text=True).split()
        # print(seg_list)
        index = 1
        answer_index = [p[0] for p in sample['answer']]
        total_count += len(answer_index)
        candidates = []
        for i, word in enumerate(seg_list):
            # count hit mubers
            '''if len(word) == 1 and is_chinese(word):
                if index in anwser_index:
                    if i == 0:
                        if len(seg_list[i+1]) == 1:
                            hit_count += 1
                    elif i == len(seg_list)-1:
                        if len(seg_list[i-1]) == 1:
                            hit_count += 1
                    else:
                        if len(seg_list[i+1]) == 1 or len(seg_list[i-1]) == 1:
                            hit_count += 1
                if i == 0:
                    if len(seg_list[i+1]) == 1:
                        cand_count += 1
                elif i == len(seg_list)-1:
                    if len(seg_list[i-1]) == 1:
                        cand_count += 1
                else:
                    if len(seg_list[i+1]) == 1 or len(seg_list[i-1]) == 1:
                        cand_count += 1
            if len(word) == 2 and word not in vocab_dict:
                cand_count += 2
                if index in anwser_index or index+1 in anwser_index:
                    hit_count += 1
            if len(word) == 3 and word not in vocab_dict:
                cand_count += 3
                if index in anwser_index or index+1 in anwser_index or index+2 in anwser_index:
                    hit_count += 1'''
            # numerate all candidates
            if len(word) == 1 and is_chinese(word) and sample['text'][index - 1] in cfs_dict.keys():
                # two consecutive single characters
                if i == 0:
                    if len(seg_list[i + 1]) == 1:
                        candidates.append([index, cfs_dict[sample['text'][index - 1]]])
                        total_cand_count += len(cfs_dict[sample['text'][index - 1]])
                elif i == len(seg_list) - 1:
                    if len(seg_list[i - 1]) == 1:
                        candidates.append([index, cfs_dict[sample['text'][index - 1]]])
                        total_cand_count += len(cfs_dict[sample['text'][index - 1]])
                else:
                    if len(seg_list[i + 1]) == 1 or len(seg_list[i - 1]) == 1:
                        candidates.append([index, cfs_dict[sample['text'][index - 1]]])
                        total_cand_count += len(cfs_dict[sample['text'][index - 1]])
            if len(word) == 2 and word not in vocab_dict:
                if sample['text'][index - 1] in cfs_dict.keys():
                    candidate = []
                    for cand in cfs_dict[sample['text'][index - 1]]:
                        if cand + sample['text'][index] in vocab_dict:
                            total_cand_count += 1
                            candidate.append(cand)
                    candidates.append([index, candidate])
                if sample['text'][index] in cfs_dict.keys():
                    candidate = []
                    for cand in cfs_dict[sample['text'][index]]:
                        if sample['text'][index - 1] + cand in vocab_dict:
                            total_cand_count += 1
                            candidate.append(cand)
                    candidates.append([index + 1, candidate])
            if len(word) == 3 and word not in vocab_dict:
                if sample['text'][index - 1] in cfs_dict.keys():
                    candidate = []
                    for cand in cfs_dict[sample['text'][index - 1]]:
                        if cand + sample['text'][index] + sample['text'][index + 1] in vocab_dict:
                            total_cand_count += 1
                            candidate.append(cand)
                    candidates.append([index, candidate])
                if sample['text'][index] in cfs_dict.keys():
                    candidate = []
                    for cand in cfs_dict[sample['text'][index]]:
                        if sample['text'][index - 1] + cand + sample['text'][index + 1] in vocab_dict:
                            total_cand_count += 1
                            candidate.append(cand)
                    candidates.append([index+1, candidate])
                if sample['text'][index + 1] in cfs_dict.keys():
                    candidate = []
                    for cand in cfs_dict[sample['text'][index + 1]]:
                        if sample['text'][index - 1] + sample['text'][index] + cand in vocab_dict:
                            total_cand_count += 1
                            candidate.append(cand)
                    candidates.append([index+2, candidate])

            index += len(word)
        print(k)
        data[k]['cand'] = candidates
    print(total_cand_count)

    # format: dict {'id' : {'text': str, 'answer': [[1, 我], [2, 是]], 'cand': [[1, [我, 你]], [2, [没, 有]]]}}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    # print(hit_count, total_count, cand_count, total_chars, hit_count/total_count, hit_count/cand_count)

def get_lm_result(data):
    pass

def cal_metric(data, result):
    ctp, cfp, dtp, dfp, p = 0., 0., 0., 0., 0.
    for k,v in data.items():
        p += len(v['answer'])
        true_index = [ans[0] for ans in v['answer']]
        for cand in result[k]['res']:
            if cand in v['answer']:
                ctp += 1
            else:
                cfp += 1
            if cand[0] in true_index:
                dtp += 1
            else:
                dfp += 1
    cp = ctp/(ctp + cfp)
    cr = ctp/p
    cf = (2 * cp * cr)/(cp + cr)
    dp = dtp/(dtp + dfp)
    dr = dtp/p
    df = (2 * dp * dr)/(dp + dr)
    res = {'cp': cp, 'cr': cr, 'cf': cf, 'dp': dp, 'df': df, 'dr': dr}
    print(res)
    return res


def get_args():
    parser = ArgumentParser(description='chinese spelling check')
    parser.add_argument('--test_text', type=str, default='data/sighan7csc_release1.0/FinalTest_/FinalTest_SubTask2.txt')
    #parser.add_argument('--test_text', type=str, default='data/sighan8csc_release1.0/Test/SIGHAN15_CSC_TestInput.txt')
    parser.add_argument('--test_gt', type=str, default='data/sighan7csc_release1.0/FinalTest_/FinalTest_SubTask2_Truth.txt')
    parser.add_argument('--data_json', type=str,
                        default='data/sighan7_simple.json')
    parser.add_argument('--dict', type=str, default='data/cedict_ts.u8')
    parser.add_argument('--dict_json', type=str, default='data/chinese_dict_simple.json')
    parser.add_argument('--cfs_pro', type=str,
                        default='data/sighan7csc_release1.0/ConfusionSet/Bakeoff2013_CharacterSet_SimilarPronunciation.txt')
    parser.add_argument('--cfs_shape', type=str,
                        default='data/sighan7csc_release1.0/ConfusionSet/Bakeoff2013_CharacterSet_SimilarShape.txt')
    parser.add_argument('--cfs_dict', type=str, default='data/confusion_set_simple.json')
    parser.add_argument('--parser', type=str, default='jieba')
    parser.add_argument('--simple', type=bool, default=True)
    parser.add_argument('--data_seg_json', type=str, default='data/sighan7_seg_simple.json')
    parser.add_argument('--data_cand_json', type=str, default='data/sighan7_cand_simple.json')
    args = parser.parse_args()
    return args

def main():
    config = get_args()
    #process_data(config.test_text, config.test_gt, config.data_json, config.simple)
    #process_dict(config.dict, config.dict_json, config.simple)
    #process_cfs(config.cfs_pro, config.cfs_shape, config.cfs_dict, config.simple)
    #data_seg(config.data_json, config.data_seg_json)
    with open(config.cfs_dict, 'r', encoding='utf-8') as f:
        cfs_dict = json.load(f)
    choice_count = 0.
    '''for k,v in cfs_dict.items():
        choice_count += len(v)
    print(choice_count/len(cfs_dict.items()))'''
    with open(config.dict_json, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    print(len(vocab_dict), vocab_dict[15])
    with open(config.data_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    make_candidate(data, vocab_dict, cfs_dict, config.data_cand_json, config)
    # format: dict={'id': {'res': [[1, 我], [2, 你]]}}
    result = get_lm_result(config.data_cand_json)
    cal_metric(data, result)




if __name__ == '__main__':
    main()