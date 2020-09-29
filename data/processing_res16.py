import _pickle as cPickle
import numpy as np
import re
import torch

with open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/token/uni_token_pos_resall16", "rb") as f:
    uni_token_pos_res_all = cPickle.load(f, encoding='utf-8')

uni_token_pos_res_train = uni_token_pos_res_all[0:2000]
uni_token_pos_res_test = uni_token_pos_res_all[2000:]

words_train = [[item[0] for item in uni] for uni in uni_token_pos_res_train]
words_test = [[item[0] for item in uni] for uni in uni_token_pos_res_test]

pos_train = [[item[1] for item in uni] for uni in uni_token_pos_res_train]
pos_test = [[item[1] for item in uni] for uni in uni_token_pos_res_test]

f_aspect_train = open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/Aspect-Opinion/aspectTerm_res16", "r")
f_aspect_test = open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/Aspect-Opinion/aspectTerm_restest16", "r")

f_opinion_train = open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/Aspect-Opinion/sentence_res_op_cat16", "r")
f_opinion_test = open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/Aspect-Opinion/sentence_restest_op_cat16", "r")

aspects_train = f_aspect_train.read().splitlines()
aspects_test = f_aspect_test.read().splitlines()
sentences_op_train = f_opinion_train.read().splitlines()
sentences_op_test = f_opinion_test.read().splitlines()

f_aspect_train.close()
f_aspect_test.close()
f_opinion_train.close()
f_opinion_test.close()

labels_train = []
labels_test = []

n = 0
for sentence_op, aspect in zip(sentences_op_train, aspects_train):
    #print(aspect)
    tokens = words_train[n]
    label = [0 for i in range(len(tokens))]
    if "##" in sentence_op:
        opinion = sentence_op.split("##")[1]
        # for res16 data
        opinion_list = opinion.split(",")[:-1]
        opinion_set = [item.split(" - ")[0] for item in opinion_list]
        # for laptop, res14
        #opinion_list = opinion.split(", ")
        #opinion_set = [' '.join(item.split(" ")[:-1]) for item in opinion_list]
        for op in opinion_set:
            op = op.strip()
            if " " not in op:
                for i, tok in enumerate(tokens):
                    if tok.lower() == op.lower():
                        label[i] = 3
            else:
                op = op.split()

                for i, tok in enumerate(tokens):
                    if tok.lower() == op[0].lower() and len(tokens) >= i + len(op):
                        match = True
                        for j in range(1, len(op)):
                            if tokens[i + j].lower() != op[j].lower():
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 3
                        for j in range(1, len(op)):
                            label[i + j] = 4

    if "NIL" not in aspect:
        aspect_list = aspect.split(",")
        for asp in aspect_list:
            asp = asp.strip()
            if " " not in asp:
                for i, tok in enumerate(tokens):
                    if tok == asp:
                        label[i] = 1
            else:
                asp = asp.split()

                for i, tok in enumerate(tokens):
                    if tok == asp[0] and len(tokens) >= i + len(asp):
                        match = True
                        for j in range(1, len(asp)):
                            if tokens[i + j] != asp[j]:
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 1
                        for j in range(1, len(asp)):
                            label[i + j] = 2

    labels_train.append(label)
    print(n)
    n += 1

n = 0
for sentence_op, aspect in zip(sentences_op_test, aspects_test):
    #print(aspect)
    tokens = words_test[n]
    label = [0 for i in range(len(tokens))]
    if "##" in sentence_op:
        opinion = sentence_op.split("##")[1]
        # for res16 data
        opinion_list = opinion.split(",")[:-1]
        opinion_set = [item.split(" - ")[0] for item in opinion_list]
         #for laptop, res14
        #opinion_list = opinion.split(", ")
        #opinion_set = [' '.join(item.split(" ")[:-1]) for item in opinion_list]
        for op in opinion_set:
            op = op.strip()
            if " " not in op:
                for i, tok in enumerate(tokens):
                    if tok.lower() == op.lower():
                        label[i] = 3
            else:
                op = op.split()

                for i, tok in enumerate(tokens):
                    if tok.lower() == op[0].lower() and len(tokens) >= i + len(op):
                        match = True
                        for j in range(1, len(op)):
                            if tokens[i + j].lower() != op[j].lower():
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 3
                        for j in range(1, len(op)):
                            label[i + j] = 4

    if "NIL" not in aspect:
        aspect_list = aspect.split(",")
        for asp in aspect_list:
            asp = asp.strip()
            if " " not in asp:
                for i, tok in enumerate(tokens):
                    if tok == asp:
                        label[i] = 1
            else:
                asp = asp.split()

                for i, tok in enumerate(tokens):
                    if tok == asp[0] and len(tokens) >= i + len(asp):
                        match = True
                        for j in range(1, len(asp)):
                            if tokens[i + j] != asp[j]:
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 1
                        for j in range(1, len(asp)):
                            label[i + j] = 2

    labels_test.append(label)
    print(n)
    n += 1

cPickle.dump((pos_train, pos_test), open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/pos_res16.tag", "wb"))
cPickle.dump(labels_train, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/labels_res16.train", "wb"))
cPickle.dump(labels_test, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/labels_res16.test", "wb"))
cPickle.dump(words_train, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/words_res16.train", "wb"))
cPickle.dump(words_test, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/words_res16.test", "wb"))


dic_file = open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/w2v_yelp300_10.txt", "r")
dic = dic_file.readlines()
dic_file.close()

f_text_train = open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/Aspect-Opinion/sentence_res16", "r")
f_text_test = open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/Aspect-Opinion/sentence_restest16", "r")
sentences_train = f_text_train.read().splitlines()
sentences_test = f_text_test.read().splitlines()
f_text_train.close()
f_text_test.close()

dictionary = {}
count = 0

for line in dic:
    word_vector = line.split(",")[:-1]
    vector_list = []
    for element in word_vector[len(word_vector) - 300:]:
        vector_list.append(float(element))
    word = ','.join(word_vector[:len(word_vector) - 300])

    vector = np.asarray(vector_list)
    dictionary[word] = vector

idxs_train = []
idxs_test = []
vocab = ["ppaadd", "punkt", "unk"]
e_pad = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
e_unk = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
e_punkt = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
embedding = [e_pad, e_punkt, e_unk]

n = 0
for sentence in sentences_train:
    tokens = words_train[n]
    idx = []
    for token in tokens:
        token = token.lower()
        if token.isalpha():
            if token not in vocab:
                if token in dictionary.keys():
                    embedding.append(dictionary[token])
                    vocab.append(token)
                    idx.append(vocab.index(token))
                else:
                    count += 1
                    embedding.append(np.asarray([(2 * np.random.rand() - 1) for i in range(300)]))
                    vocab.append(token)
                    idx.append(vocab.index(token))
            else:
                idx.append(vocab.index(token))

        else:
            idx.append(vocab.index("punkt"))

    idxs_train.append(idx)
    n += 1

n = 0
for sentence in sentences_test:
    tokens = words_test[n]
    idx = []
    for token in tokens:
        token = token.lower()
        if token.isalpha():
            if token not in vocab:
                if token in dictionary.keys():
                    embedding.append(dictionary[token])
                    vocab.append(token)
                    idx.append(vocab.index(token))
                else:
                    count += 1
                    idx.append(vocab.index("unk"))
            else:
                idx.append(vocab.index(token))

        else:
            idx.append(vocab.index("punkt"))

    idxs_test.append(idx)
    n += 1

embedding = np.asarray(embedding)


cPickle.dump(embedding, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/embedding300_res16", "wb"))
cPickle.dump(idxs_train, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/idx_res16.train", "wb"))
cPickle.dump(idxs_test, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/idx_res16.test", "wb"))

'''
emb = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/embedding300_res14", "rb"))
idxs_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/idx_res14.train", "rb"))
idxs_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/idx_res14.test", "rb"))
labels_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/labels_res14.train", "rb"))
labels_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/labels_res14.test", "rb"))
pos_train, pos_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/pos_res14.tag", "rb"))
words_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/words_res14.train", "rb"))
words_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/words_res14.test", "rb"))

'''

f_raw_parses_res_train = open('/Users/MissyLinlin/Desktop/File_transfer//deepmaxsat/data/Aspect-Opinion/raw_parses_res16')
f_raw_parses_res_test = open('/Users/MissyLinlin/Desktop/File_transfer//deepmaxsat/data/Aspect-Opinion/raw_parses_restest16')

parse_train = f_raw_parses_res_train.read().splitlines()
parse_test = f_raw_parses_res_test.read().splitlines()

f_raw_parses_res_train.close()
f_raw_parses_res_test.close()

parse_ls_train = [[]for i in range(len(idxs_train))]
i = 0
for k in range(len(parse_train)):
    txt = parse_train[k].replace('-', ',')
    txt = re.split(r'[(,)]', txt)
    if txt != ['']:
        word_1, word_2 = [x.strip() for x in txt if x.strip().isnumeric()]
        if word_1 != '0':
            if word_2 != '0':
                word_1_idx, word_2_idx = idxs_train[i][int(word_1)-1], idxs_train[i][int(word_2)-1]
                print(i, k, [word_1_idx, word_2_idx, txt[0]])
                parse_ls_train[i].append([word_1_idx, word_2_idx, txt[0]])
    else:
        i += 1

parse_ls_test = [[]for i in range(len(idxs_test))]
i = 0
for k in range(len(parse_test)):
    txt = parse_test[k].replace('-', ',')
    txt = re.split(r'[(,)]', txt)
    if txt != ['']:
        word_1, word_2 = [x.strip() for x in txt if x.strip().isnumeric()]
        if word_1 != '0':
            if word_2 != '0':
                word_1_idx, word_2_idx = idxs_test[i][int(word_1)-1], idxs_test[i][int(word_2)-1]
                print(i, k, [word_1_idx, word_2_idx, txt[0]])
                parse_ls_test[i].append([word_1_idx, word_2_idx, txt[0]])
    else:
        i += 1

cPickle.dump(parse_ls_train, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/parse_ls_res16.train", "wb"))
cPickle.dump(parse_ls_test, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/parse_ls_res16.test", "wb"))

'''
parse_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_ls_res14.train", "rb"))
parse_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_ls_res14.test", "rb"))
'''

f_raw_parses_res_train = open('/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/Aspect-Opinion/raw_parses_res16')
f_raw_parses_res_test = open('/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/Aspect-Opinion/raw_parses_restest16')

parse_train = f_raw_parses_res_train.read().splitlines()
parse_test = f_raw_parses_res_test.read().splitlines()

f_raw_parses_res_train.close()
f_raw_parses_res_test.close()

idxs_train = cPickle.load(open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/idx_res16.train", "rb"))
idxs_test = cPickle.load(open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/idx_res16.test", "rb"))
pos_train, pos_test = cPickle.load(open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/pos_res16.tag", "rb"))

parse_pos_train = [[]for i in range(len(idxs_train))]
i = 0
for k in range(len(parse_train)):
    txt = parse_train[k].replace('-', ',')
    txt = re.split(r'[(,)]', txt)
    if txt != ['']:
        word_1, word_2 = [x.strip() for x in txt if x.strip().isnumeric()]
        #if word_1 != '0' and word_2 != '0':
        #word_1_idx, word_2_idx = idxs_train[i][int(word_1)-1], idxs_train[i][int(word_2)-1]
        word_1_pos, word_2_pos = pos_train[i][int(word_1) - 1], pos_train[i][int(word_2) - 1]
        print(i, k, [word_1, word_1_pos, word_2, word_2_pos, txt[0]])
        #parse_pos_train[i].append([word_1_idx, word_1_pos, word_2_idx, word_2_pos, txt[0]])
        parse_pos_train[i].append([int(word_1) - 1, word_1_pos, int(word_2) - 1, word_2_pos, txt[0]])
    else:
        i += 1

parse_pos_test = [[]for i in range(len(idxs_test))]
i = 0
for k in range(len(parse_test)):
    txt = parse_test[k].replace('-', ',')
    txt = re.split(r'[(,)]', txt)
    if txt != ['']:
        word_1, word_2 = [x.strip() for x in txt if x.strip().isnumeric()]
        #if word_1 != '0' and word_2 != '0':
        #word_1_idx, word_2_idx = idxs_test[i][int(word_1)-1], idxs_test[i][int(word_2)-1]
        word_1_pos, word_2_pos = pos_test[i][int(word_1) - 1], pos_test[i][int(word_2) - 1]
        #print(i, k, [word_1_idx, word_1_pos, word_2_idx, word_2_pos, txt[0]])
        #parse_pos_test[i].append([word_1_idx, word_1_pos, word_2_idx, word_2_pos, txt[0]])
        parse_pos_test[i].append([int(word_1) - 1, word_1_pos, int(word_2) - 1, word_2_pos, txt[0]])
    else:
        i += 1

cPickle.dump(parse_pos_train, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/parse_pos_res16.train", "wb"))
cPickle.dump(parse_pos_test, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/parse_pos_res16.test", "wb"))

'''
parse_pos_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_pos_res14.train", "rb"))
parse_pos_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_pos_res14.test", "rb"))
'''

pos_train, pos_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/pos_res14.tag", "rb"))
parse_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_pos_res14.train", "rb"))
parse_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_pos_res14.test", "rb"))

pos_dic = ['pad']
for pos_line in pos_train:
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)

for pos_line in pos_test:
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)

parse_rel_dic = ['pad']
parse_ind_train, parse_ind_test = [], []
for parse_line in parse_train:
    parse_ind = []
    for parse in parse_line:
        print(parse)
        if parse[4] not in parse_rel_dic:
            parse_rel_dic.append(parse[4])
        parse_ind.append(torch.LongTensor([parse[0], pos_dic.index(parse[1]), parse[2], pos_dic.index(parse[3]), parse_rel_dic.index(parse[4])]))
    parse_ind_train.append(parse_ind)

for parse_line in parse_test:
    parse_ind = []
    for parse in parse_line:
        print(parse)
        if parse[4] not in parse_rel_dic:
            parse_rel_dic.append(parse[4])
        parse_ind.append(torch.LongTensor([parse[0], pos_dic.index(parse[1]), parse[2], pos_dic.index(parse[3]), parse_rel_dic.index(parse[4])]))
    parse_ind_test.append(parse_ind)

count = 0
for parse in parse_ind_train:
    if parse != []:
        parse_ind_train[count] = torch.stack(parse)
    else:
        parse_ind_train[count] = torch.zeros(5).long()
        print(count, parse)
    count += 1

count = 0
for parse in parse_ind_test:
    if parse != []:
        parse_ind_test[count] = torch.stack(parse)
    else:
        #parse_ind_test[count] = torch.zeros(5).long()
        parse_ind_test[count] = torch.LongTensor([0, len(pos_dic), 0,  len(pos_dic), len(parse_rel_dic)])
        print(count, parse)
    count += 1


cPickle.dump(parse_ind_train, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_ind_res14.train", "wb"))
cPickle.dump(parse_ind_test, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_ind_res14.test", "wb"))
cPickle.dump(parse_rel_dic, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_rel_dic_res14", "wb"))

'''
idxs_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/idx_res14.train", "rb"))
idxs_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/idx_res14.test", "rb"))
labels_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/labels_res14.train", "rb"))
labels_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/labels_res14.test", "rb"))
pos_train, pos_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/pos_res14.tag", "rb"))
parse_train = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_pos_res14.train", "rb"))
parse_test = cPickle.load(open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_pos_res14.test", "rb"))

idxs_train = [x for x in idxs_train if len(x) != 1]
idxs_test = [x for x in idxs_test if len(x) != 1]
labels_train = [x for x in labels_train if len(x) != 1]
labels_test = [x for x in labels_test if len(x) != 1]
pos_train = [x for x in pos_train if len(x) != 1]
pos_test = [x for x in pos_test if len(x) != 1]
parse_train = [x for x in parse_train if x != []]
parse_test = [x for x in parse_test if x != []]

cPickle.dump(idxs_train, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/idx_res14.train", "rb"))
cPickle.load(idxs_test, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/idx_res14.test", "rb"))
cPickle.load(labels_train, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/labels_res14.train", "rb"))
cPickle.load(labels_test, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/labels_res14.test", "rb"))
cPickle.load((pos_train, pos_test), open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/pos_res14.tag", "rb"))
cPickle.load(parse_train, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_pos_res14.train", "rb"))
cPickle.load(parse_test, open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/dms_asop/parse_pos_res14.test", "rb"))


'''


