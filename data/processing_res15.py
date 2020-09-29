import _pickle as cPickle
import numpy as np
import re
import torch

idxs_all = cPickle.load(open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/idx_res16.train", "rb"))
labels_all = cPickle.load(open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/labels_res16.train", "rb"))
pos_all, _ = cPickle.load(open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/pos_res16.tag", "rb"))
words_all = cPickle.load(open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/words_res16.train", "rb"))

words_train = words_all[:1315]
words_test = words_all[1315:]
idxs_train = idxs_all[:1315]
idxs_test = idxs_all[1315:]
labels_train = labels_all[:1315]
labels_test = labels_all[1315:]
pos_train, pos_test = pos_all[:1315], pos_all[1315:]

cPickle.dump(words_train, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/words_res15.train", "wb"))
cPickle.dump(words_test, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/words_res15.test", "wb"))
cPickle.dump(idxs_train, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/idx_res15.train", "wb"))
cPickle.dump(idxs_test, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/idx_res15.test", "wb"))
cPickle.dump(labels_train, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/labels_res15.train", "wb"))
cPickle.dump(labels_test, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/labels_res15.test", "wb"))
cPickle.dump((pos_train, pos_test), open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/pos_res15.tag", "wb"))



dic_file = open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/w2v_yelp300_10.txt", "r")
dic = dic_file.readlines()
dic_file.close()

f_text_all = open("/Users/MissyLinlin/Desktop/FYP/py/deepmaxsat/data/Aspect-Opinion/sentence_res16", "r")
sentences_all = f_text_all.read().splitlines()
f_text_all.close()

sentences_train = sentences_all[:1315]
sentences_test = sentences_all[1315:]


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


cPickle.dump(embedding, open("/Users/MissyLinlin/Desktop/File_transfer/deepmaxsat/data/dms_asop/embedding300_res15", "wb"))
