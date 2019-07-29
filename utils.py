import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from sklearn import metrics
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
def mat_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])

        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    result= torch.cat(feature_list, 0).squeeze()
    return result
def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def get_pretrained_word_embedding(filename,vocab):
    words = pd.read_table(filename, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    words=words[words.index.isin(vocab)]
    np_words = np.array(words)
    word_embedding = np.concatenate((np.zeros((1, np_words.shape[1])), np_words), axis=0)
    word_to_ix = {word: idx + 1 for idx, word in enumerate(list(words.index))}
    return word_embedding, word_to_ix
def read_vocab(file_path):
    with open(file_path) as f:
        vocab = f.read().splitlines()
    return vocab
def gen_vocab(train_csv):
    df=pd.read_csv(train_csv,dtype=str,names=['label','text'])

    texts = np.array(df['text'])
    vocab=[]
    for text in texts:
        for w in word_tokenize(text.replace("\n"," ").replace("\\n"," ")):
            vocab.append(w)
    vocab =list(set(vocab))
    with open(train_csv+'.txt','w') as f:
        f.write("\n".join(vocab))
    return list(set(vocab))
def get_max_lengths(data_path):
    df = pd.read_csv(data_path, names=['label', 'text'])
    texts = np.array(df['text'])
    len_sent=[]
    len_word=[]
    for text in texts:
        sents = sent_tokenize(text.replace("\n"," ").replace("\\n"," "))
        len_sent.append(len(sents))
        for sen in sents:
            len_word.append(len(word_tokenize(sen)))
    sorted_word_length = sorted(len_word)
    sorted_sent_length = sorted(len_sent)
    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]
def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output
# word_embedding, word_to_ix = get_pretrained_word_embedding('glove/glove.6B/glove.6B.100d.txt')
# print word_to_ix.keys()
# print gen_vocab('data/yelp_review_full_csv/train.csv')
