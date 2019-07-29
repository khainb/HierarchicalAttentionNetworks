import torch
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import get_pretrained_word_embedding,read_vocab
import numpy as np
from torch.utils.data import Dataset
class Custom_Dataset(Dataset):
    def __init__(self,texts,labels,word_to_ix, max_length_sentences=30, max_length_word=35):
        super(Custom_Dataset, self).__init__()
        self.word_to_ix=word_to_ix
        self.texts = texts
        self.labels = labels
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        document_encode = [
            [self.word_to_ix[word] if word in self.word_to_ix.keys() else 0 for word in word_tokenize(text=sentences.replace("\n"," ").replace("\\n"," "))]
            for sentences in sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [0 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[0 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)


        return document_encode.astype(np.int64), label-1
