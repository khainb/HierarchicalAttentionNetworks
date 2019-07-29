import torch
import torch.nn as nn
from WordLevelAttention import WordLevelAttention
from SentenceLevelAttention import SentenceLevelAttention

class HierarchicalAttention(nn.Module):
    def __init__(self,word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_emb,
                 max_sent_length, max_word_length):
        super(HierarchicalAttention,self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att = WordLevelAttention(word_emb=pretrained_emb, hidden_size=word_hidden_size)
        self.sent_att = SentenceLevelAttention(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state(batch_size)

    def _init_hidden_state(self, batch_size):
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.to(device)
            self.sent_hidden_state = self.sent_hidden_state.to(device)

    def forward(self, input):#input [batch_size,doclen,sent_len]

        output_list = []
        input = input.permute(1, 0, 2) #input [doclen,batch_size,sent_len]
        for i in input:
            output, self.word_hidden_state = self.word_att(i.permute(1, 0), self.word_hidden_state) # [1,batch_size,word_hidden_size]
            output_list.append(output)
        output = torch.cat(output_list, 0) #output [doclen,batch_size,word_hidden_size]
        output, self.sent_hidden_state = self.sent_att(output, self.sent_hidden_state) # [batch_size,num_classes]

        return output
