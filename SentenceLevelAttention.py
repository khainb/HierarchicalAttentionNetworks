import torch
import torch.nn as nn
import numpy as np
from utils import mat_mul,element_wise_mul
import torch.nn.functional as F
class SentenceLevelAttention(nn.Module):
    def __init__(self,sent_hidden_size=50,word_hidden_size=50,num_classes=14):
        super(SentenceLevelAttention,self).__init__()
        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))
        self.gru= nn.GRU(2*word_hidden_size, sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):# input [Doc_len,batch_size,2*word_hidden_size]
        f_output, h_output = self.gru(input, hidden_state) # f_output [Doc_len,batch_size,2*sent_hidden_size]
        output = mat_mul(f_output,self.sent_weight,self.sent_bias) # out_put [Doc_len,batch_size,2*hidden_size]
        output = mat_mul(output,self.context_weight) # [Doc_len,batch_size]
        output = F.softmax(output.permute((1,0))) #[batch_size,Doc_len]
        output = element_wise_mul(f_output,output.permute((1,0))).squeeze(0) # [batch_size,2*sent_hidden_size]
        output = self.fc(output)# [batch_size,num_classes]

        return output, h_output

